/* Copyright (c) 2026 Adithya Venkatarao
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "rendering/slang_shader_compiler.h"

#include <slang/slang.h>
#include <slang/slang-com-ptr.h>

#include "common/logger.h"
#include "core/device.h"

namespace vulkr
{

struct SlangShaderCompiler::Impl
{
	Slang::ComPtr<slang::IGlobalSession> globalSession;
	Slang::ComPtr<slang::ISession> session;

	slang::TargetDesc targetDesc{};
	slang::CompilerOptionEntry emitSpirvDirectly{};
};

SlangShaderCompiler::SlangShaderCompiler() :
	m_impl{ std::make_unique<Impl>() }
{
	slang::createGlobalSession(m_impl->globalSession.writeRef());

	// Emit SPIR-V directly instead of round tripping through the GLSL backend
	m_impl->emitSpirvDirectly.name = slang::CompilerOptionName::EmitSpirvDirectly;
	m_impl->emitSpirvDirectly.value.kind = slang::CompilerOptionValueKind::Int;
	m_impl->emitSpirvDirectly.value.intValue0 = 1;

	// Pick the highest SPIR-V revision the engine's Vulkan API version supports
	const char *spirvProfileName = getSpirvProfileName(vulkanApiVersion);
	m_impl->targetDesc.format = SLANG_SPIRV;
	m_impl->targetDesc.profile = m_impl->globalSession->findProfile(spirvProfileName);
	if (m_impl->targetDesc.profile == SLANG_PROFILE_UNKNOWN)
	{
		LOGEANDABORT("The Slang installation does not provide the {} profile", spirvProfileName);
	}

	slang::SessionDesc sessionDesc{};
	sessionDesc.targets = &m_impl->targetDesc;
	sessionDesc.targetCount = 1;
	sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
	sessionDesc.compilerOptionEntries = &m_impl->emitSpirvDirectly;
	sessionDesc.compilerOptionEntryCount = 1;

	if (SLANG_FAILED(m_impl->globalSession->createSession(sessionDesc, m_impl->session.writeRef())))
	{
		LOGEANDABORT("Failed to create a Slang session");
	}
}

SlangShaderCompiler::~SlangShaderCompiler() = default;

std::vector<VkPipelineShaderStageCreateInfo> SlangShaderCompiler::compile(Device &device, const std::string &sourcePath,
	const std::vector<std::pair<std::string, VkShaderStageFlagBits>> &entryPoints)
{
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	shaderStages.reserve(entryPoints.size());

	Slang::ComPtr<slang::IBlob> diagnostics;
	Slang::ComPtr<slang::IModule> module{ m_impl->session->loadModuleFromSource("vulkr_shader", sourcePath.c_str(), nullptr, diagnostics.writeRef()) };
	if (!module)
	{
		if (diagnostics)
		{
			LOGE("Slang failed to load {}: {}", sourcePath, static_cast<const char *>(diagnostics->getBufferPointer()));
		}

		LOGEANDABORT("Failed to load the Slang module {}", sourcePath);
	}

	for (const auto &[entryPointName, stage] : entryPoints)
	{
		// Slang does not look entry points up by name on the module, so match against the defined entry points instead
		Slang::ComPtr<slang::IEntryPoint> entryPoint;
		const SlangInt32 entryPointCount = module->getDefinedEntryPointCount();
		for (SlangInt32 entryPointIndex = 0; entryPointIndex < entryPointCount; ++entryPointIndex)
		{
			Slang::ComPtr<slang::IEntryPoint> candidateEntryPoint;
			module->getDefinedEntryPoint(entryPointIndex, candidateEntryPoint.writeRef());
			if (candidateEntryPoint && entryPointName == candidateEntryPoint->getFunctionReflection()->getName())
			{
				entryPoint = candidateEntryPoint;
				break;
			}
		}

		if (!entryPoint)
		{
			LOGEANDABORT("The Slang module {} does not define the entry point {}", sourcePath, entryPointName);
		}

		// An entry point can only be compiled as part of a composite of the module and the entry point itself
		slang::IComponentType *components[]{ module, entryPoint };
		Slang::ComPtr<slang::IComponentType> program;
		if (SLANG_FAILED(m_impl->session->createCompositeComponentType(components, 2, program.writeRef(), diagnostics.writeRef())))
		{
			if (diagnostics)
			{
				LOGE("Slang failed to link {}: {}", sourcePath, static_cast<const char *>(diagnostics->getBufferPointer()));
			}

			LOGEANDABORT("Failed to build a program for the entry point {}", entryPointName);
		}

		Slang::ComPtr<ISlangBlob> spirv;
		if (SLANG_FAILED(program->getEntryPointCode(0, 0, spirv.writeRef(), diagnostics.writeRef())))
		{
			if (diagnostics)
			{
				LOGE("Slang failed to compile {}: {}", entryPointName, static_cast<const char *>(diagnostics->getBufferPointer()));
			}

			LOGEANDABORT("Failed to compile the entry point {}", entryPointName);
		}

		VkShaderModuleCreateInfo shaderModuleCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spirv->getBufferSize(),
			.pCode = static_cast<const uint32_t *>(spirv->getBufferPointer()),
		};

		VkShaderModule shaderModule{ VK_NULL_HANDLE };
		VK_CHECK(vkCreateShaderModule(device.getHandle(), &shaderModuleCreateInfo, nullptr, &shaderModule));
		m_shaderModules.emplace_back(shaderModule);

		LOGI("Compiled the Slang entry point {} to {} bytes of SPIR-V", entryPointName, spirv->getBufferSize());

		shaderStages.emplace_back(VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = stage,
			.module = shaderModule,
			.pName = "main",
		});
	}

	return shaderStages;
}

void SlangShaderCompiler::destroyShaderModules(Device &device)
{
	for (VkShaderModule shaderModule : m_shaderModules)
	{
		vkDestroyShaderModule(device.getHandle(), shaderModule, nullptr);
	}

	m_shaderModules.clear();
}

} // namespace vulkr
