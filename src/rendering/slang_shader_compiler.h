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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/vulkan_common.h"

namespace vulkr
{

class Device;

/* Compiles Slang shader modules to SPIR-V at runtime and wraps the requested entry points in VkShaderModules.
   The compiler owns the Slang sessions and the shader modules it creates; entry points are compiled individually
   because Slang emits one SPIR-V module per entry point. */
class SlangShaderCompiler
{
public:
	SlangShaderCompiler();
	~SlangShaderCompiler();

	SlangShaderCompiler(const SlangShaderCompiler &) = delete;
	SlangShaderCompiler(SlangShaderCompiler &&) = delete;
	SlangShaderCompiler &operator=(const SlangShaderCompiler &) = delete;
	SlangShaderCompiler &operator=(SlangShaderCompiler &&) = delete;

	/* Compiles the requested entry points of the module at sourcePath and returns one shader stage per entry point.
	   Each pair holds the Slang entry point name and the Vulkan stage that entry point belongs to */
	std::vector<VkPipelineShaderStageCreateInfo> compile(Device &device, const std::string &sourcePath,
		const std::vector<std::pair<std::string, VkShaderStageFlagBits>> &entryPoints);

	/* Releases every shader module created by compile() */
	void destroyShaderModules(Device &device);
private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;

	std::vector<VkShaderModule> m_shaderModules;
};

} // namespace vulkr
