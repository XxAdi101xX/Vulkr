/* Copyright (c) 2020 Adithya Venkatarao
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

#include "pipeline.h"
#include "device.h"
#include "render_pass.h"
#include "pipeline_layout.h"
#include "physical_device.h"

#include "rendering/pipeline_state.h"
#include "rendering/shader_module.h"

#include "common/helpers.h"

namespace vulkr
{

// Pipeline
Pipeline::Pipeline(Device &device, VkPipelineBindPoint bindPoint) : device{ device }, bindPoint{ bindPoint } {}

Pipeline::~Pipeline()
{
	vkDestroyPipeline(device.getHandle(), handle, nullptr);
}

Pipeline::Pipeline(Pipeline &&other) :
	device{ other.device },
	handle{ other.handle },
	bindPoint{ other.bindPoint }
{
	other.handle = VK_NULL_HANDLE;
}

VkPipeline Pipeline::getHandle() const
{
	return handle;
}

VkPipelineBindPoint Pipeline::getBindPoint() const
{
	return bindPoint;
}

// GraphicsPipeline
GraphicsPipeline::GraphicsPipeline(Device &device, GraphicsPipelineState &pipelineState, VkPipelineCache pipelineCache) : Pipeline{ device, VK_PIPELINE_BIND_POINT_GRAPHICS }, pipelineState{ pipelineState }
{
	std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;

	for (const ShaderModule &shaderModule : pipelineState.getPipelineLayout().getShaderModules())
	{
		VkPipelineShaderStageCreateInfo shaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		shaderStageCreateInfo.stage = shaderModule.getStage();

		// Since we only need the shadermodule during the creation of the pipeline, we don't keep the information in the shader module class and delete at the end
		VkShaderModuleCreateInfo shaderModuleCreateInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		shaderModuleCreateInfo.codeSize = shaderModule.getShaderSource().getData().size();
		shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(shaderModule.getShaderSource().getData().data());
		VK_CHECK(vkCreateShaderModule(device.getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));

		shaderStageCreateInfo.pName = shaderModule.getEntryPoint().c_str();
		shaderStageCreateInfo.pSpecializationInfo = shaderModule.getSpecializationInfo().mapEntryCount != 0u ? &shaderModule.getSpecializationInfo() : nullptr;

		shaderStageCreateInfos.push_back(shaderStageCreateInfo);
	}

	VkPipelineVertexInputStateCreateInfo vertexInputState{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	vertexInputState.vertexAttributeDescriptionCount = to_u32(pipelineState.getVertexInputState().attributeDescriptions.size());
	vertexInputState.pVertexAttributeDescriptions = pipelineState.getVertexInputState().attributeDescriptions.empty() ? nullptr : pipelineState.getVertexInputState().attributeDescriptions.data();
	vertexInputState.vertexBindingDescriptionCount = to_u32(pipelineState.getVertexInputState().bindingDescriptions.size());
	vertexInputState.pVertexBindingDescriptions = pipelineState.getVertexInputState().bindingDescriptions.empty() ? nullptr : pipelineState.getVertexInputState().bindingDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
	inputAssemblyState.topology = pipelineState.getInputAssemblyState().topology;
	inputAssemblyState.primitiveRestartEnable = pipelineState.getInputAssemblyState().primitiveRestartEnable;

	VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
	viewportState.viewportCount = to_u32(pipelineState.getViewportState().viewports.size());
	viewportState.pViewports = pipelineState.getViewportState().viewports.empty() ? nullptr : pipelineState.getViewportState().viewports.data();
	viewportState.scissorCount = to_u32(pipelineState.getViewportState().scissors.size());
	viewportState.pScissors = pipelineState.getViewportState().scissors.empty() ? nullptr : pipelineState.getViewportState().scissors.data();

	VkPipelineRasterizationStateCreateInfo rasterizationState{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
	rasterizationState.depthClampEnable = pipelineState.getRasterizationState().depthClampEnable;
	rasterizationState.rasterizerDiscardEnable = pipelineState.getRasterizationState().rasterizerDiscardEnable;
	rasterizationState.polygonMode = pipelineState.getRasterizationState().polygonMode;
	rasterizationState.cullMode = pipelineState.getRasterizationState().cullMode;
	rasterizationState.frontFace = pipelineState.getRasterizationState().frontFace;
	rasterizationState.depthBiasEnable = pipelineState.getRasterizationState().depthBiasEnable;
	rasterizationState.depthBiasClamp = pipelineState.getRasterizationState().depthBiasClamp;
	rasterizationState.depthBiasSlopeFactor = pipelineState.getRasterizationState().depthBiasSlopeFactor;
	rasterizationState.lineWidth = pipelineState.getRasterizationState().lineWidth;

	VkPipelineMultisampleStateCreateInfo multisampleState{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
	multisampleState.sampleShadingEnable = pipelineState.getMultisampleState().sampleShadingEnable;
	multisampleState.rasterizationSamples = pipelineState.getMultisampleState().rasterizationSamples;
	multisampleState.minSampleShading = pipelineState.getMultisampleState().min_sample_shading;
	multisampleState.pSampleMask = pipelineState.getMultisampleState().sampleMask;
	multisampleState.alphaToCoverageEnable = pipelineState.getMultisampleState().alphaToCoverageEnable;
	multisampleState.alphaToOneEnable = pipelineState.getMultisampleState().alphaToOneEnable;

	VkPipelineDepthStencilStateCreateInfo depthStencilState{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	depthStencilState.depthTestEnable = pipelineState.getDepthStencilState().depthTestEnable;
	depthStencilState.depthWriteEnable = pipelineState.getDepthStencilState().depthWriteEnable;
	depthStencilState.depthCompareOp = pipelineState.getDepthStencilState().depthCompareOp;
	depthStencilState.depthBoundsTestEnable = pipelineState.getDepthStencilState().depthBoundsTestEnable;
	depthStencilState.stencilTestEnable = pipelineState.getDepthStencilState().stencilTestEnable;
	depthStencilState.front.failOp = pipelineState.getDepthStencilState().front.failOp;
	depthStencilState.front.passOp = pipelineState.getDepthStencilState().front.passOp;
	depthStencilState.front.depthFailOp = pipelineState.getDepthStencilState().front.depthFailOp;
	depthStencilState.front.compareOp = pipelineState.getDepthStencilState().front.compareOp;
	depthStencilState.front.compareMask = 0u;
	depthStencilState.front.writeMask = 0u;
	depthStencilState.front.reference = 0u;
	depthStencilState.back.failOp = pipelineState.getDepthStencilState().back.failOp;
	depthStencilState.back.passOp = pipelineState.getDepthStencilState().back.passOp;
	depthStencilState.back.depthFailOp = pipelineState.getDepthStencilState().back.depthFailOp;
	depthStencilState.back.compareOp = pipelineState.getDepthStencilState().back.compareOp;
	depthStencilState.back.compareMask = 0u;
	depthStencilState.back.writeMask = 0u;
	depthStencilState.back.reference = 0u;
	depthStencilState.minDepthBounds = pipelineState.getDepthStencilState().minDepthBounds;
	depthStencilState.maxDepthBounds = pipelineState.getDepthStencilState().maxDepthBounds;

	VkPipelineColorBlendStateCreateInfo colorBlendState{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
	colorBlendState.logicOpEnable = pipelineState.getColorBlendState().logicOpEnable;
	colorBlendState.logicOp = pipelineState.getColorBlendState().logicOp;
	colorBlendState.attachmentCount = to_u32(pipelineState.getColorBlendState().attachments.size());
	// This reinterpret_cast works because the structs are layed out the same way
	colorBlendState.pAttachments = pipelineState.getColorBlendState().attachments.empty() ? nullptr : reinterpret_cast<const VkPipelineColorBlendAttachmentState *>(pipelineState.getColorBlendState().attachments.data());
	colorBlendState.blendConstants[0] = pipelineState.getColorBlendState().blendConstants[0];
	colorBlendState.blendConstants[1] = pipelineState.getColorBlendState().blendConstants[1];
	colorBlendState.blendConstants[2] = pipelineState.getColorBlendState().blendConstants[2];
	colorBlendState.blendConstants[3] = pipelineState.getColorBlendState().blendConstants[3];

	VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
	dynamicState.pDynamicStates = pipelineState.getDyanmicStates().empty() ? nullptr : pipelineState.getDyanmicStates().data();
	dynamicState.dynamicStateCount = to_u32(pipelineState.getDyanmicStates().size());

	VkGraphicsPipelineCreateInfo graphicsPipeline{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
	graphicsPipeline.stageCount = to_u32(shaderStageCreateInfos.size());
	graphicsPipeline.pStages = shaderStageCreateInfos.empty() ? nullptr : shaderStageCreateInfos.data();
	graphicsPipeline.pVertexInputState = &vertexInputState;
	graphicsPipeline.pInputAssemblyState = &inputAssemblyState;
	graphicsPipeline.pViewportState = &viewportState;
	graphicsPipeline.pRasterizationState = &rasterizationState;
	graphicsPipeline.pMultisampleState = &multisampleState;
	graphicsPipeline.pDepthStencilState = &depthStencilState;
	graphicsPipeline.pColorBlendState = &colorBlendState;
	graphicsPipeline.pDynamicState = &dynamicState;

	graphicsPipeline.layout = pipelineState.getPipelineLayout().getHandle();
	graphicsPipeline.renderPass = pipelineState.getRenderPass().getHandle();
	graphicsPipeline.subpass = pipelineState.getSubpassIndex();

	// Used when you want to create a pipeline from an existing pipeline, must have the VK_PIPELINE_CREATE_DERIVATIVE_BIT enabled in the flags field of vkCreateGraphicsPipelines 
	graphicsPipeline.basePipelineHandle = VK_NULL_HANDLE; // Optional
	graphicsPipeline.basePipelineIndex = -1; // Optional

	VK_CHECK(vkCreateGraphicsPipelines(device.getHandle(), pipelineCache, 1, &graphicsPipeline, nullptr, &handle));

	for (auto shaderStageCreateInfo : shaderStageCreateInfos)
	{
		vkDestroyShaderModule(device.getHandle(), shaderStageCreateInfo.module, nullptr);
	}
}

// Compute Pipeline
ComputePipeline::ComputePipeline(Device &device, ComputePipelineState &pipelineState, VkPipelineCache pipelineCache) : Pipeline{ device, VK_PIPELINE_BIND_POINT_COMPUTE }, pipelineState{ pipelineState }
{
	if (pipelineState.getPipelineLayout().getShaderModules().size() != 1)
	{
		LOGEANDABORT("Only one shader module is expected per compute pipeline");
	}

	VkComputePipelineCreateInfo computePipelineCreateInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	computePipelineCreateInfo.layout = pipelineState.getPipelineLayout().getHandle();

	const ShaderModule &shaderModule = pipelineState.getPipelineLayout().getShaderModules()[0];
	VkPipelineShaderStageCreateInfo shaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	shaderStageCreateInfo.stage = shaderModule.getStage();

	VkShaderModuleCreateInfo shaderModuleCreateInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	shaderModuleCreateInfo.codeSize = shaderModule.getShaderSource().getData().size();
	shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(shaderModule.getShaderSource().getData().data());
	VK_CHECK(vkCreateShaderModule(device.getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));

	shaderStageCreateInfo.pName = shaderModule.getEntryPoint().c_str();
	shaderStageCreateInfo.pSpecializationInfo = shaderModule.getSpecializationInfo().mapEntryCount != 0u ? &shaderModule.getSpecializationInfo() : nullptr;
	computePipelineCreateInfo.stage = shaderStageCreateInfo;

	vkCreateComputePipelines(device.getHandle(), pipelineCache, 1, &computePipelineCreateInfo, nullptr, &handle);

	vkDestroyShaderModule(device.getHandle(), shaderStageCreateInfo.module, nullptr);
}

// RayTracingPipeline
RayTracingPipeline::RayTracingPipeline(Device &device, RayTracingPipelineState &pipelineState, VkPipelineCache pipelineCache) : Pipeline{ device, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR }, pipelineState{ pipelineState }
{
	// Spec only guarantees 1 level of ray recursion and we require atleast two to support shadow rays
	if (device.getPhysicalDevice().getRayTracingPipelineProperties().maxRayRecursionDepth <= 1)
	{
		LOGEANDABORT("Device fails to support ray recursion as maxRayRecursionDepth <= 1");
	}

	std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;
	shaderStageCreateInfos.reserve(pipelineState.getPipelineLayout().getShaderModules().size());

	for (const ShaderModule &shaderModule : pipelineState.getPipelineLayout().getShaderModules())
	{
		VkPipelineShaderStageCreateInfo shaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		shaderStageCreateInfo.stage = shaderModule.getStage();

		VkShaderModuleCreateInfo shaderModuleCreateInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		shaderModuleCreateInfo.codeSize = shaderModule.getShaderSource().getData().size();
		shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(shaderModule.getShaderSource().getData().data());
		VK_CHECK(vkCreateShaderModule(device.getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));

		shaderStageCreateInfo.pName = shaderModule.getEntryPoint().c_str();
		shaderStageCreateInfo.pSpecializationInfo = &shaderModule.getSpecializationInfo();
		shaderStageCreateInfos.push_back(shaderStageCreateInfo);
	}

	VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rayPipelineInfo.stageCount = to_u32(shaderStageCreateInfos.size());
	rayPipelineInfo.pStages = shaderStageCreateInfos.data();
	rayPipelineInfo.groupCount = to_u32(pipelineState.getRayTracingShaderGroups().size());
	rayPipelineInfo.pGroups = pipelineState.getRayTracingShaderGroups().data();
	rayPipelineInfo.maxPipelineRayRecursionDepth = 2;
	rayPipelineInfo.layout = pipelineState.getPipelineLayout().getHandle();

	vkCreateRayTracingPipelinesKHR(device.getHandle(), {}, {}, 1, &rayPipelineInfo, nullptr, &handle);

	for (auto shaderStageCreateInfo : shaderStageCreateInfos)
	{
		vkDestroyShaderModule(device.getHandle(), shaderStageCreateInfo.module, nullptr);
	}
}

} // namespace vulkr