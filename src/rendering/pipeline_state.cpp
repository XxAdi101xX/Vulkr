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

#include "pipeline_state.h"
#include "core/pipeline_layout.h"

namespace vulkr
{

/* Parent class methods */
PipelineState::PipelineState(std::unique_ptr<PipelineLayout> &&pipelineLayout) : pipelineLayout{ std::move(pipelineLayout) } {}

PipelineState::~PipelineState()
{
	pipelineLayout.reset();
}

const PipelineLayout &PipelineState::getPipelineLayout() const
{
	return *pipelineLayout;
}

/* GraphicsPipelineState methods */
GraphicsPipelineState::GraphicsPipelineState(
	std::unique_ptr<PipelineLayout> &&pipelineLayout,
	RenderPass &renderPass,
	VertexInputState vertexInputState,
	InputAssemblyState inputAssemblyState,
	ViewportState viewportState,
	RasterizationState rasterizationState,
	MultisampleState multisampleState,
	DepthStencilState depthStencilState,
	ColorBlendState colorBlendState,
	std::vector<VkDynamicState> dynamicStates
) :
	PipelineState{ std::move(pipelineLayout) },
	renderPass{ renderPass },
	vertexInputState{ vertexInputState },
	inputAssemblyState{ inputAssemblyState },
	viewportState{ viewportState },
	rasterizationState{ rasterizationState },
	multisampleState{ multisampleState },
	depthStencilState{ depthStencilState },
	colorBlendState{ colorBlendState },
	dynamicStates{ dynamicStates }
{
}

const RenderPass &GraphicsPipelineState::getRenderPass() const
{
	return renderPass;
}

const VertexInputState &GraphicsPipelineState::getVertexInputState() const
{
	return vertexInputState;
}

const InputAssemblyState &GraphicsPipelineState::getInputAssemblyState() const
{
	return inputAssemblyState;
}

const RasterizationState &GraphicsPipelineState::getRasterizationState() const
{
	return rasterizationState;
}

const ViewportState &GraphicsPipelineState::getViewportState() const
{
	return viewportState;
}

const MultisampleState &GraphicsPipelineState::getMultisampleState() const
{
	return multisampleState;
}

const DepthStencilState &GraphicsPipelineState::getDepthStencilState() const
{
	return depthStencilState;
}

const ColorBlendState &GraphicsPipelineState::getColorBlendState() const
{
	return colorBlendState;
}

const std::vector<VkDynamicState> &GraphicsPipelineState::getDyanmicStates() const
{
	return dynamicStates;
}

uint32_t GraphicsPipelineState::getSubpassIndex() const
{
	return subpassIndex;
}

/* ComputePipelineState methods */
ComputePipelineState::ComputePipelineState(std::unique_ptr<PipelineLayout> &&pipelineLayout) : PipelineState{ std::move(pipelineLayout) } {}

/* RayTracingPipelineState methods */
RayTracingPipelineState::RayTracingPipelineState(
	std::unique_ptr<PipelineLayout> &&pipelineLayout,
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> rayTracingShaderGroups
) :
	PipelineState{ std::move(pipelineLayout) },
	rayTracingShaderGroups{ rayTracingShaderGroups }
{
}

const std::vector<VkRayTracingShaderGroupCreateInfoKHR> &RayTracingPipelineState::getRayTracingShaderGroups() const
{
	return rayTracingShaderGroups;
}

} // namespace vulkr