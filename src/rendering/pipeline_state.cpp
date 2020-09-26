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

namespace vulkr
{

const PipelineLayout *PipelineState::getPipelineLayout() const
{
	return pipelineLayout;
}

const RenderPass *PipelineState::getRenderPass() const
{
	return renderPass;
}

const VertexInputState &PipelineState::getVertexInputState() const
{
	return vertexInputState;
}

const InputAssemblyState &PipelineState::getInputAssemblyState() const
{
	return inputAssemblyState;
}

const RasterizationState &PipelineState::getRasterizationState() const
{
	return rasterizationState;
}

const ViewportState &PipelineState::getViewportState() const
{
	return viewportState;
}

const MultisampleState &PipelineState::getMultisampleState() const
{
	return multisampleState;
}

const DepthStencilState &PipelineState::getDepthStencilState() const
{
	return depthStencilState;
}

const ColorBlendState &PipelineState::getColorBlendState() const
{
	return colorBlendState;
}

const std::array<VkDynamicState, 9>  &PipelineState::getDyanmicStates() const
{
	return dynamicStates;
}

uint32_t PipelineState::getSubpassIndex() const
{
	return subpassIndex;
}

} // namespace vulkr