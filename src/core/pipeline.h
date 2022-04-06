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

#pragma once

#include "common/vulkan_common.h"

namespace vulkr
{

class Device;
class GraphicsPipelineState;
class ComputePipelineState;
class RayTracingPipelineState;

class Pipeline
{
public:
	virtual ~Pipeline() = 0;
	Pipeline(Pipeline &&other);

	Pipeline(const Pipeline &) = delete;
	Pipeline &operator=(const Pipeline &) = delete;
	Pipeline &operator=(Pipeline &&) = delete;

	VkPipeline getHandle() const;
	VkPipelineBindPoint getBindPoint() const;

protected:
	Pipeline(Device &device, VkPipelineBindPoint bindPoint);
	VkPipeline handle = VK_NULL_HANDLE;
	Device &device;
	VkPipelineBindPoint bindPoint;
};

class GraphicsPipeline final : public Pipeline
{
public:
	GraphicsPipeline(Device &device, GraphicsPipelineState &pipelineState, VkPipelineCache pipelineCache); // TODO: incorportate pipeline cache
	~GraphicsPipeline() = default;
	GraphicsPipeline(GraphicsPipeline &&) = default;
private:
	GraphicsPipelineState &pipelineState;
};

class ComputePipeline final : public Pipeline
{
public:
	ComputePipeline(Device &device, ComputePipelineState &pipelineState, VkPipelineCache pipelineCache); // TODO: incorportate pipeline cache
	~ComputePipeline() = default;
	ComputePipeline(ComputePipeline &&) = default;
private:
	ComputePipelineState &pipelineState;
};

class RayTracingPipeline final : public Pipeline
{
public:
	RayTracingPipeline(Device &device, RayTracingPipelineState &pipelineState, VkPipelineCache pipelineCache); // TODO: incorportate pipeline cache
	~RayTracingPipeline() = default;
	RayTracingPipeline(RayTracingPipeline &&) = default;
private:
	RayTracingPipelineState &pipelineState;
};

} // namespace vulkr