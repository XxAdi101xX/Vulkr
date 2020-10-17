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
class CommandBuffer;
class RenderPass;
class Framebuffer;

class CommandPool
{
public:
	CommandPool(Device &device, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags);
	~CommandPool();

	CommandPool(CommandPool&& other);

	/* Disable unnecessary operators to prevent error prone usages */
	CommandPool(const CommandPool &) = delete;
	CommandPool& operator=(const CommandPool &) = delete;
	CommandPool& operator=(CommandPool &&) = delete;

	Device &getDevice();
	VkCommandPool getHandle() const;

	uint32_t getQueueFamilyIndex() const;

	CommandBuffer &requestCommandBuffer(RenderPass &renderPass, Framebuffer &framebuffer, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	// TODO: look into vkresetcommandpool and vkreset command buffers

private:
	Device& device;

	VkCommandPool handle{ VK_NULL_HANDLE };

	uint32_t queueFamilyIndex{ 0 };

	VkCommandPoolCreateFlags flags{ 0 };

	std::vector<std::unique_ptr<CommandBuffer>> primaryCommandBuffers;
	int32_t activePrimaryCommandBufferCount{ 0 };

	std::vector<std::unique_ptr<CommandBuffer>> secondaryCommandBuffers;
	int32_t activeSecondaryCommandBufferCount{ 0 };
};

} // namespace vulkr