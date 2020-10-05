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

#include "command_pool.h"
#include "common/logger.h"

namespace vulkr
{

CommandPool::CommandPool(Device &device, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags) : device{ device }, queueFamilyIndex{ queueFamilyIndex }, flags{ flags }
{
	VkCommandPoolCreateInfo commandPoolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };

	commandPoolInfo.queueFamilyIndex = queueFamilyIndex;
	commandPoolInfo.flags = flags;

	VK_CHECK(vkCreateCommandPool(device.getHandle(), &commandPoolInfo, nullptr, &handle));
}

CommandPool::~CommandPool()
{
	primaryCommandBuffers.clear();
	secondaryCommandBuffers.clear();

	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(device.getHandle(), handle, nullptr);
	}
}

CommandPool::CommandPool(CommandPool &&other) :
	device{ other.device },
	handle{ other.handle },
	queueFamilyIndex{ other.queueFamilyIndex },
	primaryCommandBuffers{ std::move(other.primaryCommandBuffers) },
	secondaryCommandBuffers{ std::move(other.secondaryCommandBuffers) }
{
	other.handle = VK_NULL_HANDLE;

	other.queueFamilyIndex = 0;
	other.flags = 0;
}

Device &CommandPool::getDevice()
{
	return device;
}

uint32_t CommandPool::getQueueFamilyIndex() const
{
	return queueFamilyIndex;
}

VkCommandPool CommandPool::getHandle() const
{
	return handle;
}

CommandBuffer &CommandPool::requestCommandBuffer(RenderPass &renderPass, Framebuffer &framebuffer, VkCommandBufferLevel level)
{
	if (level == VK_COMMAND_BUFFER_LEVEL_PRIMARY)
	{
		if (activePrimaryCommandBufferCount < primaryCommandBuffers.size())
		{
			return *(primaryCommandBuffers.at(activePrimaryCommandBufferCount++));
		}

		primaryCommandBuffers.emplace_back(std::make_unique<CommandBuffer>(*this, level, renderPass, framebuffer));
		activePrimaryCommandBufferCount++;

		return *(primaryCommandBuffers.back());
	}
	else if (level == VK_COMMAND_BUFFER_LEVEL_SECONDARY)
	{
		if (activeSecondaryCommandBufferCount < secondaryCommandBuffers.size())
		{
			return *(secondaryCommandBuffers.at(activeSecondaryCommandBufferCount++));
		}

		secondaryCommandBuffers.emplace_back(std::make_unique<CommandBuffer>(*this, level, renderPass, framebuffer));
		activeSecondaryCommandBufferCount++;

		return *(secondaryCommandBuffers.back());
	}

	LOGEANDABORT("Unknown command buffer level type requesteda");
}

} // namespace vulkr