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

#include "command_buffer.h"
#include "command_pool.h"
#include "device.h"
#include "physical_device.h"
#include "render_pass.h"
#include "framebuffer.h"

#include "common/helpers.h"

namespace vulkr
{

CommandBuffer::CommandBuffer(CommandPool &commandPool, VkCommandBufferLevel level) :
	commandPool{ commandPool },
	level{ level },
	maxPushConstantsSize{ commandPool.getDevice().getPhysicalDevice().getProperties().limits.maxPushConstantsSize }
{
	VkCommandBufferAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	allocateInfo.commandPool = commandPool.getHandle();
	allocateInfo.commandBufferCount = 1;
	allocateInfo.level = level;

	VK_CHECK(vkAllocateCommandBuffers(commandPool.getDevice().getHandle(), &allocateInfo, &handle));
}

CommandBuffer::~CommandBuffer()
{
	if (handle != VK_NULL_HANDLE)
	{
		vkFreeCommandBuffers(getDevice().getHandle(), commandPool.getHandle(), 1, &handle);
	}
}

Device &CommandBuffer::getDevice()
{
	return commandPool.getDevice();
}

const VkCommandBufferLevel CommandBuffer::getLevel() const
{
	return level;
}

const VkCommandBuffer &CommandBuffer::getHandle() const
{
	return handle;
}

bool CommandBuffer::isRecording() const
{
	return state == State::Recording;
}

void CommandBuffer::begin(VkCommandBufferUsageFlags flags, CommandBuffer *primaryCommandBuffer)
{
	if (isRecording())
	{
		LOGEANDABORT("Begin was called on a command buffer in the recording state");
	}

	state = State::Recording;

	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = flags;

	VkCommandBufferInheritanceInfo inheritance = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
	if (level == VK_COMMAND_BUFFER_LEVEL_SECONDARY)
	{
		if (primaryCommandBuffer == nullptr)
		{
			LOGEANDABORT("primary command buffer not provided when starting secondary command buffer");
		}

		if (primaryCommandBuffer->getLevel() == VK_COMMAND_BUFFER_LEVEL_SECONDARY)
		{
			LOGEANDABORT("Primary command buffer expected but received a secondary command buffer");
		}

		LOGEANDABORT("Secondary command buffers not handled yet. Need to implement!!"); /* TODO: Remove this eventually */

		//auto render_pass_binding = primaryCommandBuffer->get_current_render_pass();
		//current_render_pass.render_pass = render_pass_binding.render_pass;
		//current_render_pass.framebuffer = render_pass_binding.framebuffer;

		//inheritance.renderPass = current_render_pass.render_pass->get_handle();
		//inheritance.framebuffer = current_render_pass.framebuffer->get_handle();
		//inheritance.subpass = primaryCommandBuffer->get_current_subpass_index();

		//beginInfo.pInheritanceInfo = &inheritance;
	}

	VK_CHECK(vkBeginCommandBuffer(handle, &beginInfo));
}

void CommandBuffer::end()
{

	if (!isRecording())
	{
		LOGEANDABORT("Attempting to end a command buffer that's not recording");
	}

	VK_CHECK(vkEndCommandBuffer(handle));

	state = State::Executable;
}

void CommandBuffer::beginRenderPass(RenderPass &renderPass, Framebuffer &framebuffer, const VkExtent2D extent, const std::vector<VkClearValue> &clearValues, VkSubpassContents subpassContents)
{
	VkRenderPassBeginInfo renderPassBeginInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
	renderPassBeginInfo.renderPass = renderPass.getHandle();
	renderPassBeginInfo.framebuffer = framebuffer.getHandle();
	renderPassBeginInfo.renderArea.offset = { 0, 0 };
	renderPassBeginInfo.renderArea.extent = extent;
	renderPassBeginInfo.clearValueCount = to_u32(clearValues.size());
	renderPassBeginInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(handle, &renderPassBeginInfo, subpassContents);
}

void CommandBuffer::endRenderPass()
{
	vkCmdEndRenderPass(handle);
}

void CommandBuffer::reset()
{
	VK_CHECK(vkResetCommandBuffer(handle, 0));

	state = State::Initial;
}

} // namespace vulkr