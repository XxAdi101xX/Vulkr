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

#include "framebuffer.h"
#include "device.h"
#include "swapchain.h"
#include "render_pass.h"

#include "common/helpers.h"

namespace vulkr
{

Framebuffer::Framebuffer(Device &device, const Swapchain &swapchain, const RenderPass &renderPass, std::vector<VkImageView> attachments) :
	device{ device }
{
	VkFramebufferCreateInfo framebufferInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };

	framebufferInfo.renderPass = renderPass.getHandle();
	framebufferInfo.attachmentCount = to_u32(attachments.size());
	framebufferInfo.pAttachments = attachments.data(); // TODO: should we make attachments a reference to vector
	framebufferInfo.width = swapchain.getProperties().imageExtent.width;
	framebufferInfo.height = swapchain.getProperties().imageExtent.height;
	framebufferInfo.layers = 1;

	VK_CHECK(vkCreateFramebuffer(device.getHandle(), &framebufferInfo, nullptr, &handle));
}

Framebuffer::~Framebuffer()
{
	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyFramebuffer(device.getHandle(), handle, nullptr);
	}
}

VkFramebuffer Framebuffer::getHandle() const
{
	return handle;
}

} // namespace vulkr