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
class Subpass;

/* Description of render pass attachments */
struct Attachment
{
	VkFormat format{ VK_FORMAT_UNDEFINED };
	VkSampleCountFlagBits samples{ VK_SAMPLE_COUNT_1_BIT };
	VkAttachmentLoadOp loadOp{ VK_ATTACHMENT_LOAD_OP_LOAD };
	VkAttachmentStoreOp storeOp{ VK_ATTACHMENT_STORE_OP_STORE };
	VkAttachmentLoadOp stencilLoadOp{ VK_ATTACHMENT_LOAD_OP_LOAD };
	VkAttachmentStoreOp stencilStoreOp{ VK_ATTACHMENT_STORE_OP_STORE };
	VkImageLayout initialLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
	VkImageLayout finalLayout{ VK_IMAGE_LAYOUT_UNDEFINED };

	Attachment() = default;
};

class RenderPass
{
public:
	// TODO: introduce a subpass dependency data structure and set it up here
	RenderPass(Device &device, const std::vector<Attachment> &attachments, const std::vector<Subpass> &subpasses, const std::vector<VkSubpassDependency> subpassDependencies);
	~RenderPass();

	RenderPass(const RenderPass &) = delete;
	RenderPass(RenderPass &&) = delete;
	RenderPass &operator=(const RenderPass &) = delete;
	RenderPass &operator=(RenderPass &&) = delete;

	VkRenderPass getHandle() const;

	const std::vector<Attachment> &getAttachments() const;
	const std::vector<Subpass> &getSubpasses() const;

private:
	VkRenderPass handle{ VK_NULL_HANDLE };
	Device &device;
	const std::vector<Attachment> &attachments;
	const std::vector<Subpass> &subpasses;
};

} // namespace vulkr