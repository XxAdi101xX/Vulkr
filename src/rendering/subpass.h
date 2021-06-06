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

class Subpass
{
public:
	Subpass(
		std::vector<VkAttachmentReference> &inputAttachments,
		std::vector<VkAttachmentReference> &colorAttachments,
		std::vector<VkAttachmentReference> &resolveAttachments,
		std::vector<VkAttachmentReference> &depthStencilAttachments,
		std::vector<uint32_t> &preserveAttachments,
		VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		VkSubpassDescriptionFlags descriptionFlags = 0u
	);
	~Subpass() = default;
	Subpass(Subpass &&) = default;

	/* Disable unnecessary operators to prevent error prone usages */
	Subpass(const Subpass &) = delete;
	Subpass &operator=(const Subpass &) = delete;
	Subpass &operator=(Subpass &&) = delete;

	std::vector<VkAttachmentReference> &getInputAttachments() const;
	std::vector<VkAttachmentReference> &getColorAttachments() const;
	std::vector<VkAttachmentReference> &getResolveAttachments() const;
	std::vector<VkAttachmentReference> &getDepthStencilAttachments() const;
	std::vector<uint32_t> &getPreserveAttachments() const;
	VkPipelineBindPoint getBindPoint() const;
	VkSubpassDescriptionFlags getDescriptionFlags() const;
private:
	std::vector<VkAttachmentReference> &inputAttachments;
	std::vector<VkAttachmentReference> &colorAttachments;
	std::vector<VkAttachmentReference> &resolveAttachments;
	std::vector<VkAttachmentReference> &depthStencilAttachments;
	std::vector<uint32_t> &preserveAttachments;
	VkPipelineBindPoint bindPoint;
	VkSubpassDescriptionFlags descriptionFlags;
};

} // namespace vulkr