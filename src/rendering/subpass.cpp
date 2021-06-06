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

#include "subpass.h"
#include "common/helpers.h"

namespace vulkr
{

Subpass::Subpass(
	std::vector<VkAttachmentReference> &inputAttachments,
	std::vector<VkAttachmentReference> &colorAttachments,
	std::vector<VkAttachmentReference> &resolveAttachments,
	std::vector<VkAttachmentReference> &depthStencilAttachments,
	std::vector<uint32_t> &preserveAttachments,
	VkPipelineBindPoint bindPoint,
	VkSubpassDescriptionFlags descriptionFlags):
	inputAttachments{ inputAttachments },
	colorAttachments{ colorAttachments },
	resolveAttachments{ resolveAttachments },
	depthStencilAttachments{ depthStencilAttachments },
	preserveAttachments{ preserveAttachments },
	bindPoint{ bindPoint },
	descriptionFlags{ descriptionFlags }
{
}

std::vector<VkAttachmentReference> &Subpass::getInputAttachments() const
{
	return inputAttachments;
}

std::vector<VkAttachmentReference> &Subpass::getColorAttachments() const
{
	return colorAttachments;
}

std::vector<VkAttachmentReference> &Subpass::getResolveAttachments() const
{
	return resolveAttachments;
}

std::vector<VkAttachmentReference> &Subpass::getDepthStencilAttachments() const
{
	return depthStencilAttachments;
}

std::vector<uint32_t> &Subpass::getPreserveAttachments() const
{
	return preserveAttachments;
}

VkPipelineBindPoint Subpass::getBindPoint() const
{
	return bindPoint;
}

VkSubpassDescriptionFlags Subpass::getDescriptionFlags() const
{
	return descriptionFlags;
}

} // namespace vulkr