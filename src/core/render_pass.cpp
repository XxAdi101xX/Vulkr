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

#include "render_pass.h"
#include "device.h"

#include "rendering/subpass.h"

#include "common/logger.h"
#include "common/helpers.h"

namespace vulkr
{

RenderPass::RenderPass(Device &device, const std::vector<Attachment> &attachments, const std::vector<Subpass> &subpasses, const std::vector<VkSubpassDependency2> subpassDependencies) :
	device{ device },
	attachments{ attachments },
	subpasses{ subpasses }
{
	if (subpasses.empty())
	{
		LOGEANDABORT("There are no associated subpasses for a render pass");
	}

	std::vector<VkAttachmentDescription2> attachmentDescriptions;
	attachmentDescriptions.reserve(attachments.size());

	for (const auto &attachement : attachments)
	{
		VkAttachmentDescription2 attachmentDescription{ VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2 };
		attachmentDescription.pNext = nullptr;
		attachmentDescription.flags = 0u;
		attachmentDescription.format = attachement.format;
		attachmentDescription.samples = attachement.samples;
		attachmentDescription.loadOp = attachement.loadOp;
		attachmentDescription.storeOp = attachement.storeOp;
		attachmentDescription.stencilLoadOp = attachement.stencilLoadOp;
		attachmentDescription.stencilStoreOp = attachement.stencilStoreOp;
		attachmentDescription.initialLayout = attachement.initialLayout;
		attachmentDescription.finalLayout = attachement.finalLayout;

		attachmentDescriptions.push_back(std::move(attachmentDescription));
	}

	std::vector<VkSubpassDescription2> subpassDescriptions;
	subpassDescriptions.reserve(subpasses.size());

	for (const auto &subpass : subpasses)
	{
		VkSubpassDescription2 subpassDescription{ VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2 };
		subpassDescription.pNext = nullptr;
		subpassDescription.flags = subpass.getDescriptionFlags();
		subpassDescription.pipelineBindPoint = subpass.getBindPoint();
		subpassDescription.inputAttachmentCount = to_u32(subpass.getInputAttachments().size());
		subpassDescription.pInputAttachments = subpass.getInputAttachments().empty() ? nullptr : subpass.getInputAttachments().data();
		subpassDescription.colorAttachmentCount = to_u32(subpass.getColorAttachments().size());
		subpassDescription.pColorAttachments = subpass.getColorAttachments().empty() ? nullptr : subpass.getColorAttachments().data();
		subpassDescription.pResolveAttachments = subpass.getResolveAttachments().empty() ? nullptr : subpass.getResolveAttachments().data();
		subpassDescription.pDepthStencilAttachment = subpass.getDepthStencilAttachments().empty() ? nullptr : subpass.getDepthStencilAttachments().data();
		subpassDescription.preserveAttachmentCount = to_u32(subpass.getPreserveAttachments().size());
		subpassDescription.pPreserveAttachments = subpass.getPreserveAttachments().empty() ? nullptr : subpass.getPreserveAttachments().data();

		subpassDescriptions.push_back(std::move(subpassDescription));
	}

	if (subpasses.size() > 1)
	{
		LOGEANDABORT("TODO: check if we need to setup subpass dependencies between more than one subpass");
	}
	// TODO: check if we need to set subpass dependencies between subpasses when we have more subpasses
	//std::vector<VkSubpassDependency2> dependencies;
	//dependencies.reserve(subpasses.size() - 1);

	//for (uint32_t i = 0; i < dependencies.size(); ++i)
	//{
	//	// Transition input attachments from color attachment to shader read
	//	dependencies[i].srcSubpass = i;
	//	dependencies[i].dstSubpass = i + 1;
	//	dependencies[i].srcStageMask = VK_PIPELINE_2_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	//	dependencies[i].dstStageMask = VK_PIPELINE_2_STAGE_FRAGMENT_SHADER_BIT;
	//	dependencies[i].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
	//	dependencies[i].dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
	//	dependencies[i].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	//}

	// Create render pass
	VkRenderPassCreateInfo2 renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2 };
	renderPassInfo.pNext = nullptr;
	renderPassInfo.flags = 0u;
	renderPassInfo.attachmentCount = to_u32(attachmentDescriptions.size());
	renderPassInfo.pAttachments = attachmentDescriptions.data();
	renderPassInfo.subpassCount = to_u32(subpasses.size());
	renderPassInfo.pSubpasses = subpassDescriptions.data();
	renderPassInfo.dependencyCount = to_u32(subpassDependencies.size());
	renderPassInfo.pDependencies = subpassDependencies.data();

	VK_CHECK(vkCreateRenderPass2(device.getHandle(), &renderPassInfo, nullptr, &handle));
}

RenderPass::~RenderPass()
{
	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyRenderPass(device.getHandle(), handle, nullptr);
	}
}

VkRenderPass RenderPass::getHandle() const
{
	return handle;
}

const std::vector<Attachment> &RenderPass::getAttachments() const
{
	return attachments;
}

const std::vector<Subpass> &RenderPass::getSubpasses() const
{
	return subpasses;
}

} // namespace vulkr