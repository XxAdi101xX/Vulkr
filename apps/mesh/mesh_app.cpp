/* Copyright (c) 2026 Adithya Venkatarao
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

#include "mesh_app.h"

#include <limits>

#include "common/logger.h"
#include "common/helpers.h"
#include "platform/platform.h"

#include "core/command_buffer.h"
#include "core/command_pool.h"
#include "core/device.h"
#include "core/fence_pool.h"
#include "core/image.h"
#include "core/instance.h"
#include "core/queue.h"
#include "core/semaphore_pool.h"
#include "core/swapchain.h"

namespace vulkr
{

MeshApp::MeshApp(Platform &platform, std::string name) :
	Application(platform, std::move(name))
{
}

MeshApp::~MeshApp()
{
	cleanup();

	m_device.reset();

	if (m_surface != VK_NULL_HANDLE)
	{
		vkDestroySurfaceKHR(m_instance->getHandle(), m_surface, nullptr);
		m_surface = VK_NULL_HANDLE;
	}

	m_instance.reset();
}

void MeshApp::prepare()
{
	Application::prepare();

	// Create the Vulkan instance and surface for the application
	m_instance = std::make_unique<Instance>(getName());

	platform.createSurface(m_instance->getHandle());
	m_surface = platform.getSurface();

	// Create the Vulkan physical and logical device with the required extensions
	std::unique_ptr<PhysicalDevice> physicalDevice = m_instance->getSuitablePhysicalDevice();

	m_device = std::make_unique<Device>(std::move(physicalDevice), m_surface, deviceExtensions);

	// Find general queue for graphics, compute, transfer and presentation
	int32_t desiredQueueFamilyIndex = m_device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, true);

	if (desiredQueueFamilyIndex >= 0)
	{
		m_generalQueue = m_device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
	}
	else
	{
		LOGEANDABORT("TODO: Devices where a queue supporting graphics, compute and transfer do not exist are not supported yet");
	}

	// Create swapchain
	const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT };
	m_swapchain = std::make_unique<Swapchain>(*m_device, m_surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags, m_generalQueue->getFamilyIndex(), m_generalQueue->getFamilyIndex());

	createSwapchainImageViews();
	createFrameData();
}

void MeshApp::createSwapchainImageViews()
{
	m_swapchainImageViews.reserve(m_swapchain->getImages().size());

	for (const std::unique_ptr<Image> &swapchainImage : m_swapchain->getImages())
	{
		VkImageViewCreateInfo imageViewCreateInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchainImage->getHandle(),
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = m_swapchain->getProperties().surfaceFormat.format,
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u },
		};

		VkImageView imageView{ VK_NULL_HANDLE };
		VK_CHECK(vkCreateImageView(m_device->getHandle(), &imageViewCreateInfo, nullptr, &imageView));

		m_swapchainImageViews.emplace_back(imageView);
	}
}

void MeshApp::createFrameData()
{
	m_fencePool = std::make_unique<FencePool>(*m_device);
	m_semaphorePool = std::make_unique<SemaphorePool>(*m_device);

	for (FrameData &frame : m_frames)
	{
		frame.commandPool = std::make_unique<CommandPool>(*m_device, m_generalQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		frame.commandBuffer = std::make_shared<CommandBuffer>(*frame.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		frame.inFlightFence = m_fencePool->requestFence();
		frame.imageAvailableSemaphore = m_semaphorePool->requestSemaphore();
	}

	// One render finished semaphore per swapchain image: presentation may still be using an image by the time the frame
	// that signalled its semaphore comes around again, so the semaphore cannot belong to a frame
	m_renderFinishedSemaphores.resize(m_swapchainImageViews.size(), VK_NULL_HANDLE);
	for (VkSemaphore &renderFinishedSemaphore : m_renderFinishedSemaphores)
	{
		renderFinishedSemaphore = m_semaphorePool->requestSemaphore();
	}
}

void MeshApp::update()
{
	FrameData &frame = m_frames[m_currentFrame];

	// Nothing may be recorded into this frame's command buffer until the GPU is done with it
	m_fencePool->wait(&frame.inFlightFence);
	m_fencePool->reset(&frame.inFlightFence);

	VkResult result = vkAcquireNextImageKHR(m_device->getHandle(), m_swapchain->getHandle(), std::numeric_limits<uint64_t>::max(), frame.imageAvailableSemaphore, VK_NULL_HANDLE, &m_swapchainImageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		LOGEANDABORT("Failed to acquire the next swapchain image");
	}

	recordCommandBuffer();

	// The acquire semaphore is waited on at the color attachment stage because that is the first stage that touches the image
	VkSemaphoreSubmitInfo imageAvailableSemaphoreSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = frame.imageAvailableSemaphore,
		.value = 0u, // Ignored for binary semaphores
		.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.deviceIndex = 0u,
	};
	VkSemaphoreSubmitInfo renderFinishedSemaphoreSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = m_renderFinishedSemaphores[m_swapchainImageIndex],
		.value = 0u, // Ignored for binary semaphores
		.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.deviceIndex = 0u,
	};
	VkCommandBufferSubmitInfo commandBufferSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.commandBuffer = frame.commandBuffer->getHandle(),
		.deviceMask = 0u,
	};
	VkSubmitInfo2 submitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.waitSemaphoreInfoCount = 1u,
		.pWaitSemaphoreInfos = &imageAvailableSemaphoreSubmitInfo,
		.commandBufferInfoCount = 1u,
		.pCommandBufferInfos = &commandBufferSubmitInfo,
		.signalSemaphoreInfoCount = 1u,
		.pSignalSemaphoreInfos = &renderFinishedSemaphoreSubmitInfo,
	};
	VK_CHECK(vkQueueSubmit2(m_generalQueue->getHandle(), 1u, &submitInfo, frame.inFlightFence));

	const VkSwapchainKHR swapchainHandle = m_swapchain->getHandle();
	VkPresentInfoKHR presentInfo{
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1u,
		.pWaitSemaphores = &m_renderFinishedSemaphores[m_swapchainImageIndex],
		.swapchainCount = 1u,
		.pSwapchains = &swapchainHandle,
		.pImageIndices = &m_swapchainImageIndex,
	};

	result = vkQueuePresentKHR(m_generalQueue->getHandle(), &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		recreateSwapchain();
	}
	else if (result != VK_SUCCESS)
	{
		LOGEANDABORT("Failed to present the swapchain image");
	}

	m_currentFrame = (m_currentFrame + 1u) % maxFramesInFlight;
}

void MeshApp::recordCommandBuffer()
{
	FrameData &frame = m_frames[m_currentFrame];
	VkCommandBuffer commandBuffer = frame.commandBuffer->getHandle();
	frame.commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	const VkImage swapchainImage = m_swapchain->getImages()[m_swapchainImageIndex]->getHandle();
	const VkImageSubresourceRange subresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u };

	// Take ownership of the image from the presentation engine. The acquire semaphore is waited on at the color attachment
	// stage, so the transition has to be ordered against that stage for the layout transition not to race with the acquire
	VkImageMemoryBarrier2 toColorAttachmentBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_NONE,
		.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchainImage,
		.subresourceRange = subresourceRange,
	};
	VkDependencyInfo toColorAttachmentDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toColorAttachmentBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toColorAttachmentDependencyInfo);

	// Clear the image without a render pass or a framebuffer
	VkClearValue clearColor{};
	clearColor.color = { { 0.05f, 0.05f, 0.08f, 1.0f } };

	VkRenderingAttachmentInfo colorAttachment{
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = m_swapchainImageViews[m_swapchainImageIndex],
		.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = clearColor,
	};
	VkRenderingInfo renderingInfo{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { { 0u, 0u }, m_swapchain->getProperties().imageExtent },
		.layerCount = 1u,
		.colorAttachmentCount = 1u,
		.pColorAttachments = &colorAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &renderingInfo);
	vkCmdEndRendering(commandBuffer);

	// Hand the image back to the presentation engine
	VkImageMemoryBarrier2 toPresentBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
		.dstAccessMask = VK_ACCESS_2_NONE,
		.oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchainImage,
		.subresourceRange = subresourceRange,
	};
	VkDependencyInfo toPresentDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toPresentBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toPresentDependencyInfo);

	frame.commandBuffer->end();
}

void MeshApp::recreateSwapchain()
{
	// The platform window is not resizable, so rebuilding the swapchain is not needed yet
	LOGW("Swapchain is out of date; recreation is not implemented yet");
}

void MeshApp::cleanup()
{
	if (m_device == nullptr)
	{
		return;
	}

	m_device->waitIdle();

	for (VkImageView imageView : m_swapchainImageViews)
	{
		vkDestroyImageView(m_device->getHandle(), imageView, nullptr);
	}
	m_swapchainImageViews.clear();
	m_renderFinishedSemaphores.clear();

	// A CommandBuffer holds a reference to its CommandPool, so the command buffers have to be released before the pools.
	// Assigning a default constructed FrameData here would release the members in declaration order (pool first) instead
	for (FrameData &frame : m_frames)
	{
		frame.commandBuffer.reset();
		frame.commandPool.reset();
	}

	m_fencePool.reset();
	m_semaphorePool.reset();

	m_swapchain.reset();
}

} // namespace vulkr
