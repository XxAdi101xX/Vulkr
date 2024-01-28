/* Copyright (c) 2023 Adithya Venkatarao
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

#include "fluid_simulation.h"
#include "platform/platform.h"

namespace vulkr
{

FluidSimulation::FluidSimulation(Platform &platform, std::string name) : Application{ platform, name } {}

FluidSimulation::~FluidSimulation()
{
	device->waitIdle();

	semaphorePool.reset();
	fencePool.reset();

	cleanupSwapchain();

	textureSampler.reset();

	fluidSimulationInputDescriptorSetLayout.reset();
	fluidSimulationOutputDescriptorSetLayout.reset();

	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.commandPools[i].reset();
	}

	device.reset();

	if (m_surface != VK_NULL_HANDLE)
	{
		vkDestroySurfaceKHR(m_instance->getHandle(), m_surface, nullptr);
	}

	m_instance.reset();
}

void FluidSimulation::cleanupSwapchain()
{
	// Command buffers
	for (uint32_t i = 0u; i < maxFramesInFlight; ++i)
	{
		for (uint32_t j = 0u; j < commandBufferCountForFrame; ++j)
		{
			frameData.commandBuffers[i][j].reset();
		}
	}

	// Pipelines
	pipelines.computeVelocityAdvection.pipeline.reset();
	pipelines.computeVelocityAdvection.pipelineState.reset();
	pipelines.computeDensityAdvection.pipeline.reset();
	pipelines.computeDensityAdvection.pipelineState.reset();
	pipelines.computeVelocityGaussianSplat.pipeline.reset();
	pipelines.computeVelocityGaussianSplat.pipelineState.reset();
	pipelines.computeDensityGaussianSplat.pipeline.reset();
	pipelines.computeDensityGaussianSplat.pipelineState.reset();
	pipelines.computeFluidVelocityDivergence.pipeline.reset();
	pipelines.computeFluidVelocityDivergence.pipelineState.reset();
	pipelines.computeJacobi.pipeline.reset();
	pipelines.computeJacobi.pipelineState.reset();
	pipelines.computeGradientSubtraction.pipeline.reset();
	pipelines.computeGradientSubtraction.pipelineState.reset();

	// Swapchain
	swapchain.reset();

	// Descriptor Sets
	fluidSimulationInputDescriptorSet.reset();
	fluidSimulationOutputDescriptorSet.reset();

	// Textures
	fluidVelocityInputImageView.reset();
	fluidVelocityDivergenceInputImageView.reset();
	fluidPressureInputImageView.reset();
	fluidDensityInputImageView.reset();
	fluidSimulationOutputImageView.reset();

	descriptorPool.reset();

	cameraController.reset();
}

void FluidSimulation::prepare()
{
	Application::prepare();

	createInstance();
	createSurface();
	createDevice();

	// For simplicity, we will try to get a queue that supports graphics, compute, transfer and present
	int32_t desiredQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, true);

	// TODO: check if ray tracing requires a compute queue, currently it is just using the graphics queue and not checking for compute capabilities
	if (desiredQueueFamilyIndex >= 0)
	{
		m_graphicsQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
		m_computeQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
		m_presentQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
		m_transferQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
	}
	else
	{
		// TODO: add concurrency with separate transfer queues
		LOGEANDABORT("TODO: Devices where a queue supporting graphics, compute and transfer do not exist are not supported yet");

		// Find a queue that supports graphics and present operations
		int32_t desiredGraphicsQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT, true);
		if (desiredGraphicsQueueFamilyIndex >= 0)
		{
			m_graphicsQueue = device->getQueue(to_u32(desiredGraphicsQueueFamilyIndex), 0u);
			m_presentQueue = device->getQueue(to_u32(desiredGraphicsQueueFamilyIndex), 0u);
		}
		else
		{
			int32_t graphicsQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT, false);
			int32_t presentQueueFamilyIndex = device->getQueueFamilyIndexByFlags(0u, true);

			if (graphicsQueueFamilyIndex < 0 || presentQueueFamilyIndex < 0)
			{
				LOGEANDABORT("Unable to find a queue that supports graphics and/or presentation")
			}
			m_graphicsQueue = device->getQueue(to_u32(graphicsQueueFamilyIndex), 0u);
			m_presentQueue = device->getQueue(to_u32(presentQueueFamilyIndex), 0u);
		}

		// Find a queue that supports transfer operations
		int32_t desiredTransferQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_TRANSFER_BIT, false);
		if (desiredTransferQueueFamilyIndex < 0)
		{
			LOGEANDABORT("Unable to find a queue that supports transfer operations");
			m_transferQueue = device->getQueue(to_u32(desiredTransferQueueFamilyIndex), 0u);
		}
	}

	createSwapchain();
	createCommandPools();
	createCommandBuffers();

	createDescriptorSetLayouts();
	createTextureSampler();
	initializeFluidSimulationResources();
	createVelocityAdvectionComputePipeline();
	createDensityAdvectionComputePipeline();
	createVelocityGaussianSplatComputePipeline();
	createDensityGaussianSplatComputePipeline();
	createFluidVelocityDivergenceComputePipeline();
	createJacobiComputePipeline();
	createGradientSubtractionComputePipeline();
	createDescriptorPool();
	createDescriptorSets();
	setupCamera();

	createSemaphoreAndFencePools();
	setupSynchronizationObjects();
}

void FluidSimulation::update()
{
	fencePool->wait(&frameData.inFlightFences[currentFrame]);
	fencePool->reset(&frameData.inFlightFences[currentFrame]);

	// Now that we are sure that the commands finished executing, we can safely reset the command buffers for the frame to begin recording again
	for (uint32_t i = 0u; i < commandBufferCountForFrame; ++i)
	{
		frameData.commandBuffers[currentFrame][i]->reset();
	}

	uint32_t swapchainImageIndex;
	VkResult result = vkAcquireNextImageKHR(device->getHandle(), swapchain->getHandle(), std::numeric_limits<uint64_t>::max(), frameData.imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &swapchainImageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		LOGEANDABORT("Failed to acquire swap chain image");
	}

	if (imagesInFlight[swapchainImageIndex] != VK_NULL_HANDLE)
	{
		fencePool->wait(&imagesInFlight[swapchainImageIndex]);
	}
	imagesInFlight[swapchainImageIndex] = frameData.inFlightFences[currentFrame];

	// Begin command buffer for offscreen pass
	frameData.commandBuffers[currentFrame][0]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	// Compute shader invocations
	computeFluidSimulation();

	// Add memory barrier to ensure that the particleIntegrate computer shader has finished writing to the currentFrameObjInstanceBuffer
	VkMemoryBarrier2 memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
	memoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
	memoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
	memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;

	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 1u;
	dependencyInfo.pMemoryBarriers = &memoryBarrier;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 0u;
	dependencyInfo.pImageMemoryBarriers = nullptr;

	// TODO: current commandBuffer[0] is empty and has no work when submitted
	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);
	/*
	frameData.commandBuffers[currentFrame][0]->beginRenderPass(*mainRenderPass.renderPass, *(frameData.offscreenFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, offscreenFramebufferClearValues, VK_SUBPASS_CONTENTS_INLINE);
	rasterize();
	frameData.commandBuffers[currentFrame][0]->endRenderPass();
	*/

	// End command buffer for offscreen pass
	frameData.commandBuffers[currentFrame][0]->end();

	// I have setup a subpass dependency to ensure that the render pass waits for the swapchain to finish reading from the image before accessing it
	// hence I don't need to set the wait stages to VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT 
	VkCommandBufferSubmitInfo offscreenCommandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	offscreenCommandBufferSubmitInfo.pNext = nullptr;
	offscreenCommandBufferSubmitInfo.commandBuffer = frameData.commandBuffers[currentFrame][0]->getHandle();
	offscreenCommandBufferSubmitInfo.deviceMask = 0u;

	VkSemaphoreSubmitInfo offScreenWaitSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	offScreenWaitSemaphoreSubmitInfo.pNext = nullptr;
	offScreenWaitSemaphoreSubmitInfo.semaphore = frameData.imageAvailableSemaphores[currentFrame];
	offScreenWaitSemaphoreSubmitInfo.value = 0u; // Optional: ignored since this isn't a timeline semaphore
	offScreenWaitSemaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	offScreenWaitSemaphoreSubmitInfo.deviceIndex = 0u; // replaces VkDeviceGroupSubmitInfo but we don't have that in our pNext chain so not used.

	VkSemaphoreSubmitInfo offScreenSignalSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	offScreenSignalSemaphoreSubmitInfo.pNext = nullptr;
	offScreenSignalSemaphoreSubmitInfo.semaphore = frameData.offscreenRenderingFinishedSemaphores[currentFrame];
	offScreenSignalSemaphoreSubmitInfo.value = 0u; // Optional: ignored since this isn't a timeline semaphore
	offScreenSignalSemaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	offScreenSignalSemaphoreSubmitInfo.deviceIndex = 0u; // replaces VkDeviceGroupSubmitInfo but we don't have that in our pNext chain so not used.

	VkSubmitInfo2 offscreenPassSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	offscreenPassSubmitInfo.pNext = nullptr;
	offscreenPassSubmitInfo.flags = 0u;
	offscreenPassSubmitInfo.waitSemaphoreInfoCount = 1u;
	offscreenPassSubmitInfo.pWaitSemaphoreInfos = &offScreenWaitSemaphoreSubmitInfo;
	offscreenPassSubmitInfo.commandBufferInfoCount = 1u;
	offscreenPassSubmitInfo.pCommandBufferInfos = &offscreenCommandBufferSubmitInfo;
	offscreenPassSubmitInfo.signalSemaphoreInfoCount = 1u;
	offscreenPassSubmitInfo.pSignalSemaphoreInfos = &offScreenSignalSemaphoreSubmitInfo;

	// TODO: current commandBuffer[1] is empty and has no work when submitted
	// Begin command buffer for post process pass
	frameData.commandBuffers[currentFrame][1]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	/*
	// Post offscreen renderpass
	frameData.commandBuffers[currentFrame][1]->beginRenderPass(*postRenderPass.renderPass, *(frameData.postProcessFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, postProcessFramebufferClearValues, VK_SUBPASS_CONTENTS_INLINE);
	if (temporalAntiAliasingEnabled)
	{
		postProcess();
	}

	frameData.commandBuffers[currentFrame][1]->endRenderPass();
	*/

	// End command buffer for post process pass
	frameData.commandBuffers[currentFrame][1]->end();

	// Wait on postProcess pass to complete before copy operations
	// I have setup a subpass dependency to ensure that the render pass waits for the swapchain to finish reading from the image before accessing it
	// hence I don't need to set the wait stages to VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT 
	VkCommandBufferSubmitInfo postProcessCommandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	postProcessCommandBufferSubmitInfo.pNext = nullptr;
	postProcessCommandBufferSubmitInfo.commandBuffer = frameData.commandBuffers[currentFrame][1]->getHandle();
	postProcessCommandBufferSubmitInfo.deviceMask = 0u;

	VkSemaphoreSubmitInfo postProcessWaitSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	postProcessWaitSemaphoreSubmitInfo.pNext = nullptr;
	postProcessWaitSemaphoreSubmitInfo.semaphore = frameData.offscreenRenderingFinishedSemaphores[currentFrame];
	postProcessWaitSemaphoreSubmitInfo.value = 0u; // Optional: ignored since this isn't a timeline semaphore
	postProcessWaitSemaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	postProcessWaitSemaphoreSubmitInfo.deviceIndex = 0u; // replaces VkDeviceGroupSubmitInfo but we don't have that in our pNext chain so not used.

	VkSemaphoreSubmitInfo postProcessSignalSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	postProcessSignalSemaphoreSubmitInfo.pNext = nullptr;
	postProcessSignalSemaphoreSubmitInfo.semaphore = frameData.postProcessRenderingFinishedSemaphores[currentFrame];
	postProcessSignalSemaphoreSubmitInfo.value = 0u; // Optional: ignored since this isn't a timeline semaphore
	postProcessSignalSemaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	postProcessSignalSemaphoreSubmitInfo.deviceIndex = 0u; // replaces VkDeviceGroupSubmitInfo but we don't have that in our pNext chain so not used.

	VkSubmitInfo2 postProcessPassSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	postProcessPassSubmitInfo.pNext = nullptr;
	postProcessPassSubmitInfo.flags = 0u;
	postProcessPassSubmitInfo.waitSemaphoreInfoCount = 1u;
	postProcessPassSubmitInfo.pWaitSemaphoreInfos = &postProcessWaitSemaphoreSubmitInfo;
	postProcessPassSubmitInfo.commandBufferInfoCount = 1u;
	postProcessPassSubmitInfo.pCommandBufferInfos = &postProcessCommandBufferSubmitInfo;
	postProcessPassSubmitInfo.signalSemaphoreInfoCount = 1u;
	postProcessPassSubmitInfo.pSignalSemaphoreInfos = &postProcessSignalSemaphoreSubmitInfo;

	// Begin command buffer for outputImage copy operations to swapchain and history buffer
	frameData.commandBuffers[currentFrame][2]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0u;
	subresourceRange.levelCount = 1u;
	subresourceRange.baseArrayLayer = 0u;
	subresourceRange.layerCount = 1u;

	// Prepare the current swapchain as a transfer destination
	VkImageMemoryBarrier2 transitionSwapchainLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionSwapchainLayoutBarrier.pNext = nullptr;
	transitionSwapchainLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // We have a semaphore that synchronizes the post processing pass with all of the things happening on frameData.commandBuffers[currentFrame][2]->getHandle()
	transitionSwapchainLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionSwapchainLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	transitionSwapchainLayoutBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionSwapchainLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	transitionSwapchainLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionSwapchainLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionSwapchainLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionSwapchainLayoutBarrier.image = swapchain->getImages()[swapchainImageIndex]->getHandle();
	transitionSwapchainLayoutBarrier.subresourceRange = subresourceRange;

	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionSwapchainLayoutBarrier;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][2]->getHandle(), &dependencyInfo);

	// Transition density texture layout to prepare as a blit source
	VkImageMemoryBarrier2 transitionDensityTextureBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionDensityTextureBarrier.pNext = nullptr;
	transitionDensityTextureBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // We have a semaphore that synchronizes the post processing pass with all of the things happening on frameData.commandBuffers[currentFrame][2]->getHandle()
	transitionDensityTextureBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionDensityTextureBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	transitionDensityTextureBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	transitionDensityTextureBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionDensityTextureBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	transitionDensityTextureBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionDensityTextureBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionDensityTextureBarrier.image = fluidDensityInputImageView->getImage()->getHandle();
	transitionDensityTextureBarrier.subresourceRange = subresourceRange;

	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionDensityTextureBarrier;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][2]->getHandle(), &dependencyInfo); // TODO combine the pipelineBarrier2 call for the swapchain with the one below

	VkImageBlit2 imageBlit{ VK_STRUCTURE_TYPE_IMAGE_BLIT_2 };
	imageBlit.pNext = nullptr;
	imageBlit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	imageBlit.srcOffsets[0] = { 0, 0, 0 };
	imageBlit.srcOffsets[1] = { static_cast<int32_t>(swapchain->getProperties().imageExtent.width), static_cast<int32_t>(swapchain->getProperties().imageExtent.height), 1 };
	imageBlit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	imageBlit.dstOffsets[0] = { 0, 0, 0 };
	imageBlit.dstOffsets[1] = { static_cast<int32_t>(swapchain->getProperties().imageExtent.width), static_cast<int32_t>(swapchain->getProperties().imageExtent.height), 1 };

	VkBlitImageInfo2 blitImageInfo{ VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2 };
	blitImageInfo.pNext = nullptr;
	blitImageInfo.srcImage = fluidDensityInputImageView->getImage()->getHandle();
	blitImageInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	blitImageInfo.dstImage = swapchain->getImages()[swapchainImageIndex]->getHandle();
	blitImageInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	blitImageInfo.regionCount = 1u;
	blitImageInfo.pRegions = &imageBlit;
	blitImageInfo.filter = VK_FILTER_NEAREST;

	// We are using vkCmdBlitImage2 instead of vkCmdCopyImage to deal with the format conversions from VK_FORMAT_R32G32B32A32_SFLOAT to VK_FORMAT_B8G8R8A8_UNORM
	vkCmdBlitImage2(frameData.commandBuffers[currentFrame][2]->getHandle(), &blitImageInfo);

	// Transition density texture back to general from being a blit source
	transitionDensityTextureBarrier.pNext = nullptr;
	transitionDensityTextureBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait on transfer to finish
	transitionDensityTextureBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	transitionDensityTextureBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE; // No further synchronization required on densityTextureBuffer
	transitionDensityTextureBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionDensityTextureBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	transitionDensityTextureBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionDensityTextureBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionDensityTextureBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionDensityTextureBarrier.image = fluidDensityInputImageView->getImage()->getHandle();
	transitionDensityTextureBarrier.subresourceRange = subresourceRange;

	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionDensityTextureBarrier;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][2]->getHandle(), &dependencyInfo);

	// Transition the current swapchain image back for presentation
	transitionSwapchainLayoutBarrier.pNext = nullptr;
	transitionSwapchainLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait for transfer to finish
	transitionSwapchainLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionSwapchainLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionSwapchainLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionSwapchainLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionSwapchainLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	transitionSwapchainLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionSwapchainLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionSwapchainLayoutBarrier.image = swapchain->getImages()[swapchainImageIndex]->getHandle();
	transitionSwapchainLayoutBarrier.subresourceRange = subresourceRange;

	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionSwapchainLayoutBarrier;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][2]->getHandle(), &dependencyInfo);

	// End command buffer for copy operations
	frameData.commandBuffers[currentFrame][2]->end();

	// Wait on post process pass before doing any transfer on resources written to during the previous stages
	VkCommandBufferSubmitInfo outputImageTransferCommandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	outputImageTransferCommandBufferSubmitInfo.pNext = nullptr;
	outputImageTransferCommandBufferSubmitInfo.commandBuffer = frameData.commandBuffers[currentFrame][2]->getHandle();
	outputImageTransferCommandBufferSubmitInfo.deviceMask = 0u;

	VkSemaphoreSubmitInfo outputImageTransferWaitSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	outputImageTransferWaitSemaphoreSubmitInfo.pNext = nullptr;
	outputImageTransferWaitSemaphoreSubmitInfo.semaphore = frameData.postProcessRenderingFinishedSemaphores[currentFrame];
	outputImageTransferWaitSemaphoreSubmitInfo.value = 0ull; // Optional: ignored since this isn't a timeline semaphore
	outputImageTransferWaitSemaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	outputImageTransferWaitSemaphoreSubmitInfo.deviceIndex = 0u; // replaces VkDeviceGroupSubmitInfo but we don't have that in our pNext chain so not used.

	VkSemaphoreSubmitInfo outputImageTransferSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	outputImageTransferSemaphoreSubmitInfo.pNext = nullptr;
	outputImageTransferSemaphoreSubmitInfo.semaphore = frameData.outputImageCopyFinishedSemaphores[currentFrame];
	outputImageTransferSemaphoreSubmitInfo.value = 0ull; // Optional: ignored since this isn't a timeline semaphore
	outputImageTransferSemaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	outputImageTransferSemaphoreSubmitInfo.deviceIndex = 0u; // replaces VkDeviceGroupSubmitInfo but we don't have that in our pNext chain so not used.

	VkSubmitInfo2 outputImageTransferPassSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	outputImageTransferPassSubmitInfo.pNext = nullptr;
	outputImageTransferPassSubmitInfo.flags = 0u;
	outputImageTransferPassSubmitInfo.waitSemaphoreInfoCount = 1u;
	outputImageTransferPassSubmitInfo.pWaitSemaphoreInfos = &outputImageTransferWaitSemaphoreSubmitInfo;
	outputImageTransferPassSubmitInfo.commandBufferInfoCount = 1u;
	outputImageTransferPassSubmitInfo.pCommandBufferInfos = &outputImageTransferCommandBufferSubmitInfo;
	outputImageTransferPassSubmitInfo.signalSemaphoreInfoCount = 1u;
	outputImageTransferPassSubmitInfo.pSignalSemaphoreInfos = &outputImageTransferSemaphoreSubmitInfo;

	std::array<VkSubmitInfo2, 3> submitInfo{ offscreenPassSubmitInfo, postProcessPassSubmitInfo, outputImageTransferPassSubmitInfo };

	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), to_u32(submitInfo.size()), submitInfo.data(), frameData.inFlightFences[currentFrame]));

	VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	presentInfo.waitSemaphoreCount = 1u;
	presentInfo.pWaitSemaphores = &frameData.outputImageCopyFinishedSemaphores[currentFrame];

	std::array<VkSwapchainKHR, 1> swapchains{ swapchain->getHandle() };
	presentInfo.swapchainCount = to_u32(swapchains.size());
	presentInfo.pSwapchains = swapchains.data();

	presentInfo.pImageIndices = &swapchainImageIndex;

	result = vkQueuePresentKHR(m_presentQueue->getHandle(), &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		recreateSwapchain();
	}
	else if (result != VK_SUCCESS)
	{
		LOGEANDABORT("Failed to present swap chain image!");
	}

	currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

void FluidSimulation::recreateSwapchain()
{
	LOGEANDABORT("Swapchain recreation is not supported");
}

void FluidSimulation::handleInputEvents(const InputEvent &inputEvent)
{
	cameraController->handleInputEvents(inputEvent);

	if (inputEvent.getEventSource() == EventSource::Mouse)
	{
		const MouseInputEvent &mouseInputEvent = dynamic_cast<const MouseInputEvent &>(inputEvent);

		if (mouseInputEvent.getAction() == MouseAction::Unknown)
		{
			LOGEANDABORT("Unknown mouse action encountered");
		}

		if (mouseInputEvent.getInput() == MouseInput::None)
		{
			// User is holding down left click and moving
			if (activeMouseInput == MouseInput::Left)
			{
				fluidSimulationPushConstant.splatForce = glm::vec3(mouseInputEvent.getPositionX() - lastMousePosition.x, mouseInputEvent.getPositionY() - lastMousePosition.y, 0.0f);
				fluidSimulationPushConstant.splatPosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
				lastMousePosition.x = static_cast<float>(mouseInputEvent.getPositionX());
				lastMousePosition.y = static_cast<float>(mouseInputEvent.getPositionY());
			}
		}
		else if (mouseInputEvent.getInput() == MouseInput::Left)
		{
			if (mouseInputEvent.getAction() == MouseAction::Click)
			{
				activeMouseInput = mouseInputEvent.getInput();
				lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
			}
			else if (mouseInputEvent.getAction() == MouseAction::Release)
			{
				activeMouseInput = MouseInput::None;
			}
			else
			{
				LOGEANDABORT("Mouse input action is neither click or release");
			}
		}
	}
}

/* Private methods start here */
void FluidSimulation::copyFluidOutputTextureToInputTexture(const Image *imageToCopyTo)
{
	// Layout transitions for the fluidSimulationOutputImageView as a transfer src and the imageToCopyTo as the transfer dst
	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0u;
	subresourceRange.levelCount = 1u;
	subresourceRange.baseArrayLayer = 0u;
	subresourceRange.layerCount = 1u;

	VkImageMemoryBarrier2 transitionImageToCopyLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionImageToCopyLayoutBarrier.pNext = nullptr;
	transitionImageToCopyLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; // The imageToCopyTo is read in a compute shader
	transitionImageToCopyLayoutBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
	transitionImageToCopyLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Image will be written to
	transitionImageToCopyLayoutBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionImageToCopyLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionImageToCopyLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionImageToCopyLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionImageToCopyLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionImageToCopyLayoutBarrier.image = imageToCopyTo->getHandle();
	transitionImageToCopyLayoutBarrier.subresourceRange = subresourceRange;

	VkImageMemoryBarrier2 transitionFluidSimulationOutputLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionFluidSimulationOutputLayoutBarrier.pNext = nullptr;
	transitionFluidSimulationOutputLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; // The fluidSimluationOutputTexture is written to in a compute shader
	transitionFluidSimulationOutputLayoutBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
	transitionFluidSimulationOutputLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Image will be read as transfer source
	transitionFluidSimulationOutputLayoutBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	transitionFluidSimulationOutputLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionFluidSimulationOutputLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	transitionFluidSimulationOutputLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionFluidSimulationOutputLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionFluidSimulationOutputLayoutBarrier.image = fluidSimulationOutputImageView->getImage()->getHandle();
	transitionFluidSimulationOutputLayoutBarrier.subresourceRange = subresourceRange;

	std::array<VkImageMemoryBarrier2, 2> imageTransitionForTransferMemoryBarriers{ transitionImageToCopyLayoutBarrier, transitionFluidSimulationOutputLayoutBarrier };
	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = to_u32(imageTransitionForTransferMemoryBarriers.size());
	dependencyInfo.pImageMemoryBarriers = imageTransitionForTransferMemoryBarriers.data();

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);

	// Copy fluidVelocityOutputTextures to imageToCopyTo
	VkImageCopy fluidVelocityTextureCopyRegion{};
	fluidVelocityTextureCopyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };
	fluidVelocityTextureCopyRegion.srcOffset = { 0, 0, 0 };
	fluidVelocityTextureCopyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };
	fluidVelocityTextureCopyRegion.dstOffset = { 0, 0, 0 };
	fluidVelocityTextureCopyRegion.extent = { swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1u };

	vkCmdCopyImage(frameData.commandBuffers[currentFrame][0]->getHandle(), fluidSimulationOutputImageView->getImage()->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageToCopyTo->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &fluidVelocityTextureCopyRegion);

	// Layout transitions for the fluidVelocityOutputTextures and the imageToCopyTo as general
	transitionImageToCopyLayoutBarrier.pNext = nullptr;
	transitionImageToCopyLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait for write transfer to finish
	transitionImageToCopyLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionImageToCopyLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; // The texture will be used as as a read target in a compute shader
	transitionImageToCopyLayoutBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
	transitionImageToCopyLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionImageToCopyLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionImageToCopyLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionImageToCopyLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionImageToCopyLayoutBarrier.image = imageToCopyTo->getHandle();
	transitionImageToCopyLayoutBarrier.subresourceRange = subresourceRange;

	transitionFluidSimulationOutputLayoutBarrier.pNext = nullptr;
	transitionFluidSimulationOutputLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait for transfer to finish as the src
	transitionFluidSimulationOutputLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	transitionFluidSimulationOutputLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; // Image will be written to in a compute shader
	transitionFluidSimulationOutputLayoutBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
	transitionFluidSimulationOutputLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	transitionFluidSimulationOutputLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionFluidSimulationOutputLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionFluidSimulationOutputLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionFluidSimulationOutputLayoutBarrier.image = fluidSimulationOutputImageView->getImage()->getHandle();
	transitionFluidSimulationOutputLayoutBarrier.subresourceRange = subresourceRange;

	std::array<VkImageMemoryBarrier2, 2> imageTransitionForComputeUsageMemoryBarriers{ transitionImageToCopyLayoutBarrier, transitionFluidSimulationOutputLayoutBarrier };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = to_u32(imageTransitionForComputeUsageMemoryBarriers.size());
	dependencyInfo.pImageMemoryBarriers = imageTransitionForComputeUsageMemoryBarriers.data();

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);
}

void FluidSimulation::computeFluidSimulation()
{
	// Acquire
	if (m_graphicsQueue->getFamilyIndex() != m_computeQueue->getFamilyIndex())
	{
		LOGEANDABORT("Gotta verify this logic since we have assumed that computeQueue == graphicsQueue so far");
		// TODO this code has not been tested or verified; gotta check if the src and dst stage/access values are correct
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0u;
		subresourceRange.levelCount = 1u;
		subresourceRange.baseArrayLayer = 0u;
		subresourceRange.layerCount = 1u;

		VkImageMemoryBarrier2 fluidVelocityInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidVelocityInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		fluidVelocityInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		fluidVelocityInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidVelocityInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidVelocityInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidVelocityInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidVelocityInputTextureImageMemoryBarrier.image = fluidVelocityInputImageView->getImage()->getHandle();
		fluidVelocityInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidVelocityDivergenceInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.image = fluidVelocityDivergenceInputImageView->getImage()->getHandle();
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidPressureInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidPressureInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		fluidPressureInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		fluidPressureInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidPressureInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidPressureInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidPressureInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidPressureInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidPressureInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidPressureInputTextureImageMemoryBarrier.image = fluidPressureInputImageView->getImage()->getHandle();
		fluidPressureInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidDensityInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidDensityInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		fluidDensityInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		fluidDensityInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidDensityInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidDensityInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidDensityInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidDensityInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidDensityInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidDensityInputTextureImageMemoryBarrier.image = fluidDensityInputImageView->getImage()->getHandle();
		fluidDensityInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidSimulationOutputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidSimulationOutputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		fluidSimulationOutputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		fluidSimulationOutputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidSimulationOutputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidSimulationOutputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidSimulationOutputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidSimulationOutputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidSimulationOutputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidSimulationOutputTextureImageMemoryBarrier.image = fluidSimulationOutputImageView->getImage()->getHandle();
		fluidSimulationOutputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		std::array<VkImageMemoryBarrier2, 5> imageMemoryBarriers
		{
			fluidVelocityInputTextureImageMemoryBarrier,
				fluidVelocityDivergenceInputTextureImageMemoryBarrier,
				fluidPressureInputTextureImageMemoryBarrier,
				fluidDensityInputTextureImageMemoryBarrier,
				fluidSimulationOutputTextureImageMemoryBarrier
		};
		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = to_u32(imageMemoryBarriers.size());
		dependencyInfo.pImageMemoryBarriers = imageMemoryBarriers.data();

		vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);
	}

	// Update the time step
	fluidSimulationPushConstant.timestep += (1.0f / 60.0f); // TODO: tie this to fps?

	uint32_t fluidSimulationGridSize = to_u32(fluidSimulationPushConstant.gridSize.x * fluidSimulationPushConstant.gridSize.y);

	// TODO: for both the advection and gaussian splat shaders, we can refactor to just have one of each shader with differetn descriptor sets instead
	// First pass: Compute velocity advection
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityAdvection.pipeline->getBindPoint(), pipelines.computeVelocityAdvection.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityAdvection.pipeline->getBindPoint(), pipelines.computeVelocityAdvection.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityAdvection.pipeline->getBindPoint(), pipelines.computeVelocityAdvection.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityAdvection.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);

	// Ensures that compute shader has finished its writing before the transfer operation is done and ensures that it completes before any future compute operations
	copyFluidOutputTextureToInputTexture(fluidVelocityInputImageView->getImage());

	// Second pass: Compute density advection
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityAdvection.pipeline->getBindPoint(), pipelines.computeDensityAdvection.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityAdvection.pipeline->getBindPoint(), pipelines.computeDensityAdvection.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityAdvection.pipeline->getBindPoint(), pipelines.computeDensityAdvection.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityAdvection.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);

	copyFluidOutputTextureToInputTexture(fluidDensityInputImageView->getImage());

	// Third pass: Compute velocity and density gaussian splat from key input; if splatForce is 0, we don't have to run this pass
	if (fluidSimulationPushConstant.splatForce != glm::vec3(0.0f))
	{
		vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityGaussianSplat.pipeline->getBindPoint(), pipelines.computeVelocityGaussianSplat.pipeline->getHandle());
		vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityGaussianSplat.pipeline->getBindPoint(), pipelines.computeVelocityGaussianSplat.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
		vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityGaussianSplat.pipeline->getBindPoint(), pipelines.computeVelocityGaussianSplat.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
		vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeVelocityGaussianSplat.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);
		vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);

		copyFluidOutputTextureToInputTexture(fluidVelocityInputImageView->getImage());

		vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityGaussianSplat.pipeline->getBindPoint(), pipelines.computeDensityGaussianSplat.pipeline->getHandle());
		vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityGaussianSplat.pipeline->getBindPoint(), pipelines.computeDensityGaussianSplat.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
		vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityGaussianSplat.pipeline->getBindPoint(), pipelines.computeDensityGaussianSplat.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
		vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeDensityGaussianSplat.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);
		vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);

		copyFluidOutputTextureToInputTexture(fluidDensityInputImageView->getImage());


		fluidSimulationPushConstant.splatForce = glm::vec3(0.0f); // Reset to zero; will be overriden by the inputController if it detects any mouse drags
	}

	// Projection steps: find divergence of velocity, solve the poisson pressure equation and then subtract the gradient of p from the intermediate velocity field
	// Fourth pass: Compute divergence of velocity
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeFluidVelocityDivergence.pipeline->getBindPoint(), pipelines.computeFluidVelocityDivergence.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeFluidVelocityDivergence.pipeline->getBindPoint(), pipelines.computeFluidVelocityDivergence.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeFluidVelocityDivergence.pipeline->getBindPoint(), pipelines.computeFluidVelocityDivergence.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeFluidVelocityDivergence.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);

	copyFluidOutputTextureToInputTexture(fluidVelocityDivergenceInputImageView->getImage());

	// Fifth pass: Compute Jacobi Iteration
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeJacobi.pipeline->getBindPoint(), pipelines.computeJacobi.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeJacobi.pipeline->getBindPoint(), pipelines.computeJacobi.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeJacobi.pipeline->getBindPoint(), pipelines.computeJacobi.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeJacobi.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);

	const uint32_t jacobiIterationCount = 40u;
	for (uint32_t i = 0u; i < jacobiIterationCount; ++i)
	{
		vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);
		copyFluidOutputTextureToInputTexture(fluidPressureInputImageView->getImage());
	}

	// Sixth pass: Compute gradient and subtract it from velocity
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeGradientSubtraction.pipeline->getBindPoint(), pipelines.computeGradientSubtraction.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeGradientSubtraction.pipeline->getBindPoint(), pipelines.computeGradientSubtraction.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &fluidSimulationInputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeGradientSubtraction.pipeline->getBindPoint(), pipelines.computeGradientSubtraction.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &fluidSimulationOutputDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeGradientSubtraction.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant), &fluidSimulationPushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(fluidSimulationGridSize / m_workGroupSize) + 1u, 1u, 1u);

	copyFluidOutputTextureToInputTexture(fluidVelocityInputImageView->getImage());

	// Release
	if (m_graphicsQueue->getFamilyIndex() != m_computeQueue->getFamilyIndex())
	{
		LOGEANDABORT("Gotta verify this logic since we have assumed that computeQueue == graphicsQueue so far");

		// TODO this code has not been tested or verified; gotta check if the src and dst stage/access values are correct
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0u;
		subresourceRange.levelCount = 1u;
		subresourceRange.baseArrayLayer = 0u;
		subresourceRange.layerCount = 1u;

		VkImageMemoryBarrier2 fluidVelocityInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidVelocityInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidVelocityInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidVelocityInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
		fluidVelocityInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		fluidVelocityInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidVelocityInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidVelocityInputTextureImageMemoryBarrier.image = fluidVelocityInputImageView->getImage()->getHandle();
		fluidVelocityInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidVelocityDivergenceInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.image = fluidVelocityDivergenceInputImageView->getImage()->getHandle();
		fluidVelocityDivergenceInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidPressureInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidPressureInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidPressureInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidPressureInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
		fluidPressureInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		fluidPressureInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidPressureInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidPressureInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidPressureInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidPressureInputTextureImageMemoryBarrier.image = fluidPressureInputImageView->getImage()->getHandle();
		fluidPressureInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidDensityInputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidDensityInputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidDensityInputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidDensityInputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
		fluidDensityInputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		fluidDensityInputTextureImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidDensityInputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidDensityInputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidDensityInputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidDensityInputTextureImageMemoryBarrier.image = fluidDensityInputImageView->getImage()->getHandle();
		fluidDensityInputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier2 fluidSimulationOutputTextureImageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		fluidSimulationOutputTextureImageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		fluidSimulationOutputTextureImageMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		fluidSimulationOutputTextureImageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
		fluidSimulationOutputTextureImageMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		fluidSimulationOutputTextureImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		fluidSimulationOutputTextureImageMemoryBarrier.srcQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		fluidSimulationOutputTextureImageMemoryBarrier.dstQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		fluidSimulationOutputTextureImageMemoryBarrier.image = fluidSimulationOutputImageView->getImage()->getHandle();
		fluidSimulationOutputTextureImageMemoryBarrier.subresourceRange = subresourceRange;

		std::array<VkImageMemoryBarrier2, 5> imageMemoryBarriers
		{
			fluidVelocityInputTextureImageMemoryBarrier,
				fluidVelocityDivergenceInputTextureImageMemoryBarrier,
				fluidPressureInputTextureImageMemoryBarrier,
				fluidDensityInputTextureImageMemoryBarrier,
				fluidSimulationOutputTextureImageMemoryBarrier
		};
		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = to_u32(imageMemoryBarriers.size());
		dependencyInfo.pImageMemoryBarriers = imageMemoryBarriers.data();

		vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);
	}
}

void FluidSimulation::createInstance()
{
	m_instance = std::make_unique<Instance>(getName());
}

void FluidSimulation::createSurface()
{
	platform.createSurface(m_instance->getHandle());
	m_surface = platform.getSurface();
}

void FluidSimulation::createDevice()
{
	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.geometryShader = VK_TRUE;
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.shaderInt64 = VK_TRUE;

	std::unique_ptr<PhysicalDevice> physicalDevice = m_instance->getSuitablePhysicalDevice();
	physicalDevice->setRequestedFeatures(deviceFeatures);

	m_workGroupSize = std::min(64u, physicalDevice->getProperties().limits.maxComputeWorkGroupSize[0]);
	m_shadedMemorySize = std::min(1024u, physicalDevice->getProperties().limits.maxComputeSharedMemorySize);

	device = std::make_unique<Device>(std::move(physicalDevice), m_surface, deviceExtensions);
}

void FluidSimulation::createSwapchain()
{
	const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT };
	swapchain = std::make_unique<Swapchain>(*device, m_surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags, m_graphicsQueue->getFamilyIndex(), m_presentQueue->getFamilyIndex());
}

void FluidSimulation::createDescriptorSetLayouts()
{
	// Fluid simulation input descriptor set layout
	VkDescriptorSetLayoutBinding fluidVelocityInputLayoutBinding{};
	fluidVelocityInputLayoutBinding.binding = 0u;
	fluidVelocityInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	fluidVelocityInputLayoutBinding.descriptorCount = 1u;
	fluidVelocityInputLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fluidVelocityInputLayoutBinding.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding fluidVelocityDivergenceInputLayoutBinding{};
	fluidVelocityDivergenceInputLayoutBinding.binding = 1u;
	fluidVelocityDivergenceInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	fluidVelocityDivergenceInputLayoutBinding.descriptorCount = 1u;
	fluidVelocityDivergenceInputLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fluidVelocityDivergenceInputLayoutBinding.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding fluidPressureInputLayoutBinding{};
	fluidPressureInputLayoutBinding.binding = 2u;
	fluidPressureInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	fluidPressureInputLayoutBinding.descriptorCount = 1u;
	fluidPressureInputLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fluidPressureInputLayoutBinding.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding fluidDensityInputLayoutBinding{};
	fluidDensityInputLayoutBinding.binding = 3u;
	fluidDensityInputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	fluidDensityInputLayoutBinding.descriptorCount = 1u;
	fluidDensityInputLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fluidDensityInputLayoutBinding.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> fluidSimulationInputDescriptorSetLayoutBindings{ fluidVelocityInputLayoutBinding, fluidVelocityDivergenceInputLayoutBinding, fluidPressureInputLayoutBinding, fluidDensityInputLayoutBinding };
	fluidSimulationInputDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, fluidSimulationInputDescriptorSetLayoutBindings);

	// Fluid simulation output descriptor set layout
	VkDescriptorSetLayoutBinding fluidSimulationOutputLayoutBinding{};
	fluidSimulationOutputLayoutBinding.binding = 0u;
	fluidSimulationOutputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	fluidSimulationOutputLayoutBinding.descriptorCount = 1u;
	fluidSimulationOutputLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fluidSimulationOutputLayoutBinding.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> fluidSimulationOutputDescriptorSetLayoutBindings{ fluidSimulationOutputLayoutBinding };
	fluidSimulationOutputDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, fluidSimulationOutputDescriptorSetLayoutBindings);
}

void FluidSimulation::createVelocityAdvectionComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/velocityAdvection.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeVelocityAdvection.pipelineState = std::move(computePipelineState);
	pipelines.computeVelocityAdvection.pipeline = std::move(computePipeline);
}

void FluidSimulation::createDensityAdvectionComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/densityAdvection.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeDensityAdvection.pipelineState = std::move(computePipelineState);
	pipelines.computeDensityAdvection.pipeline = std::move(computePipeline);
}

void FluidSimulation::createVelocityGaussianSplatComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/velocityGaussianSplat.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeVelocityGaussianSplat.pipelineState = std::move(computePipelineState);
	pipelines.computeVelocityGaussianSplat.pipeline = std::move(computePipeline);
}

void FluidSimulation::createDensityGaussianSplatComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/densityGaussianSplat.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeDensityGaussianSplat.pipelineState = std::move(computePipelineState);
	pipelines.computeDensityGaussianSplat.pipeline = std::move(computePipeline);
}

void FluidSimulation::createFluidVelocityDivergenceComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/divergence.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeFluidVelocityDivergence.pipelineState = std::move(computePipelineState);
	pipelines.computeFluidVelocityDivergence.pipeline = std::move(computePipeline);
}

void FluidSimulation::createJacobiComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/jacobi.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeJacobi.pipelineState = std::move(computePipelineState);
	pipelines.computeJacobi.pipeline = std::move(computePipeline);
}

void FluidSimulation::createGradientSubtractionComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("fluid_simulation/gradient.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) }
		}
	};
	specializationData.workGroupSize = m_workGroupSize;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ fluidSimulationInputDescriptorSetLayout->getHandle(), fluidSimulationOutputDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FluidSimulationPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeGradientSubtraction.pipelineState = std::move(computePipelineState);
	pipelines.computeGradientSubtraction.pipeline = std::move(computePipeline);
}

void FluidSimulation::createCommandPools()
{
	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.commandPools[i] = std::make_unique<CommandPool>(*device, m_graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		setDebugUtilsObjectName(device->getHandle(), frameData.commandPools[i]->getHandle(), "commandPool for frame #" + std::to_string(i));
	}
}

void FluidSimulation::createCommandBuffers()
{
	if (commandBufferCountForFrame != 3) LOGEANDABORT("commandBufferCountForFrame should be 3");

	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.commandBuffers[i][0] = std::make_unique<CommandBuffer>(*frameData.commandPools[i], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		setDebugUtilsObjectName(device->getHandle(), frameData.commandBuffers[i][0]->getHandle(), "compute/offscreen commandBuffer for frame #" + std::to_string(i));

		frameData.commandBuffers[i][1] = std::make_unique<CommandBuffer>(*frameData.commandPools[i], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		setDebugUtilsObjectName(device->getHandle(), frameData.commandBuffers[i][1]->getHandle(), "post-process commandBuffer for frame #" + std::to_string(i));

		frameData.commandBuffers[i][2] = std::make_unique<CommandBuffer>(*frameData.commandPools[i], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		setDebugUtilsObjectName(device->getHandle(), frameData.commandBuffers[i][2]->getHandle(), "image transfer commandBuffer for frame #" + std::to_string(i));
	}
}

void FluidSimulation::copyBufferToImage(const Buffer &srcBuffer, const Image &dstImage, uint32_t width, uint32_t height)
{
	VkBufferImageCopy region{};
	region.bufferOffset = 0ull;
	region.bufferRowLength = 0u;
	region.bufferImageHeight = 0u;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0u;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { width, height, 1u };

	std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
	vkCmdCopyBufferToImage(commandBuffer->getHandle(), srcBuffer.getHandle(), dstImage.getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &region);
	commandBuffer->end();

	VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	commandBufferSubmitInfo.pNext = nullptr;
	commandBufferSubmitInfo.commandBuffer = commandBuffer->getHandle();
	commandBufferSubmitInfo.deviceMask = 0u;

	VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.pNext = nullptr;
	submitInfo.waitSemaphoreInfoCount = 0u;
	submitInfo.pWaitSemaphoreInfos = nullptr;
	submitInfo.commandBufferInfoCount = 1u;
	submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
	submitInfo.signalSemaphoreInfoCount = 0u;
	submitInfo.pSignalSemaphoreInfos = nullptr;

	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());
}

std::unique_ptr<Image> FluidSimulation::createTextureImageWithInitialValue(uint32_t texWidth, uint32_t texHeight, VkImageUsageFlags imageUsageFlags)
{
	// TODO: we should probably do this initialization on the GPU side with a shader; this is inefficient since we have to mark the texture as a transfer destination, just for a one time initalization
	// Create the staging buffer
	VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
	VkDeviceSize imageSize{ static_cast<VkDeviceSize>(texWidth * texHeight * 16u /* bytes */) }; /* Since our format is VK_FORMAT_R32G32B32A32_SFLOAT, we allocate 4 byte (32 bits) per channel of which there are 4 */
	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = imageSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	// Initialize the texture with the initial values
	const float initalValue = 0.0f;
	void *mappedData = stagingBuffer->map();
	std::vector<float> initialTextureData(imageSize, initalValue);
	memcpy(mappedData, initialTextureData.data(), static_cast<size_t>(imageSize));
	stagingBuffer->unmap();

	// Create the texture image
	VkExtent3D extent{ texWidth, texHeight, 1u };
	std::unique_ptr<Image> textureImage = std::make_unique<Image>(*device, format, extent, imageUsageFlags, VMA_MEMORY_USAGE_GPU_ONLY /* default values for remaining params */);

	// TODO: we can pass in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL as the initial layout when we create these images to avoid this layout transition
	// Transition the texture image to be prepared as a destination target
	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0u;
	subresourceRange.levelCount = 1u;
	subresourceRange.baseArrayLayer = 0u;
	subresourceRange.layerCount = 1u;

	VkImageMemoryBarrier2 transitionTextureImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionTextureImageLayoutBarrier.pNext = nullptr;
	transitionTextureImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // No dependencies
	transitionTextureImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionTextureImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Image will be written to as transfer source
	transitionTextureImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionTextureImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	transitionTextureImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionTextureImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionTextureImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionTextureImageLayoutBarrier.image = textureImage->getHandle();
	transitionTextureImageLayoutBarrier.subresourceRange = subresourceRange;

	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionTextureImageLayoutBarrier;

	std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
	vkCmdPipelineBarrier2KHR(commandBuffer->getHandle(), &dependencyInfo);
	commandBuffer->end();

	VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	commandBufferSubmitInfo.pNext = nullptr;
	commandBufferSubmitInfo.commandBuffer = commandBuffer->getHandle();
	commandBufferSubmitInfo.deviceMask = 0u;

	VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.pNext = nullptr;
	submitInfo.waitSemaphoreInfoCount = 0u;
	submitInfo.pWaitSemaphoreInfos = nullptr;
	submitInfo.commandBufferInfoCount = 1u;
	submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
	submitInfo.signalSemaphoreInfoCount = 0u;
	submitInfo.pSignalSemaphoreInfos = nullptr;

	// TODO: we can refactor this to only required one queuesubmit and no queuewaitidles
	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());

	copyBufferToImage(*stagingBuffer, *textureImage, texWidth, texHeight);

	// Transition the texture image to be prepared to be read and written to by shaders
	transitionTextureImageLayoutBarrier.pNext = nullptr;
	transitionTextureImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait for transfer to finish as the destination
	transitionTextureImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionTextureImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionTextureImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionTextureImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionTextureImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionTextureImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionTextureImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionTextureImageLayoutBarrier.image = textureImage->getHandle();
	transitionTextureImageLayoutBarrier.subresourceRange = subresourceRange;

	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionTextureImageLayoutBarrier;

	commandBuffer->reset();
	commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
	vkCmdPipelineBarrier2KHR(commandBuffer->getHandle(), &dependencyInfo);
	commandBuffer->end();

	// TODO: we can refactor this to only required one queuesubmit and no queuewaitidles
	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());

	return textureImage;
}

void FluidSimulation::copyBufferToBuffer(const Buffer &srcBuffer, const Buffer &dstBuffer, VkDeviceSize size)
{
	VkBufferCopy bufferCopyRegion{};
	bufferCopyRegion.srcOffset = 0ull;
	bufferCopyRegion.dstOffset = 0ull;
	bufferCopyRegion.size = size;

	std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	vkCmdCopyBuffer(commandBuffer->getHandle(), srcBuffer.getHandle(), dstBuffer.getHandle(), 1, &bufferCopyRegion);

	// Execute a transfer to the compute queue, if necessary
	if (m_graphicsQueue->getFamilyIndex() != m_transferQueue->getFamilyIndex())
	{
		LOGEANDABORT("Cases when the graphics and transfer queue are not the same are not supported yet. This logic requires verification as well.");

		VkBufferMemoryBarrier2 bufferMemoryBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
		bufferMemoryBarrier.pNext = nullptr;
		bufferMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		bufferMemoryBarrier.srcAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
		bufferMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		bufferMemoryBarrier.dstAccessMask = 0u;
		bufferMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		bufferMemoryBarrier.dstQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		bufferMemoryBarrier.buffer = srcBuffer.getHandle();
		bufferMemoryBarrier.offset = 0u;
		bufferMemoryBarrier.size = size;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 1u;
		dependencyInfo.pBufferMemoryBarriers = &bufferMemoryBarrier;
		dependencyInfo.imageMemoryBarrierCount = 0u;
		dependencyInfo.pImageMemoryBarriers = nullptr;

		vkCmdPipelineBarrier2KHR(commandBuffer->getHandle(), &dependencyInfo);
	}

	commandBuffer->end();

	VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	commandBufferSubmitInfo.pNext = nullptr;
	commandBufferSubmitInfo.commandBuffer = commandBuffer->getHandle();
	commandBufferSubmitInfo.deviceMask = 0u;

	VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.pNext = nullptr;
	submitInfo.waitSemaphoreInfoCount = 0u;
	submitInfo.pWaitSemaphoreInfos = nullptr;
	submitInfo.commandBufferInfoCount = 1u;
	submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
	submitInfo.signalSemaphoreInfoCount = 0u;
	submitInfo.pSignalSemaphoreInfos = nullptr;

	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());
}

// Initialize the fluid simulation buffers
void FluidSimulation::initializeFluidSimulationResources()
{
	std::unique_ptr<Image> fluidVelocityInputImage = createTextureImageWithInitialValue(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	VkFormat fluidVelocityInputImageFormat = fluidVelocityInputImage->getFormat();
	fluidVelocityInputImageView = std::make_unique<ImageView>(std::move(fluidVelocityInputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, fluidVelocityInputImageFormat);

	std::unique_ptr<Image> fluidVelocityDivergenceInputImage = createTextureImageWithInitialValue(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	VkFormat fluidVelocityDivergenceInputImageFormat = fluidVelocityDivergenceInputImage->getFormat();
	fluidVelocityDivergenceInputImageView = std::make_unique<ImageView>(std::move(fluidVelocityDivergenceInputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, fluidVelocityDivergenceInputImageFormat);

	std::unique_ptr<Image> fluidPressureInputImage = createTextureImageWithInitialValue(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	VkFormat fluidPressureInputImageFormat = fluidPressureInputImage->getFormat();
	fluidPressureInputImageView = std::make_unique<ImageView>(std::move(fluidPressureInputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, fluidPressureInputImageFormat);

	std::unique_ptr<Image> fluidDensityInputImage = createTextureImageWithInitialValue(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	VkFormat fluidDensityInputImageFormat = fluidDensityInputImage->getFormat();
	fluidDensityInputImageView = std::make_unique<ImageView>(std::move(fluidDensityInputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, fluidDensityInputImageFormat);

	// TODO: the VK_IMAGE_USAGE_TRANSFER_DST_BIT flag is required since in createTextureImage, there is code to 0 initialize we should do the initialization in a shader and remove this flag eventually
	std::unique_ptr<Image> fluidSimulationOutputImage = createTextureImageWithInitialValue(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	VkFormat fluidSimulationOutputImageFormat = fluidSimulationOutputImage->getFormat();
	fluidSimulationOutputImageView = std::make_unique<ImageView>(std::move(fluidSimulationOutputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, fluidSimulationOutputImageFormat);

	setDebugUtilsObjectName(device->getHandle(), fluidVelocityInputImageView->getImage()->getHandle(), "fluidVelocityInputImageView image");
	setDebugUtilsObjectName(device->getHandle(), fluidVelocityInputImageView->getHandle(), "fluidVelocityInputImageView imageView");
	setDebugUtilsObjectName(device->getHandle(), fluidVelocityDivergenceInputImageView->getImage()->getHandle(), "fluidVelocityDivergenceInputImageView image");
	setDebugUtilsObjectName(device->getHandle(), fluidVelocityDivergenceInputImageView->getHandle(), "fluidVelocityDivergenceInputImageView imageView");
	setDebugUtilsObjectName(device->getHandle(), fluidPressureInputImageView->getImage()->getHandle(), "fluidPressureInputImageView image");
	setDebugUtilsObjectName(device->getHandle(), fluidPressureInputImageView->getHandle(), "fluidPressureInputImageView imageView");
	setDebugUtilsObjectName(device->getHandle(), fluidDensityInputImageView->getImage()->getHandle(), "fluidDensityInputImageView image");
	setDebugUtilsObjectName(device->getHandle(), fluidDensityInputImageView->getHandle(), "fluidDensityInputImageView imageView");
	setDebugUtilsObjectName(device->getHandle(), fluidSimulationOutputImageView->getImage()->getHandle(), "fluidSimulationOutputImageView image");
	setDebugUtilsObjectName(device->getHandle(), fluidSimulationOutputImageView->getHandle(), "fluidSimulationOutputImageView imageView");

	// Initialize fluidSimulationPushConstant data
	fluidSimulationPushConstant.gridSize = glm::vec2(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height);
}

void FluidSimulation::createDescriptorPool()
{
	std::vector<VkDescriptorPoolSize> poolSizes{};
	poolSizes.resize(4);
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = 10u;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 10u;
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[2].descriptorCount = 10u;
	poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[3].descriptorCount = 10u;


	uint32_t maxSets = 40u; // we are allocating more space than currenly used
	descriptorPool = std::make_unique<DescriptorPool>(*device, poolSizes, maxSets, 0);
}

void FluidSimulation::createTextureSampler()
{
	VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	samplerInfo.magFilter = VK_FILTER_NEAREST;
	samplerInfo.minFilter = VK_FILTER_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = device->getPhysicalDevice().getProperties().limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 0.0f;

	textureSampler = std::make_unique<Sampler>(*device, samplerInfo);
}


void FluidSimulation::createDescriptorSets()
{
	// Fluid Simulation Input Descriptor Set
	VkDescriptorSetAllocateInfo fluidSimulationInputDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	fluidSimulationInputDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	fluidSimulationInputDescriptorSetAllocateInfo.descriptorSetCount = 1;
	fluidSimulationInputDescriptorSetAllocateInfo.pSetLayouts = &fluidSimulationInputDescriptorSetLayout->getHandle();
	fluidSimulationInputDescriptorSet = std::make_unique<DescriptorSet>(*device, fluidSimulationInputDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), fluidSimulationInputDescriptorSet->getHandle(), "fluidSimulationInputDescriptorSet");

	// Binding 0 is the fluid velocity input texture
	VkDescriptorImageInfo fluidVelocityInputTextureInfo{};
	fluidVelocityInputTextureInfo.sampler = textureSampler->getHandle();
	fluidVelocityInputTextureInfo.imageView = fluidVelocityInputImageView->getHandle();
	fluidVelocityInputTextureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	// Binding 1 is the fluid velocity divergence input texture
	VkDescriptorImageInfo fluidVelocityDivergenceInputTextureInfo{};
	fluidVelocityDivergenceInputTextureInfo.sampler = textureSampler->getHandle();
	fluidVelocityDivergenceInputTextureInfo.imageView = fluidVelocityDivergenceInputImageView->getHandle();
	fluidVelocityDivergenceInputTextureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	// Binding 2 is the fluid pressure input texture
	VkDescriptorImageInfo fluidPressureInputTextureInfo{};
	fluidPressureInputTextureInfo.sampler = textureSampler->getHandle();
	fluidPressureInputTextureInfo.imageView = fluidPressureInputImageView->getHandle();
	fluidPressureInputTextureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	// Binding 3 is the fluid density input texture
	VkDescriptorImageInfo fluidDensityInputTextureInfo{};
	fluidDensityInputTextureInfo.sampler = textureSampler->getHandle();
	fluidDensityInputTextureInfo.imageView = fluidDensityInputImageView->getHandle();
	fluidDensityInputTextureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	std::array<VkDescriptorImageInfo, 4> fluidSimulationInputTextureInfos{ fluidVelocityInputTextureInfo, fluidVelocityDivergenceInputTextureInfo, fluidPressureInputTextureInfo, fluidDensityInputTextureInfo };

	VkWriteDescriptorSet writeFluidSimulationInputDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeFluidSimulationInputDescriptorSet.dstSet = fluidSimulationInputDescriptorSet->getHandle();
	writeFluidSimulationInputDescriptorSet.dstBinding = 0;
	writeFluidSimulationInputDescriptorSet.dstArrayElement = 0;
	writeFluidSimulationInputDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writeFluidSimulationInputDescriptorSet.descriptorCount = to_u32(fluidSimulationInputTextureInfos.size());
	writeFluidSimulationInputDescriptorSet.pImageInfo = fluidSimulationInputTextureInfos.data();
	writeFluidSimulationInputDescriptorSet.pBufferInfo = nullptr;
	writeFluidSimulationInputDescriptorSet.pTexelBufferView = nullptr;

	// Fluid Simulation Output Descriptor Set
	VkDescriptorSetAllocateInfo fluidSimulationOutputDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	fluidSimulationOutputDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	fluidSimulationOutputDescriptorSetAllocateInfo.descriptorSetCount = 1u;
	fluidSimulationOutputDescriptorSetAllocateInfo.pSetLayouts = &fluidSimulationOutputDescriptorSetLayout->getHandle();
	fluidSimulationOutputDescriptorSet = std::make_unique<DescriptorSet>(*device, fluidSimulationOutputDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), fluidSimulationOutputDescriptorSet->getHandle(), "fluidSimulationOutputDescriptorSet");

	// Binding 0 is the fluid velocity output texture
	VkDescriptorImageInfo fluidSimulationOutputTextureInfo{};
	fluidSimulationOutputTextureInfo.sampler = VK_NULL_HANDLE;
	fluidSimulationOutputTextureInfo.imageView = fluidSimulationOutputImageView->getHandle();
	fluidSimulationOutputTextureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	std::array<VkDescriptorImageInfo, 1> fluidSimulationOutputTextureInfos{ fluidSimulationOutputTextureInfo };

	VkWriteDescriptorSet writeFluidSimulationOutputDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeFluidSimulationOutputDescriptorSet.dstSet = fluidSimulationOutputDescriptorSet->getHandle();
	writeFluidSimulationOutputDescriptorSet.dstBinding = 0;
	writeFluidSimulationOutputDescriptorSet.dstArrayElement = 0;
	writeFluidSimulationOutputDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writeFluidSimulationOutputDescriptorSet.descriptorCount = to_u32(fluidSimulationOutputTextureInfos.size());
	writeFluidSimulationOutputDescriptorSet.pImageInfo = fluidSimulationOutputTextureInfos.data();
	writeFluidSimulationOutputDescriptorSet.pBufferInfo = nullptr;
	writeFluidSimulationOutputDescriptorSet.pTexelBufferView = nullptr;


	std::array<VkWriteDescriptorSet, 2> writeDescriptorSets
	{
		writeFluidSimulationInputDescriptorSet,
		writeFluidSimulationOutputDescriptorSet
	};
	vkUpdateDescriptorSets(device->getHandle(), to_u32(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void FluidSimulation::createSemaphoreAndFencePools()
{
	semaphorePool = std::make_unique<SemaphorePool>(*device);
	fencePool = std::make_unique<FencePool>(*device);
}

void FluidSimulation::setupSynchronizationObjects()
{
	imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);

	for (size_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.imageAvailableSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.offscreenRenderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.postProcessRenderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.outputImageCopyFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.inFlightFences[i] = fencePool->requestFence();
	}
}

void FluidSimulation::setupCamera()
{
	cameraController = std::make_unique<CameraController>(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height);
	cameraController->getCamera()->setPerspectiveProjection(45.0f, swapchain->getProperties().imageExtent.width / (float)swapchain->getProperties().imageExtent.height, 0.1f, 100.0f);
	cameraController->getCamera()->setView(glm::vec3(-24.5f, 19.1f, 1.9f), glm::vec3(-22.5f, 17.5f, 1.8f), glm::vec3(0.0f, 1.0f, 0.0f));
}
} // namespace vulkr

int main()
{
	vulkr::Platform platform;
	std::unique_ptr<vulkr::FluidSimulation> app = std::make_unique<vulkr::FluidSimulation>(platform, "2D Fluid Simulation");

	platform.initialize(std::move(app));
	platform.prepareApplication();

	platform.runMainProcessingLoop();
	platform.terminate();

	return EXIT_SUCCESS;
}

