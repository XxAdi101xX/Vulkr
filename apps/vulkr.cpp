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

#include "vulkr.h"
#include "platform/platform.h"

#include <thread>

 // TODO move this into raytracing helpers file when it's created
VkTransformMatrixKHR toTransformMatrixKHR(glm::mat4 matrix)
{
	glm::mat4 temp = glm::transpose(matrix);
	VkTransformMatrixKHR outMatrix;
	memcpy(&outMatrix, &temp, sizeof(VkTransformMatrixKHR));
	return outMatrix;
}

namespace vulkr
{

VulkrApp::VulkrApp(Platform &platform, std::string name) : Application{ platform, name } {}

VulkrApp::~VulkrApp()
{
	device->waitIdle();

	semaphorePool.reset();
	fencePool.reset();

	cleanupSwapchain();

	textureSampler.reset();

	globalDescriptorSetLayout.reset();
	objectDescriptorSetLayout.reset();
	textureDescriptorSetLayout.reset();
	postProcessingDescriptorSetLayout.reset();
	taaDescriptorSetLayout.reset();
	particleComputeDescriptorSetLayout.reset();
	gltfMaterialSamplersDescriptorSetLayout.reset();
	gltfNodeDescriptorSetLayout.reset();
	gltfMaterialDescriptorSetLayout.reset();

	for (auto &it : objModelsToRender)
	{
		it.indexBuffer.reset();
		it.vertexBuffer.reset();
		it.materialsBuffer.reset();
		it.materialsIndexBuffer.reset();
	}
	objModelsToRender.clear();
	objInstances.clear();

	for (auto &it : gltfModelsToRender)
	{
		it.gltfModel.destroy(device.get());
		it.materialsBuffer.reset();
	}
	gltfModelsToRender.clear();

	for (auto &it : textureImageViews)
	{
		it.reset();
	}

	m_blas.clear();
	m_tlas.reset();
	m_instBuffer.reset();
	m_rtSBTBuffer.reset();

	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.commandPools[i].reset();
	}

	imguiPool.reset();

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	device.reset();

	if (m_surface != VK_NULL_HANDLE)
	{
		vkDestroySurfaceKHR(m_instance->getHandle(), m_surface, nullptr);
	}

	m_instance.reset();
}

void VulkrApp::cleanupSwapchain()
{
	// Command buffers
	for (uint32_t i = 0u; i < maxFramesInFlight; ++i)
	{
		for (uint32_t j = 0u; j < commandBufferCountForFrame; ++j)
		{
			frameData.commandBuffers[i][j].reset();
		}
	}

	// Depth image
	depthImageView.reset();

	// Framebuffers
	for (uint32_t i = 0u; i < maxFramesInFlight; ++i)
	{
		frameData.offscreenFramebuffers[i].reset();
		frameData.postProcessFramebuffers[i].reset();
	}

	// Pipelines
	pipelines.pbr.pipeline.reset();
	pipelines.pbr.pipelineState.reset();
	pipelines.offscreen.pipeline.reset();
	pipelines.offscreen.pipelineState.reset();
	pipelines.postProcess.pipeline.reset();
	pipelines.postProcess.pipelineState.reset();
	pipelines.computeModelAnimation.pipeline.reset();
	pipelines.computeModelAnimation.pipelineState.reset();
	pipelines.computeParticleCalculate.pipeline.reset();
	pipelines.computeParticleCalculate.pipelineState.reset();
	pipelines.computeParticleIntegrate.pipeline.reset();
	pipelines.computeParticleIntegrate.pipelineState.reset();
	pipelines.rayTracing.pipeline.reset();
	pipelines.rayTracing.pipelineState.reset();

	// Renderpasses
	mainRenderPass.renderPass.reset();
	mainRenderPass.subpasses.clear();
	mainRenderPass.inputAttachments.clear();
	mainRenderPass.colorAttachments.clear();
	mainRenderPass.resolveAttachments.clear();
	mainRenderPass.depthStencilAttachments.clear();
	mainRenderPass.preserveAttachments.clear();

	postRenderPass.renderPass.reset();
	postRenderPass.subpasses.clear();
	postRenderPass.inputAttachments.clear();
	postRenderPass.colorAttachments.clear();
	postRenderPass.resolveAttachments.clear();
	postRenderPass.depthStencilAttachments.clear();
	postRenderPass.preserveAttachments.clear();

	// Swapchain
	swapchain.reset();

	// Buffers
	lightBuffer.reset();
	cameraBuffer.reset();
	previousFrameCameraBuffer.reset();
	objectBuffer.reset();
	previousFrameObjectBuffer.reset();
	particleBuffer.reset();

	// Descriptor Sets
	globalDescriptorSet.reset();
	objectDescriptorSet.reset();
	postProcessingDescriptorSet.reset();
	taaDescriptorSet.reset();
	particleComputeDescriptorSet.reset();
	// textureDescriptorSet.reset(); // Don't need to reset textureDescriptorSet this one up on recreate since the texture data is the same across different swapchain configurations
	raytracingDescriptorSet.reset();
	gltfMaterialDescriptorSet.reset();

	// Textures
	outputImageView.reset();
	copyOutputImageView.reset();
	historyImageView.reset();
	velocityImageView.reset();

	for (uint8_t i = 0; i < std::thread::hardware_concurrency(); ++i)
	{
		initCommandPools[i].reset();
	}

	descriptorPool.reset();

#ifndef RENDERDOC_DEBUG
	m_rtDescSetLayout.reset();
	m_rtDescPool.reset();
#endif

	cameraController.reset();
}

void VulkrApp::prepare()
{
	Application::prepare();

	setupTimer();
	initializeHaltonSequenceArray();

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
	createDepthResources();
	createMainRenderPass();
	createPostRenderPass();
	createCommandPools();
	createCommandBuffers();
	createImageResourcesForFrames();
	createFramebuffers();

	initializeImGui();
	loadModels();

	createDescriptorSetLayouts();
	createMainRasterizationPipeline();
	//createModelAnimationComputePipeline();
	createPostProcessingPipeline();
	createPbrRasterizationPipeline();
	createTextureSampler();
	createUniformBuffers();
	createSSBOs();
	prepareParticleData();
	createParticleCalculateComputePipeline();
	createParticleIntegrateComputePipeline();
	createDescriptorPool();
	createDescriptorSets();
	setupCamera();
	createSceneInstances();
	createSceneLights();

#ifndef RENDERDOC_DEBUG
	buildBlas();
	buildTlas(false);
	createRaytracingDescriptorPool();
	createRaytracingDescriptorLayout();
	createRaytracingDescriptorSets();
	createRaytracingPipeline();
	createRaytracingShaderBindingTable();
#endif

	createSemaphoreAndFencePools();
	setupSynchronizationObjects();

	initializeBufferData();

	// Ensure that the first tick is with respect to the time after all of the steup
	drawingTimer->tick();
}

void VulkrApp::update()
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
		// TODO: enabling this check causes errors on swapchain recreation, will need to eventually resolve this for correctness
		fencePool->wait(&imagesInFlight[swapchainImageIndex]);
	}
	imagesInFlight[swapchainImageIndex] = frameData.inFlightFences[currentFrame];

	// Maintain the deltaTime for both frames in flight
	if (currentFrame == 0)
	{
		computeParticlesPushConstant.deltaTime = static_cast<float>(drawingTimer->tick());
	}
	drawImGuiInterface();
	animateInstances();
	dataUpdatePerFrame();

#ifndef RENDERDOC_DEBUG
	if (raytracingEnabled)
	{
		buildTlas(true);
	}
#endif

	// TODO check if this jitter value is correct
	if (temporalAntiAliasingEnabled)
	{
		taaPushConstant.jitter = haltonSequence[std::min(taaPushConstant.frameSinceViewChange, static_cast<int>(taaDepth - 1))];
		taaPushConstant.jitter.x = (taaPushConstant.jitter.x / swapchain->getProperties().imageExtent.width) * 5.0f;
		taaPushConstant.jitter.y = (taaPushConstant.jitter.y / swapchain->getProperties().imageExtent.height) * 5.0f;
	}
	else
	{
		taaPushConstant.jitter = glm::vec2(0.0f, 0.0f);
	}

	// Begin command buffer for offscreen pass
	frameData.commandBuffers[currentFrame][0]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	// Compute shader invocations
	computeParticles();

	// Compute vertices with compute shader; need to uncomment createModelAnimationComputePipeline if using this TODO: fix the validation errors that happen when this is enabled, might be due to incorrect sytnax with obj buffer in animate.comp or the fact that the objBuffer is readonly? Not totally sure.
	//animateWithCompute(); 

	if (raytracingEnabled)
	{
		// Add memory barrier to ensure that the particleIntegrate computer shader has finished writing to the currentFrameObjectBuffer
		VkMemoryBarrier2 memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
		memoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
		memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 1u;
		dependencyInfo.pMemoryBarriers = &memoryBarrier;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = 0u;
		dependencyInfo.pImageMemoryBarriers = nullptr;

		vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);

		raytrace();
	}
	else
	{
		// Add memory barrier to ensure that the particleIntegrate computer shader has finished writing to the currentFrameObjectBuffer
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

		vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);

		frameData.commandBuffers[currentFrame][0]->beginRenderPass(*mainRenderPass.renderPass, *(frameData.offscreenFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, offscreenFramebufferClearValues, VK_SUBPASS_CONTENTS_INLINE);
		rasterizeObj();
		rasterizeGltf();
		frameData.commandBuffers[currentFrame][0]->endRenderPass();
	}

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

	// Begin command buffer for post process pass
	frameData.commandBuffers[currentFrame][1]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	// Post offscreen renderpass
	frameData.commandBuffers[currentFrame][1]->beginRenderPass(*postRenderPass.renderPass, *(frameData.postProcessFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, postProcessFramebufferClearValues, VK_SUBPASS_CONTENTS_INLINE);
	if (temporalAntiAliasingEnabled)
	{
		postProcess();
	}
	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frameData.commandBuffers[currentFrame][1]->getHandle());
	frameData.commandBuffers[currentFrame][1]->endRenderPass();

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

	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionSwapchainLayoutBarrier;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][2]->getHandle(), &dependencyInfo);

	// Prepare the historyImage as a transfer destination
	VkImageMemoryBarrier2 transitionHistoryImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionHistoryImageLayoutBarrier.pNext = nullptr;
	transitionHistoryImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // We have a semaphore that synchronizes the post processing pass with all of the things happening on frameData.commandBuffers[currentFrame][2]->getHandle() hence no other required
	transitionHistoryImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionHistoryImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	transitionHistoryImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionHistoryImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionHistoryImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionHistoryImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionHistoryImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionHistoryImageLayoutBarrier.image = historyImageView->getImage()->getHandle();
	transitionHistoryImageLayoutBarrier.subresourceRange = subresourceRange;

	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 1u;
	dependencyInfo.pImageMemoryBarriers = &transitionHistoryImageLayoutBarrier;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][2]->getHandle(), &dependencyInfo);
	// Note that the layout of the outputImage has been transitioned from VK_IMAGE_LAYOUT_GENERAL to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL as defined in the postProcessingRenderPass configuration

	VkImageCopy outputImageCopyRegion{};
	outputImageCopyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };
	outputImageCopyRegion.srcOffset = { 0, 0, 0 };
	outputImageCopyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };
	outputImageCopyRegion.dstOffset = { 0, 0, 0 };
	outputImageCopyRegion.extent = { swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1u };
	// Copy output image to swapchain image and history image (note that they can execute in any order due to lack of barriers)
	vkCmdCopyImage(frameData.commandBuffers[currentFrame][2]->getHandle(), outputImageView->getImage()->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain->getImages()[swapchainImageIndex]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &outputImageCopyRegion);
	vkCmdCopyImage(frameData.commandBuffers[currentFrame][2]->getHandle(), outputImageView->getImage()->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, historyImageView->getImage()->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &outputImageCopyRegion);

	VkBufferCopy cameraBufferCopyRegion{};
	cameraBufferCopyRegion.srcOffset = 0ull;
	cameraBufferCopyRegion.dstOffset = 0ull;
	cameraBufferCopyRegion.size = sizeof(CameraData);
	vkCmdCopyBuffer(frameData.commandBuffers[currentFrame][2]->getHandle(), cameraBuffer->getHandle(), previousFrameCameraBuffer->getHandle(), 1, &cameraBufferCopyRegion);

	VkBufferCopy objectBufferCopyRegion{};
	objectBufferCopyRegion.srcOffset = 0ull;
	objectBufferCopyRegion.dstOffset = 0ull;
	objectBufferCopyRegion.size = sizeof(ObjInstance) * maxInstanceCount;
	vkCmdCopyBuffer(frameData.commandBuffers[currentFrame][2]->getHandle(), objectBuffer->getHandle(), previousFrameObjectBuffer->getHandle(), 1, &objectBufferCopyRegion);

	// Transition the history image and output image back to the general layout
	transitionHistoryImageLayoutBarrier.pNext = nullptr;
	transitionHistoryImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait on transfer to finish
	transitionHistoryImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionHistoryImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionHistoryImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionHistoryImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionHistoryImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionHistoryImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionHistoryImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionHistoryImageLayoutBarrier.image = historyImageView->getImage()->getHandle();
	transitionHistoryImageLayoutBarrier.subresourceRange = subresourceRange;

	VkImageMemoryBarrier2 transitionOutputImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionOutputImageLayoutBarrier.pNext = nullptr;
	transitionOutputImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait on transfer to finish
	transitionOutputImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	transitionOutputImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionOutputImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionOutputImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	transitionOutputImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionOutputImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionOutputImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionOutputImageLayoutBarrier.image = outputImageView->getImage()->getHandle();
	transitionOutputImageLayoutBarrier.subresourceRange = subresourceRange;

	std::array<VkImageMemoryBarrier2, 2> transitionImageBarriers{ transitionHistoryImageLayoutBarrier, transitionOutputImageLayoutBarrier };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = to_u32(transitionImageBarriers.size());
	dependencyInfo.pImageMemoryBarriers = transitionImageBarriers.data();

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

// TODO: this totally does not work and needs an entire overhaul, also need to handle recreate for raytracing
void VulkrApp::recreateSwapchain()
{
	LOGEANDABORT("Swapchain recreation is not supported");
	// TODO: update window width and high variables on window resize callback??
	// TODO: enable the imagesInFlight check in the update() function and resolve the swapchain recreation bug
	device->waitIdle();
	cleanupSwapchain();

	createSwapchain();
	setupCamera();
	createMainRenderPass();
	createPostRenderPass();
	createMainRasterizationPipeline();
	createPostProcessingPipeline();
	createPbrRasterizationPipeline();
	//createModelAnimationComputePipeline();
	createDepthResources();
	createFramebuffers();
	createCommandBuffers();
	createUniformBuffers();
	createSSBOs();
	createDescriptorPool();
	createDescriptorSets();
#ifndef RENDERDOC_DEBUG
	updateRaytracingDescriptorSet();
#endif
	createSceneInstances();

	imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);
}

void VulkrApp::handleInputEvents(const InputEvent &inputEvent)
{
	ImGuiIO &io = ImGui::GetIO();
	if (io.WantCaptureMouse) return;

	cameraController->handleInputEvents(inputEvent);
}

/* Private methods start here */

void VulkrApp::drawImGuiInterface()
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool changed{ false };
	glm::vec3 position = cameraController->getCamera()->getPosition();
	glm::vec3 center = cameraController->getCamera()->getCenter();
	float fovy = cameraController->getCamera()->getFovY();

	// Creating UI
	// TODO stop using default debug window and use begin/end to control the window; we would need to create a separate class altogether
	// ImGui::Begin();
	if (ImGui::BeginTabBar("Main Tab"))
	{
		if (ImGui::BeginTabItem("Scene"))
		{
#ifndef RENDERDOC_DEBUG
			changed |= ImGui::Checkbox("Raytracing enabled", &raytracingEnabled);
#endif
			changed |= ImGui::Checkbox("Temporal anti-aliasing enabled (Rasterization only)", &temporalAntiAliasingEnabled);
			if (changed)
			{
				resetFrameSinceViewChange();
				changed = false;
			}

			ImGui::Text("Position");
			ImGui::SameLine();
			ImGui::InputFloat3("##Position", &position.x);
			changed |= ImGui::IsItemDeactivatedAfterEdit();
			if (changed)
			{
				cameraController->getCamera()->setPosition(position);
				changed = false;
			}
			ImGui::Text("Center");
			ImGui::SameLine();
			ImGui::InputFloat3("##Center", &center.x);
			changed |= ImGui::IsItemDeactivatedAfterEdit();
			if (changed)
			{
				cameraController->getCamera()->setCenter(center);
				changed = false;
			}
			ImGui::Text("FOV");
			ImGui::SameLine();
			changed |= ImGui::DragFloat("##FOV", &fovy, 1.0f, 1.0f, 179.0f, "%.3f", 0);
			if (changed)
			{
				cameraController->getCamera()->setFovY(fovy);
				changed = false;
			}

			ImGui::EndTabItem();
		}

		if (ImGui::BeginTabItem("Environment"))
		{
			for (int lightIndex = 0; lightIndex < sceneLights.size(); ++lightIndex)
			{
				std::ostringstream oss;
				oss << "Light " << lightIndex;
				if (ImGui::CollapsingHeader(oss.str().c_str()))
				{
					oss.str(""); oss << "Point" << "##" << lightIndex;
					changed |= ImGui::RadioButton(oss.str().c_str(), &sceneLights[lightIndex].type, 0);
					ImGui::SameLine();
					oss.str(""); oss << "Directional" << "##" << lightIndex;
					changed |= ImGui::RadioButton(oss.str().c_str(), &sceneLights[lightIndex].type, 1);

					oss.str(""); oss << "Position" << "##" << lightIndex;
					changed |= ImGui::SliderFloat3(oss.str().c_str(), &(sceneLights[lightIndex].position.x), -50.f, 50.f);
					oss.str(""); oss << "Intensity" << "##" << lightIndex;
					changed |= ImGui::SliderFloat(oss.str().c_str(), &sceneLights[lightIndex].intensity, 0.f, 250.f);
					oss.str(""); oss << "Rotation" << "##" << lightIndex;
					changed |= ImGui::SliderFloat2(oss.str().c_str(), &(sceneLights[lightIndex].rotation.x), 0.0f, 360.0f);
				}
			}

			if (changed)
			{
				haveLightsUpdated = true;
				resetFrameSinceViewChange();
				changed = false;
			}

			ImGui::EndTabItem();
		}

		ImGui::EndTabBar();
	}
	// ImGui::End();

	// ImGui::ShowDemoWindow();
}

void VulkrApp::animateInstances()
{
	const uint64_t startingIndex{ 2 };
	const int64_t wusonInstanceCount{
		std::count_if(objInstances.begin(), objInstances.end(), [this](ObjInstance i) { return i.objIndex == getObjModelIndex(defaultModelFilePath + "wuson.obj"); })
	};
	if (wusonInstanceCount == 0)
	{
		LOGW("No wuson instances found to animate");
		return;
	}

	const float deltaAngle = 6.28318530718f / static_cast<float>(wusonInstanceCount);
	const float wusonLength = 3.f;
	const float radius = wusonLength / (2.f * sin(deltaAngle / 2.0f));
	const float time = std::chrono::duration<float, std::chrono::seconds::period>(drawingTimer->elapsed()).count();
	const float offset = time * 0.5f;

	for (uint64_t i = startingIndex; i < startingIndex + wusonInstanceCount; i++)
	{
		objInstances[i].transform = glm::rotate(glm::mat4(1.0f), i * deltaAngle + offset, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::translate(glm::mat4{ 1.0 }, glm::vec3(radius, 0.f, 0.f));
		objInstances[i].transformIT = glm::transpose(glm::inverse(objInstances[i].transform));
	}

	// Update the transformation for the wuson instances
	void *mappedData = objectBuffer->map();
	ObjInstance *objectSSBO = static_cast<ObjInstance *>(mappedData);
	for (int i = startingIndex; i < startingIndex + wusonInstanceCount; i++)
	{
		objectSSBO[i].transform = objInstances[i].transform;
		objectSSBO[i].transformIT = objInstances[i].transformIT;
	}
	objectBuffer->unmap();
}

void VulkrApp::animateWithCompute()
{
	// TODO: we might require a buffer memory barrier similar to the code in the other compute workflows
	const uint64_t wusonModelIndex{ getObjModelIndex(defaultModelFilePath + "wuson.obj") };

	computePushConstant.indexCount = objModelsToRender[wusonModelIndex].indicesCount;
	computePushConstant.time = std::chrono::duration<float, std::chrono::seconds::period>(drawingTimer->elapsed()).count();

	PipelineData &pipelineData = pipelines.computeModelAnimation;
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0, 1, &objectDescriptorSet->getHandle(), 0, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstant), &computePushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), objModelsToRender[wusonModelIndex].indicesCount / m_workGroupSize, 1u, 1u);
}

void VulkrApp::computeParticles()
{
	// Acquire
	if (m_graphicsQueue->getFamilyIndex() != m_computeQueue->getFamilyIndex())
	{
		LOGEANDABORT("Gotta verify this logic since we have assumed that computeQueue == graphicsQueue so far");
		VkBufferMemoryBarrier2 bufferMemoryBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
		bufferMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		bufferMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		bufferMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
		bufferMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		bufferMemoryBarrier.srcQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		bufferMemoryBarrier.dstQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		bufferMemoryBarrier.buffer = particleBuffer->getHandle();
		bufferMemoryBarrier.offset = 0ull;
		bufferMemoryBarrier.size = particleBufferSize;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 1u;
		dependencyInfo.pBufferMemoryBarriers = &bufferMemoryBarrier;
		dependencyInfo.imageMemoryBarrierCount = 0u;
		dependencyInfo.pImageMemoryBarriers = nullptr;

		vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);
	}

	// First pass: Calculate particle movement
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleCalculate.pipeline->getBindPoint(), pipelines.computeParticleCalculate.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleCalculate.pipeline->getBindPoint(), pipelines.computeParticleCalculate.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &particleComputeDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleCalculate.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(ComputeParticlesPushConstant), &computeParticlesPushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(computeParticlesPushConstant.particleCount / m_workGroupSize) + 1u, 1u, 1u);

	// Add memory barrier to ensure that the computer shader has finished writing to the buffer
	VkMemoryBarrier2 memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
	memoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
	memoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
	memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 1u;
	dependencyInfo.pMemoryBarriers = &memoryBarrier;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 0u;
	dependencyInfo.pImageMemoryBarriers = nullptr;

	vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);

	// Second pass: Integrate particles
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipeline->getBindPoint(), pipelines.computeParticleIntegrate.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipeline->getBindPoint(), pipelines.computeParticleIntegrate.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &particleComputeDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipeline->getBindPoint(), pipelines.computeParticleIntegrate.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &objectDescriptorSet->getHandle(), 0u, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(ComputeParticlesPushConstant), &computeParticlesPushConstant);
	vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(computeParticlesPushConstant.particleCount / m_workGroupSize) + 1u, 1u, 1u); // round up invocation

	// Release
	if (m_graphicsQueue->getFamilyIndex() != m_computeQueue->getFamilyIndex())
	{
		LOGEANDABORT("Gotta verify this logic since we have assumed that computeQueue == graphicsQueue so far");
		VkBufferMemoryBarrier2 bufferMemoryBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
		bufferMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
		bufferMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		bufferMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		bufferMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT; // TODO: is this correct
		bufferMemoryBarrier.srcQueueFamilyIndex = m_computeQueue->getFamilyIndex();
		bufferMemoryBarrier.dstQueueFamilyIndex = m_graphicsQueue->getFamilyIndex();
		bufferMemoryBarrier.buffer = particleBuffer->getHandle();
		bufferMemoryBarrier.offset = 0ull;
		bufferMemoryBarrier.size = particleBufferSize;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 1u;
		dependencyInfo.pBufferMemoryBarriers = &bufferMemoryBarrier;
		dependencyInfo.imageMemoryBarrierCount = 0u;
		dependencyInfo.pImageMemoryBarriers = nullptr;

		vkCmdPipelineBarrier2KHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &dependencyInfo);
	}
}

void VulkrApp::renderNode(vulkr::gltf::Node *node)
{
	if (node->mesh)
	{
		// Render mesh primitives
		for (vulkr::gltf::Primitive *primitive : node->mesh->primitives)
		{
			std::string pipelineName = "pbr";
			// TODO cleanup these comments
			//std::string pipelineVariant = "";

			//if (primitive->material.unlit) {
			//    // KHR_materials_unlit
			//    pipelineName = "unlit";
			//};

			//const VkPipeline pipeline = pipelines[pipelineName + pipelineVariant];
			PipelineData &pipelineData = pipelines.pbr;

			// This was moved out of the function
			//if (pipeline != boundPipeline) {
			//    vkCmdBindPipeline(commandBuffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
			//    boundPipeline = pipeline;
			//}

			vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineData.pipeline->getHandle());

			const std::vector<VkDescriptorSet> descriptorsets =
			{
				//descriptorSets[cbIndex].scene,
				globalDescriptorSet->getHandle(),
				primitive->material.descriptorSet,
				node->mesh->uniformBuffer.descriptorSet,
				gltfMaterialDescriptorSet->getHandle()
			};
			vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0, static_cast<uint32_t>(descriptorsets.size()), descriptorsets.data(), 0, NULL);

			// Pass material index for this primitive using a push constant, the shader uses this to index into the material buffer
			gltfPushConstant.cameraPos = cameraController->getCamera()->getPosition();
			gltfPushConstant.materialIndex = primitive->material.index;
			gltfPushConstant.lightCount = static_cast<int>(sceneLights.size());;

			vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GltfPushConstant), &gltfPushConstant);

			if (primitive->hasIndices)
			{
				vkCmdDrawIndexed(frameData.commandBuffers[currentFrame][0]->getHandle(), primitive->indexCount, 1, primitive->firstIndex, 0, 0);
			}
			else
			{
				vkCmdDraw(frameData.commandBuffers[currentFrame][0]->getHandle(), primitive->vertexCount, 1, 0, 0);
			}
		}
	}

	for (auto child : node->children)
	{
		renderNode(child);
	}
}

void VulkrApp::rasterizeGltf()
{
	for (auto &gltfModelToRender : gltfModelsToRender)
	{
		vulkr::gltf::Model &model = gltfModelToRender.gltfModel;
		VkBuffer vertexBuffers[] = { model.vertexBuffer->getHandle() };
		VkDeviceSize offsets[] = { 0ull };

		vkCmdBindVertexBuffers(frameData.commandBuffers[currentFrame][0]->getHandle(), 0, 1, vertexBuffers, offsets);
		if (model.indexBuffer->getHandle() != VK_NULL_HANDLE)
		{
			vkCmdBindIndexBuffer(frameData.commandBuffers[currentFrame][0]->getHandle(), model.indexBuffer->getHandle(), 0, VK_INDEX_TYPE_UINT32);
		}

		// TODO check if we also want to render by visibility
		//boundPipeline = VK_NULL_HANDLE;

		//// Opaque primitives first
		//for (auto node : model.nodes) {
		//    renderNode(node, i, vkglTF::Material::ALPHAMODE_OPAQUE);
		//}
		//// Alpha masked primitives
		//for (auto node : model.nodes) {
		//    renderNode(node, i, vkglTF::Material::ALPHAMODE_MASK);
		//}
		//// Transparent primitives
		//// TODO: Correct depth sorting
		//for (auto node : model.nodes) {
		//    renderNode(node, i, vkglTF::Material::ALPHAMODE_BLEND);
		//}

		for (auto node : model.nodes)
		{
			renderNode(node);
		}
	}

}

void VulkrApp::initializeBufferData()
{
	for (uint32_t frame = 0u; frame < maxFramesInFlight; ++frame)
	{
		// Update the camera buffer
		CameraData cameraData{};
		cameraData.view = cameraController->getCamera()->getView();
		cameraData.proj = cameraController->getCamera()->getProjection();

		void *mappedData = cameraBuffer->map();
		memcpy(mappedData, &cameraData, sizeof(cameraData));
		cameraBuffer->unmap();

		// Update the light buffer
		mappedData = lightBuffer->map();
		memcpy(mappedData, sceneLights.data(), sizeof(LightData) * sceneLights.size());
		lightBuffer->unmap();

		// Update the object buffer
		mappedData = objectBuffer->map();
		ObjInstance *objectSSBO = static_cast<ObjInstance *>(mappedData);
		for (int instanceIndex = 0; instanceIndex < objInstances.size(); instanceIndex++)
		{
			objectSSBO[instanceIndex].transform = objInstances[instanceIndex].transform;
			objectSSBO[instanceIndex].transformIT = objInstances[instanceIndex].transformIT;
			objectSSBO[instanceIndex].objIndex = objInstances[instanceIndex].objIndex;
			objectSSBO[instanceIndex].textureOffset = objInstances[instanceIndex].textureOffset;
			objectSSBO[instanceIndex].vertices = objInstances[instanceIndex].vertices;
			objectSSBO[instanceIndex].indices = objInstances[instanceIndex].indices;
			objectSSBO[instanceIndex].materials = objInstances[instanceIndex].materials;
			objectSSBO[instanceIndex].materialIndices = objInstances[instanceIndex].materialIndices;
		}
		objectBuffer->unmap();
	}
}

void VulkrApp::dataUpdatePerFrame()
{
	bool hasCameraUpdated = cameraController->getCamera()->isUpdated();
	cameraController->getCamera()->resetUpdatedFlag();

	if (hasCameraUpdated)
	{
		// Update the camera buffer
		CameraData cameraData{};
		cameraData.view = cameraController->getCamera()->getView();
		cameraData.proj = cameraController->getCamera()->getProjection();

		void *mappedData = cameraBuffer->map();
		memcpy(mappedData, &cameraData, sizeof(cameraData));
		cameraBuffer->unmap();
	}

	if (haveLightsUpdated)
	{
		// Update the light buffer
		void *mappedData = lightBuffer->map();
		memcpy(mappedData, sceneLights.data(), sizeof(LightData) * sceneLights.size());
		lightBuffer->unmap();

		haveLightsUpdated = false;
	}

	// TAA Check
	if (!temporalAntiAliasingEnabled && !raytracingEnabled)
	{
		raytracingPushConstant.frameSinceViewChange = 0; // TODO remove
		taaPushConstant.frameSinceViewChange = 0;
		taaPushConstant.jitter = glm::vec2(0.0f);
	}
	else
	{
		// If the camera has updated, we don't want to use the previous frame for anti aliasing
		if (hasCameraUpdated)
		{
			resetFrameSinceViewChange();
		}
		raytracingPushConstant.frameSinceViewChange += 1;// TODO remove
		taaPushConstant.frameSinceViewChange += 1;

	}
}

// TODO: as opposed to doing slot based binding of descriptor sets which leads to multiple vkCmdBindDescriptorSets calls per drawcall, you can use
// frequency based descriptor sets and use dynamicOffsetCount: see https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/, or just bindless decriptors altogether
void VulkrApp::rasterizeObj()
{
	debugUtilBeginLabel(frameData.commandBuffers[currentFrame][0]->getHandle(), "Rasterize");

	PipelineData &pipelineData = pipelines.offscreen;

	// Bind the pipeline
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipeline->getHandle());

	// Global data descriptor
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0u, 1u, &globalDescriptorSet->getHandle(), 0u, nullptr);

	// Object data descriptor
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 1u, 1u, &objectDescriptorSet->getHandle(), 0u, nullptr);

	// Texture descriptor
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 2u, 1u, &textureDescriptorSet->getHandle(), 0u, nullptr);

	// Taa data descriptor
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 3u, 1u, &taaDescriptorSet->getHandle(), 0u, nullptr);

	// Push constants
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(TaaPushConstant), &taaPushConstant);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(TaaPushConstant), sizeof(RasterizationPushConstant), &rasterizationPushConstant);

	// Bind vertices, indices and call the draw method
	uint64_t lastObjIndex{ 0ull };
	for (uint32_t index = 0u; index < objInstances.size(); index++)
	{
		ObjModelRenderingData &objModelRenderingData = objModelsToRender[objInstances[index].objIndex];
		// Bind the vertex and index buffers if the instance model is different from the previous one (we always bind for the first one)
		if (index == 0 || objInstances[index].objIndex != lastObjIndex)
		{
			VkBuffer vertexBuffers[] = { objModelRenderingData.vertexBuffer->getHandle() };
			VkDeviceSize offsets[] = { 0ull };
			vkCmdBindVertexBuffers(frameData.commandBuffers[currentFrame][0]->getHandle(), 0u, 1u, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(frameData.commandBuffers[currentFrame][0]->getHandle(), objModelRenderingData.indexBuffer->getHandle(), 0ull, VK_INDEX_TYPE_UINT32);

			lastObjIndex = objInstances[index].objIndex;
		}

		vkCmdDrawIndexed(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(objModelRenderingData.indicesCount), 1u, 0u, 0, index);
	}

	debugUtilEndLabel(frameData.commandBuffers[currentFrame][0]->getHandle());
}

void VulkrApp::postProcess()
{
	debugUtilBeginLabel(frameData.commandBuffers[currentFrame][1]->getHandle(), "Post Process");

	PipelineData &pipelineData = pipelines.postProcess;

	// Bind the pipeline
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][1]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipeline->getHandle());

	// Post processing descriptor
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][1]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0, 1, &postProcessingDescriptorSet->getHandle(), 0, nullptr);

	// Limit blending once enough frames have been occumulated (this optimization is commented out since it is only relavent for static scenes)
	//float blendFactor = 0.0f;
	//if (taaPushConstant.frameSinceViewChange < taaDepth)
	//{
	//    blendFactor = 1.0f / float(taaPushConstant.frameSinceViewChange + 1);
	//}

	// Blend constants (history buffer is set to 1 - blendFactor)
	float blendFactor = 0.2f;
	float blendConstants[4] = { blendFactor, blendFactor, blendFactor, blendFactor };
	vkCmdSetBlendConstants(frameData.commandBuffers[currentFrame][1]->getHandle(), blendConstants);

	// Push constants
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][1]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostProcessPushConstant), &postProcessPushConstant);

	// Bind vertices, indices and call the draw method
	uint64_t lastObjIndex{ 0ull };
	for (uint32_t index = 0u; index < objInstances.size(); index++)
	{
		ObjModelRenderingData &objModelRenderingData = objModelsToRender[objInstances[index].objIndex];
		// Bind the objModel if it's a different one from last one (we always bind the first time)
		if (index == 0 || objInstances[index].objIndex != lastObjIndex)
		{
			VkBuffer vertexBuffers[] = { objModelRenderingData.vertexBuffer->getHandle() };
			VkDeviceSize offsets[] = { 0ull };
			vkCmdBindVertexBuffers(frameData.commandBuffers[currentFrame][1]->getHandle(), 0u, 1u, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(frameData.commandBuffers[currentFrame][1]->getHandle(), objModelRenderingData.indexBuffer->getHandle(), 0u, VK_INDEX_TYPE_UINT32);

			lastObjIndex = objInstances[index].objIndex;
		}

		vkCmdDrawIndexed(frameData.commandBuffers[currentFrame][1]->getHandle(), to_u32(objModelRenderingData.indicesCount), 1u, 0u, 0, index);
	}

	debugUtilEndLabel(frameData.commandBuffers[currentFrame][1]->getHandle());
}

void VulkrApp::setupTimer()
{
	drawingTimer = std::make_unique<Timer>();
	drawingTimer->start();
}

float createHaltonSequence(uint32_t index, uint32_t base)
{
	float f = 1.0f;
	float r = 0.0f;
	uint32_t current = index;
	do
	{
		f = f / base;
		r = r + f * (current % base);
		current = static_cast<uint32_t>(glm::floor(current / base));
	} while (current > 0);
	return r;
}

void VulkrApp::initializeHaltonSequenceArray()
{
	for (int i = 0; i < taaDepth; ++i)
	{
		haltonSequence[i] = glm::vec2(createHaltonSequence(i + 1, 2), createHaltonSequence(i + 1, 3));
	}
}

void VulkrApp::createInstance()
{
	m_instance = std::make_unique<Instance>(getName());
	g_instanceHandle = m_instance->getHandle();
}

void VulkrApp::createSurface()
{
	platform.createSurface(m_instance->getHandle());
	m_surface = platform.getSurface();
}

void VulkrApp::createDevice()
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

void VulkrApp::createSwapchain()
{
	const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT };
	swapchain = std::make_unique<Swapchain>(*device, m_surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags, m_graphicsQueue->getFamilyIndex(), m_presentQueue->getFamilyIndex());

	postProcessPushConstant.imageExtent = glm::vec2(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height);
}

void VulkrApp::setupCamera()
{
	cameraController = std::make_unique<CameraController>(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height);
	cameraController->getCamera()->setPerspectiveProjection(45.0f, swapchain->getProperties().imageExtent.width / (float)swapchain->getProperties().imageExtent.height, 0.1f, 100.0f);
	cameraController->getCamera()->setView(glm::vec3(-24.5f, 19.1f, 1.9f), glm::vec3(-22.5f, 17.5f, 1.8f), glm::vec3(0.0f, 1.0f, 0.0f));
}

void VulkrApp::createMainRenderPass()
{
	std::vector<Attachment> attachments;
	Attachment outputImageAttachment{}; // outputImage
	outputImageAttachment.format = swapchain->getProperties().surfaceFormat.format;
	outputImageAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	outputImageAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	outputImageAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	outputImageAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	outputImageAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	outputImageAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	// We could set the finalLayout to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL since the postProcessing step requires that but then we would have to add a
	// layout transitions after a raytracing step to change the image from VK_IMAGE_LAYOUT_GENERAL to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL. For simplicity,
	// I've made it so that the finalLayout regardless of rasterization/raytracing is VK_IMAGE_LAYOUT_GENERAL
	outputImageAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
	attachments.push_back(outputImageAttachment);

	VkAttachmentReference2 outputImageAttachmentRef{ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 };
	outputImageAttachmentRef.pNext = nullptr;
	outputImageAttachmentRef.attachment = 0u;
	outputImageAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	outputImageAttachmentRef.aspectMask = 0u;
	mainRenderPass.colorAttachments.push_back(outputImageAttachmentRef);

	Attachment copyOutputImageAttachment{}; // copyOutputImage
	copyOutputImageAttachment.format = swapchain->getProperties().surfaceFormat.format;
	copyOutputImageAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	copyOutputImageAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	copyOutputImageAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	copyOutputImageAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	copyOutputImageAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	copyOutputImageAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	copyOutputImageAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
	attachments.push_back(copyOutputImageAttachment);

	VkAttachmentReference2 copyOutputImageAttachmentRef{ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 };
	copyOutputImageAttachmentRef.pNext = nullptr;
	copyOutputImageAttachmentRef.attachment = 1u;
	copyOutputImageAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	copyOutputImageAttachmentRef.aspectMask = 0u;
	mainRenderPass.colorAttachments.push_back(copyOutputImageAttachmentRef);

	Attachment velocityImageAttachment{}; // velocityImage
	velocityImageAttachment.format = swapchain->getProperties().surfaceFormat.format;
	velocityImageAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	velocityImageAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	velocityImageAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	velocityImageAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	velocityImageAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	velocityImageAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	velocityImageAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
	attachments.push_back(velocityImageAttachment);

	VkAttachmentReference2 velocityImageAttachmentRef{ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 };
	velocityImageAttachmentRef.pNext = nullptr;
	velocityImageAttachmentRef.attachment = 2u;
	velocityImageAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	velocityImageAttachmentRef.aspectMask = 0u;
	mainRenderPass.colorAttachments.push_back(velocityImageAttachmentRef);

	Attachment depthAttachment{};
	depthAttachment.format = getSupportedDepthFormat(device->getPhysicalDevice().getHandle());
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments.push_back(depthAttachment);

	VkAttachmentReference2 depthAttachmentRef{ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 };
	depthAttachmentRef.pNext = nullptr;
	depthAttachmentRef.attachment = 3u;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	depthAttachmentRef.aspectMask = 0u;
	mainRenderPass.depthStencilAttachments.push_back(depthAttachmentRef);

	mainRenderPass.subpasses.emplace_back(
		mainRenderPass.inputAttachments,
		mainRenderPass.colorAttachments,
		mainRenderPass.resolveAttachments,
		mainRenderPass.depthStencilAttachments,
		mainRenderPass.preserveAttachments,
		VK_PIPELINE_BIND_POINT_GRAPHICS
	);

	std::vector<VkSubpassDependency2> dependencies;
	dependencies.resize(2);

	// TODO: verify these subpass dependencies are correct
	// Only need a dependency coming in to ensure that the first layout transition happens at the right time.
	// Second external dependency is implied by having a different finalLayout and subpass layout.
	VkMemoryBarrier2 memoryBarrier1 = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
		.srcAccessMask = 0u, // We don't have anything that we need to flush
		.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
		.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
	};

	dependencies[0].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
	dependencies[0].pNext = &memoryBarrier1;
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0u; // References the subpass index in the subpasses array
	// srcStageMask, dstStageMask, srcAccessMask and dstAccessMask on subpassDependency2 are ignored since we're passing in a memory barrier
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkMemoryBarrier2 memoryBarrier2 = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
		.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT
	};

	dependencies[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
	dependencies[1].pNext = &memoryBarrier2;
	dependencies[1].srcSubpass = 0u;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	// srcStageMask, dstStageMask, srcAccessMask and dstAccessMask on subpassDependency2 are ignored since we're passing in a memory barrier
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	// Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
	// but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
	// which injects an external dependency here with dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. 

	mainRenderPass.renderPass = std::make_unique<RenderPass>(*device, attachments, mainRenderPass.subpasses, dependencies);
	setDebugUtilsObjectName(device->getHandle(), mainRenderPass.renderPass->getHandle(), "mainRenderPass");
}

void VulkrApp::createPostRenderPass()
{
	std::vector<Attachment> attachments;
	Attachment colorAttachment{}; // outputImage
	colorAttachment.format = swapchain->getProperties().surfaceFormat.format;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	attachments.push_back(colorAttachment);

	VkAttachmentReference2 colorAttachmentRef{ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 };
	colorAttachmentRef.pNext = nullptr;
	colorAttachmentRef.attachment = 0u;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorAttachmentRef.aspectMask = 0u;
	postRenderPass.colorAttachments.push_back(colorAttachmentRef);

	Attachment depthAttachment{};
	depthAttachment.format = getSupportedDepthFormat(device->getPhysicalDevice().getHandle());
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments.push_back(depthAttachment);

	VkAttachmentReference2 depthAttachmentRef{ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 };
	depthAttachmentRef.pNext = nullptr;
	depthAttachmentRef.attachment = 1u;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	postRenderPass.depthStencilAttachments.push_back(depthAttachmentRef);

	postRenderPass.subpasses.emplace_back(
		postRenderPass.inputAttachments,
		postRenderPass.colorAttachments,
		postRenderPass.resolveAttachments,
		postRenderPass.depthStencilAttachments,
		postRenderPass.preserveAttachments,
		VK_PIPELINE_BIND_POINT_GRAPHICS
	);

	std::vector<VkSubpassDependency2> dependencies;
	dependencies.resize(2);

	// TODO: verify these subpass dependencies are correct
	// Only need a dependency coming in to ensure that the first layout transition happens at the right time.
	// Second external dependency is implied by having a different finalLayout and subpass layout.
	VkMemoryBarrier2 memoryBarrier1 = {
	.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
	.pNext = nullptr,
	.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
	.srcAccessMask = 0, // we don't have anything to flush
	.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
	.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT // The clear on depth counts as a write operation I believe so we need appropriate access masks
	};

	dependencies[0].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
	dependencies[0].pNext = &memoryBarrier1;
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0u; // References the subpass index in the subpasses array
	// srcStageMask, dstStageMask, srcAccessMask and dstAccessMask on subpassDependency2 are ignored since we're passing in a memory barrier
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkMemoryBarrier2 memoryBarrier2 = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
		.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT
	};

	dependencies[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
	dependencies[1].pNext = &memoryBarrier2;
	dependencies[1].srcSubpass = 0u;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	// srcStageMask, dstStageMask, srcAccessMask and dstAccessMask on subpassDependency2 are ignored since we're passing in a memory barrier
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	// Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
	// but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
	// which injects an external dependency here with dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. 

	postRenderPass.renderPass = std::make_unique<RenderPass>(*device, attachments, postRenderPass.subpasses, dependencies);
	setDebugUtilsObjectName(device->getHandle(), postRenderPass.renderPass->getHandle(), "postProcessRenderPass");
}


void VulkrApp::createDescriptorSetLayouts()
{
	// Global descriptor set layout
	VkDescriptorSetLayoutBinding cameraBufferLayoutBinding{};
	cameraBufferLayoutBinding.binding = 0u;
	cameraBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	cameraBufferLayoutBinding.descriptorCount = 1u;
	cameraBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; // TODO: not used in fragment bit, closest hit shader
	cameraBufferLayoutBinding.pImmutableSamplers = nullptr; // Optional
	VkDescriptorSetLayoutBinding lightBufferLayoutBinding{};
	lightBufferLayoutBinding.binding = 1u;
	lightBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	lightBufferLayoutBinding.descriptorCount = 1u;
	lightBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; // TODO not being used in vertex bit
	lightBufferLayoutBinding.pImmutableSamplers = nullptr; // Optional

	std::vector<VkDescriptorSetLayoutBinding> globalDescriptorSetLayoutBindings{ cameraBufferLayoutBinding, lightBufferLayoutBinding };
	globalDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, globalDescriptorSetLayoutBindings);

	// Object descriptor set layout
	VkDescriptorSetLayoutBinding objectBufferLayoutBinding{};
	objectBufferLayoutBinding.binding = 0u;
	objectBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	objectBufferLayoutBinding.descriptorCount = 1u;
	objectBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
	objectBufferLayoutBinding.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> objectDescriptorSetLayoutBindings{ objectBufferLayoutBinding };
	objectDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, objectDescriptorSetLayoutBindings);

	// Post processing descriptor set layout
	VkDescriptorSetLayoutBinding currentFrameCameraBufferLayoutBindingForPost{};
	currentFrameCameraBufferLayoutBindingForPost.binding = 0u;
	currentFrameCameraBufferLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	currentFrameCameraBufferLayoutBindingForPost.descriptorCount = 1u;
	currentFrameCameraBufferLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	currentFrameCameraBufferLayoutBindingForPost.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding currentFrameObjectBufferLayoutBindingForPost{};
	currentFrameObjectBufferLayoutBindingForPost.binding = 1u;
	currentFrameObjectBufferLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	currentFrameObjectBufferLayoutBindingForPost.descriptorCount = 1u;
	currentFrameObjectBufferLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	currentFrameObjectBufferLayoutBindingForPost.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding historyImageLayoutBindingForPost{};
	historyImageLayoutBindingForPost.binding = 2u;
	historyImageLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	historyImageLayoutBindingForPost.descriptorCount = 1u;
	historyImageLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	historyImageLayoutBindingForPost.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding velocityImageLayoutBindingForPost{};
	velocityImageLayoutBindingForPost.binding = 3u;
	velocityImageLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	velocityImageLayoutBindingForPost.descriptorCount = 1u;
	velocityImageLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	velocityImageLayoutBindingForPost.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding copyOutputImageLayoutBindingForPost{};
	copyOutputImageLayoutBindingForPost.binding = 4u;
	copyOutputImageLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	copyOutputImageLayoutBindingForPost.descriptorCount = 1u;
	copyOutputImageLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	copyOutputImageLayoutBindingForPost.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> postProcessingDescriptorSetLayoutBindings{
		currentFrameCameraBufferLayoutBindingForPost,
		currentFrameObjectBufferLayoutBindingForPost,
		historyImageLayoutBindingForPost,
		velocityImageLayoutBindingForPost,
		copyOutputImageLayoutBindingForPost
	};
	postProcessingDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, postProcessingDescriptorSetLayoutBindings);

	// Taa descriptor set layout
	VkDescriptorSetLayoutBinding previousFrameCameraBufferLayoutBindingForTaa{};
	previousFrameCameraBufferLayoutBindingForTaa.binding = 0u;
	previousFrameCameraBufferLayoutBindingForTaa.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	previousFrameCameraBufferLayoutBindingForTaa.descriptorCount = 1u;
	previousFrameCameraBufferLayoutBindingForTaa.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	previousFrameCameraBufferLayoutBindingForTaa.pImmutableSamplers = nullptr;
	VkDescriptorSetLayoutBinding previousFrameObjectBufferLayoutBindingForTaa{};
	previousFrameObjectBufferLayoutBindingForTaa.binding = 1u;
	previousFrameObjectBufferLayoutBindingForTaa.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	previousFrameObjectBufferLayoutBindingForTaa.descriptorCount = 1u;
	previousFrameObjectBufferLayoutBindingForTaa.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	previousFrameObjectBufferLayoutBindingForTaa.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> taaDescriptorSetLayoutBindings{
		previousFrameCameraBufferLayoutBindingForTaa,
		previousFrameObjectBufferLayoutBindingForTaa
	};
	taaDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, taaDescriptorSetLayoutBindings);

	// Texture descriptor set layout
	VkDescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 0u;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.descriptorCount = to_u32(textureImageViews.size());
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	samplerLayoutBinding.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> textureDescriptorSetLayoutBindings{ samplerLayoutBinding };
	textureDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, textureDescriptorSetLayoutBindings);

	// Particle compute descriptor set layout
	VkDescriptorSetLayoutBinding particleBufferLayoutBinding{};
	particleBufferLayoutBinding.binding = 0u;
	particleBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	particleBufferLayoutBinding.descriptorCount = 1u;
	particleBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	particleBufferLayoutBinding.pImmutableSamplers = nullptr;

	std::vector<VkDescriptorSetLayoutBinding> particleComputeDescriptorSetLayoutBindings{ particleBufferLayoutBinding };
	particleComputeDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, particleComputeDescriptorSetLayoutBindings);

	// glTF material sampler descriptor set layout
	std::vector<VkDescriptorSetLayoutBinding> gltfMaterialSamplersLayoutBinding =
	{
		{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
	};
	gltfMaterialSamplersDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, gltfMaterialSamplersLayoutBinding);

	// glTF node descriptor set layout
	std::vector<VkDescriptorSetLayoutBinding> gltfNodeLayoutBinding =
	{
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },
	};
	gltfNodeDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, gltfNodeLayoutBinding);

	// glTF material descriptor set layout
	std::vector<VkDescriptorSetLayoutBinding> gltfMaterialSetLayoutBindings = {
	{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
	};
	gltfMaterialDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, gltfMaterialSetLayoutBindings);
}

void VulkrApp::createMainRasterizationPipeline()
{
	VertexInputState vertexInputState{};
	vertexInputState.bindingDescriptions.reserve(1);
	vertexInputState.attributeDescriptions.reserve(3);

	VkVertexInputBindingDescription bindingDescription{};
	bindingDescription.binding = 0;
	bindingDescription.stride = sizeof(VertexObj);
	bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	vertexInputState.bindingDescriptions.emplace_back(bindingDescription);

	// Position at location 0
	VkVertexInputAttributeDescription positionAttributeDescription;
	positionAttributeDescription.location = 0;
	positionAttributeDescription.binding = 0;
	positionAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttributeDescription.offset = offsetof(VertexObj, position);
	// Normal at location 1
	VkVertexInputAttributeDescription normalAttributeDescription;
	normalAttributeDescription.location = 1;
	normalAttributeDescription.binding = 0;
	normalAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttributeDescription.offset = offsetof(VertexObj, normal);
	// Color at location 2
	VkVertexInputAttributeDescription colorAttributeDescription;
	colorAttributeDescription.location = 2;
	colorAttributeDescription.binding = 0;
	colorAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttributeDescription.offset = offsetof(VertexObj, color);
	// TexCoord at location 3
	VkVertexInputAttributeDescription textureCoordinateAttributeDescription;
	textureCoordinateAttributeDescription.location = 3;
	textureCoordinateAttributeDescription.binding = 0;
	textureCoordinateAttributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
	textureCoordinateAttributeDescription.offset = offsetof(VertexObj, textureCoordinate);

	vertexInputState.attributeDescriptions.emplace_back(positionAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(normalAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(colorAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(textureCoordinateAttributeDescription);

	InputAssemblyState inputAssemblyState{};
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyState.primitiveRestartEnable = VK_FALSE;

	ViewportState viewportState{};
	VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(swapchain->getProperties().imageExtent.width), static_cast<float>(swapchain->getProperties().imageExtent.height), 0.0f, 1.0f };
	viewportState.viewports.emplace_back(viewport);
	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = swapchain->getProperties().imageExtent;
	viewportState.scissors.emplace_back(scissor);

	RasterizationState rasterizationState{};
	rasterizationState.depthClampEnable = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.lineWidth = 1.0f;
	rasterizationState.depthBiasEnable = VK_FALSE;

	MultisampleState multisampleState{};
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampleState.sampleShadingEnable = VK_FALSE;

	DepthStencilState depthStencilState{};
	depthStencilState.depthTestEnable = VK_TRUE;
	depthStencilState.depthWriteEnable = VK_TRUE;
	// TODO: change this to VK_COMPARE_OP_GREATER: https://developer.nvidia.com/content/depth-precision-visualized , will need to also change the depthStencil clearValue to 0.0f instead of 1.0f
	depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.stencilTestEnable = VK_FALSE;

	ColorBlendAttachmentState colorBlendAttachmentState{};
	colorBlendAttachmentState.blendEnable = VK_FALSE;

	ColorBlendState colorBlendState{};
	colorBlendState.logicOpEnable = VK_FALSE;
	colorBlendState.logicOp = VK_LOGIC_OP_COPY;
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState); // No blending for output image
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState); // No blending for copy output image
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState); // No blending for velocity image
	colorBlendState.blendConstants[0] = 0.0f;
	colorBlendState.blendConstants[1] = 0.0f;
	colorBlendState.blendConstants[2] = 0.0f;
	colorBlendState.blendConstants[3] = 0.0f;

	std::vector<VkDynamicState> dynamicStates;

	std::shared_ptr<ShaderSource> mainVertexShader = std::make_shared<ShaderSource>("rasterization/main.vert.spv");
	VkSpecializationInfo mainVertexShaderSpecializationInfo;
	mainVertexShaderSpecializationInfo.mapEntryCount = 0;
	mainVertexShaderSpecializationInfo.dataSize = 0;

	std::shared_ptr<ShaderSource> mainFragmentShader = std::make_shared<ShaderSource>("rasterization/main.frag.spv");
	struct SpecializationData
	{
		uint32_t maxLightCount;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, maxLightCount), sizeof(uint32_t) }
		}
	};
	specializationData.maxLightCount = maxLightCount;

	VkSpecializationInfo mainFragmentShaderSpecializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, mainVertexShaderSpecializationInfo, mainVertexShader);
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, mainFragmentShaderSpecializationInfo, mainFragmentShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{
		globalDescriptorSetLayout->getHandle(),
			objectDescriptorSetLayout->getHandle(),
			textureDescriptorSetLayout->getHandle(),
			taaDescriptorSetLayout->getHandle()
	};

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange taaPushConstantRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(TaaPushConstant) };
	pushConstantRangeHandles.push_back(taaPushConstantRange);
	VkPushConstantRange rasterizationPushConstantRange{ VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(TaaPushConstant), sizeof(RasterizationPushConstant) };
	pushConstantRangeHandles.push_back(rasterizationPushConstantRange);

	std::unique_ptr<GraphicsPipelineState> mainRasterizationPipelineState = std::make_unique<GraphicsPipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles),
		*mainRenderPass.renderPass,
		vertexInputState,
		inputAssemblyState,
		viewportState,
		rasterizationState,
		multisampleState,
		depthStencilState,
		colorBlendState,
		dynamicStates
	);
	std::unique_ptr<GraphicsPipeline> mainRasterizationPipeline = std::make_unique<GraphicsPipeline>(*device, *mainRasterizationPipelineState, nullptr);

	pipelines.offscreen.pipelineState = std::move(mainRasterizationPipelineState);
	pipelines.offscreen.pipeline = std::move(mainRasterizationPipeline);
}

void VulkrApp::createPbrRasterizationPipeline()
{
	VertexInputState vertexInputState{};
	vertexInputState.bindingDescriptions.reserve(1);
	vertexInputState.attributeDescriptions.reserve(3);

	VkVertexInputBindingDescription bindingDescription{};
	bindingDescription.binding = 0;
	bindingDescription.stride = sizeof(vulkr::gltf::Model::Vertex);
	bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	vertexInputState.bindingDescriptions.emplace_back(bindingDescription);

	// Position at location 0
	VkVertexInputAttributeDescription positionAttributeDescription;
	positionAttributeDescription.location = 0;
	positionAttributeDescription.binding = 0;
	positionAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, pos);
	// Normal at location 1
	VkVertexInputAttributeDescription normalAttributeDescription;
	normalAttributeDescription.location = 1;
	normalAttributeDescription.binding = 0;
	normalAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, normal);
	// uv0 at location 2
	VkVertexInputAttributeDescription uv0AttributeDescription;
	uv0AttributeDescription.location = 2;
	uv0AttributeDescription.binding = 0;
	uv0AttributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
	uv0AttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, uv0);
	// uv1 at location 3
	VkVertexInputAttributeDescription uv1AttributeDescription;
	uv1AttributeDescription.location = 3;
	uv1AttributeDescription.binding = 0;
	uv1AttributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
	uv1AttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, uv1);
	// joint0 at location 4
	VkVertexInputAttributeDescription joint0AttributeDescription;
	joint0AttributeDescription.location = 4;
	joint0AttributeDescription.binding = 0;
	joint0AttributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	joint0AttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, joint0);
	// weight0 at location 5
	VkVertexInputAttributeDescription weight0AttributeDescription;
	weight0AttributeDescription.location = 5;
	weight0AttributeDescription.binding = 0;
	weight0AttributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	weight0AttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, weight0);
	// color at location 6
	VkVertexInputAttributeDescription colorAttributeDescription;
	colorAttributeDescription.location = 6;
	colorAttributeDescription.binding = 0;
	colorAttributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	colorAttributeDescription.offset = offsetof(vulkr::gltf::Model::Vertex, color);

	vertexInputState.attributeDescriptions.emplace_back(positionAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(normalAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(uv0AttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(uv1AttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(joint0AttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(weight0AttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(colorAttributeDescription);

	InputAssemblyState inputAssemblyState{};
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyState.primitiveRestartEnable = VK_FALSE;

	ViewportState viewportState{};
	VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(swapchain->getProperties().imageExtent.width), static_cast<float>(swapchain->getProperties().imageExtent.height), 0.0f, 1.0f };
	viewportState.viewports.emplace_back(viewport);
	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = swapchain->getProperties().imageExtent;
	viewportState.scissors.emplace_back(scissor);

	RasterizationState rasterizationState{};
	rasterizationState.depthClampEnable = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.lineWidth = 1.0f;
	rasterizationState.depthBiasEnable = VK_FALSE;

	MultisampleState multisampleState{};
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampleState.sampleShadingEnable = VK_FALSE;

	DepthStencilState depthStencilState{};
	depthStencilState.depthTestEnable = VK_TRUE;
	depthStencilState.depthWriteEnable = VK_TRUE;
	// TODO: change this to VK_COMPARE_OP_GREATER: https://developer.nvidia.com/content/depth-precision-visualized , will need to also change the depthStencil clearValue to 0.0f instead of 1.0f
	depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.stencilTestEnable = VK_FALSE;

	ColorBlendAttachmentState colorBlendAttachmentState{};
	colorBlendAttachmentState.blendEnable = VK_FALSE;

	ColorBlendState colorBlendState{};
	colorBlendState.logicOpEnable = VK_FALSE;
	colorBlendState.logicOp = VK_LOGIC_OP_COPY;
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState); // No blending for output image
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState); // No blending for copy output image
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState); // No blending for velocity image
	colorBlendState.blendConstants[0] = 0.0f;
	colorBlendState.blendConstants[1] = 0.0f;
	colorBlendState.blendConstants[2] = 0.0f;
	colorBlendState.blendConstants[3] = 0.0f;

	std::vector<VkDynamicState> dynamicStates;

	std::shared_ptr<ShaderSource> vertexShader = std::make_shared<ShaderSource>("rasterization/pbr.vert.spv");
	VkSpecializationInfo vertexShaderSpecializationInfo;
	vertexShaderSpecializationInfo.mapEntryCount = 0;
	vertexShaderSpecializationInfo.dataSize = 0;

	std::shared_ptr<ShaderSource> fragmentShader = std::make_shared<ShaderSource>("rasterization/material_pbr.frag.spv");
	struct SpecializationData
	{
		uint32_t maxLightCount;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, offsetof(SpecializationData, maxLightCount), sizeof(uint32_t) }
		}
	};
	specializationData.maxLightCount = maxLightCount;

	VkSpecializationInfo fragmentShaderSpecializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, vertexShaderSpecializationInfo, vertexShader);
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShaderSpecializationInfo, fragmentShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{
		globalDescriptorSetLayout->getHandle(),
		gltfMaterialSamplersDescriptorSetLayout->getHandle(),
		gltfNodeDescriptorSetLayout->getHandle(),
		gltfMaterialDescriptorSetLayout->getHandle()
	};

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange pushConstantRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GltfPushConstant) };
	pushConstantRangeHandles.push_back(pushConstantRange);

	std::unique_ptr<GraphicsPipelineState> pbrRasterizationPipelineState = std::make_unique<GraphicsPipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles),
		*mainRenderPass.renderPass,
		vertexInputState,
		inputAssemblyState,
		viewportState,
		rasterizationState,
		multisampleState,
		depthStencilState,
		colorBlendState,
		dynamicStates
	);
	std::unique_ptr<GraphicsPipeline> pbrRasterizationPipeline = std::make_unique<GraphicsPipeline>(*device, *pbrRasterizationPipelineState, nullptr);

	pipelines.pbr.pipelineState = std::move(pbrRasterizationPipelineState);
	pipelines.pbr.pipeline = std::move(pbrRasterizationPipeline);
}

void VulkrApp::createPostProcessingPipeline()
{
	VertexInputState vertexInputState{};
	vertexInputState.bindingDescriptions.reserve(1);
	vertexInputState.attributeDescriptions.reserve(3);

	VkVertexInputBindingDescription bindingDescription{};
	bindingDescription.binding = 0;
	bindingDescription.stride = sizeof(VertexObj);
	bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	vertexInputState.bindingDescriptions.emplace_back(bindingDescription);

	// Position at location 0
	VkVertexInputAttributeDescription positionAttributeDescription;
	positionAttributeDescription.location = 0;
	positionAttributeDescription.binding = 0;
	positionAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttributeDescription.offset = offsetof(VertexObj, position);
	// Normal at location 1
	VkVertexInputAttributeDescription normalAttributeDescription;
	normalAttributeDescription.location = 1;
	normalAttributeDescription.binding = 0;
	normalAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttributeDescription.offset = offsetof(VertexObj, normal);
	// Color at location 2
	VkVertexInputAttributeDescription colorAttributeDescription;
	colorAttributeDescription.location = 2;
	colorAttributeDescription.binding = 0;
	colorAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttributeDescription.offset = offsetof(VertexObj, color);
	// TexCoord at location 3
	VkVertexInputAttributeDescription textureCoordinateAttributeDescription;
	textureCoordinateAttributeDescription.location = 3;
	textureCoordinateAttributeDescription.binding = 0;
	textureCoordinateAttributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
	textureCoordinateAttributeDescription.offset = offsetof(VertexObj, textureCoordinate);

	vertexInputState.attributeDescriptions.emplace_back(positionAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(normalAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(colorAttributeDescription);
	vertexInputState.attributeDescriptions.emplace_back(textureCoordinateAttributeDescription);

	InputAssemblyState inputAssemblyState{};
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyState.primitiveRestartEnable = VK_FALSE;

	ViewportState viewportState{};
	VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(swapchain->getProperties().imageExtent.width), static_cast<float>(swapchain->getProperties().imageExtent.height), 0.0f, 1.0f };
	viewportState.viewports.emplace_back(viewport);
	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = swapchain->getProperties().imageExtent;
	viewportState.scissors.emplace_back(scissor);

	RasterizationState rasterizationState{};
	rasterizationState.depthClampEnable = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.lineWidth = 1.0f;
	rasterizationState.depthBiasEnable = VK_FALSE;

	MultisampleState multisampleState{};
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampleState.sampleShadingEnable = VK_FALSE;

	DepthStencilState depthStencilState{};
	depthStencilState.depthTestEnable = VK_TRUE;
	depthStencilState.depthWriteEnable = VK_TRUE;
	// TODO: change this to VK_COMPARE_OP_GREATER: https://developer.nvidia.com/content/depth-precision-visualized , will need to also change the depthStencil clearValue to 0.0f instead of 1.0f
	depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.stencilTestEnable = VK_FALSE;

	ColorBlendAttachmentState colorBlendAttachmentState{};
	colorBlendAttachmentState.blendEnable = VK_TRUE;
	colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; // The value of the history buffer is returned
	colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR; // The outputImage is the non-accumulated image
	colorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Ignore the alpha value in the history buffer
	colorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Always take the alpha value of the most recent frame
	colorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	ColorBlendState colorBlendState{};
	colorBlendState.logicOpEnable = VK_FALSE;
	colorBlendState.logicOp = VK_LOGIC_OP_COPY;
	colorBlendState.attachments.emplace_back(colorBlendAttachmentState);
	// Blend constants must be set dynamically since the dynamic state is defined

	std::vector<VkDynamicState> dynamicStates{ VK_DYNAMIC_STATE_BLEND_CONSTANTS };

	std::shared_ptr<ShaderSource> postProcessVertexShader = std::make_shared<ShaderSource>("post_processing/postProcess.vert.spv");
	VkSpecializationInfo postProcessVertexShaderSpecializationInfo;
	postProcessVertexShaderSpecializationInfo.mapEntryCount = 0;
	postProcessVertexShaderSpecializationInfo.dataSize = 0;
	std::shared_ptr<ShaderSource> postProcessFragmentShader = std::make_shared<ShaderSource>("post_processing/postProcess.frag.spv");
	VkSpecializationInfo postProcessFragmentShaderSpecializationInfo;
	postProcessFragmentShaderSpecializationInfo.mapEntryCount = 0;
	postProcessFragmentShaderSpecializationInfo.dataSize = 0;

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, postProcessVertexShaderSpecializationInfo, postProcessVertexShader);
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, postProcessFragmentShaderSpecializationInfo, postProcessFragmentShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ postProcessingDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange postProcessPushConstantRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostProcessPushConstant) };
	pushConstantRangeHandles.push_back(postProcessPushConstantRange);

	std::unique_ptr<GraphicsPipelineState> postProcessingPipelineState = std::make_unique<GraphicsPipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles),
		*postRenderPass.renderPass,
		vertexInputState,
		inputAssemblyState,
		viewportState,
		rasterizationState,
		multisampleState,
		depthStencilState,
		colorBlendState,
		dynamicStates
	);
	std::unique_ptr<GraphicsPipeline> postProcessingPipeline = std::make_unique<GraphicsPipeline>(*device, *postProcessingPipelineState, nullptr);

	pipelines.postProcess.pipelineState = std::move(postProcessingPipelineState);
	pipelines.postProcess.pipeline = std::move(postProcessingPipeline);
}

void VulkrApp::createModelAnimationComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("animate.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, to_u32(0 * sizeof(uint32_t)),  sizeof(uint32_t) }
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

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ objectDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeModelAnimation.pipelineState = std::move(computePipelineState);
	pipelines.computeModelAnimation.pipeline = std::move(computePipeline);
}

void VulkrApp::createParticleCalculateComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("particle_system/particleCalculate.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
		uint32_t sharedDataSize;
		float gravity;
		float power;
		float soften;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 5> entries{
		{
			{ 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) },
			{ 1u, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t) },
			{ 2u, offsetof(SpecializationData, gravity), sizeof(float) },
			{ 3u, offsetof(SpecializationData, power), sizeof(float) },
			{ 4u, offsetof(SpecializationData, soften), sizeof(float) },
		}
	};
	specializationData.workGroupSize = m_workGroupSize;
	specializationData.sharedDataSize = m_workGroupSize / sizeof(glm::vec4);
	specializationData.gravity = 0.002f;
	specializationData.power = 0.75f;
	specializationData.soften = 0.05f;

	VkSpecializationInfo specializationInfo =
	{
		to_u32(entries.size()),
		entries.data(),
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> shaderModules;
	shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, computeShader);

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ particleComputeDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeParticlesPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeParticleCalculate.pipelineState = std::move(computePipelineState);
	pipelines.computeParticleCalculate.pipeline = std::move(computePipeline);
}

void VulkrApp::createParticleIntegrateComputePipeline()
{
	std::shared_ptr<ShaderSource> computeShader = std::make_shared<ShaderSource>("particle_system/particleIntegrate.comp.spv");

	struct SpecializationData
	{
		uint32_t workGroupSize;
	} specializationData;
	const std::array<VkSpecializationMapEntry, 1> entries{
		{
			{ 0u, to_u32(0 * sizeof(uint32_t)), sizeof(uint32_t) }
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

	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ particleComputeDescriptorSetLayout->getHandle(), objectDescriptorSetLayout->getHandle() };

	std::vector<VkPushConstantRange> pushConstantRangeHandles;
	VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeParticlesPushConstant) };
	pushConstantRangeHandles.push_back(computePushConstantRange);

	std::unique_ptr<ComputePipelineState> computePipelineState = std::make_unique<ComputePipelineState>(
		std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
	);
	std::unique_ptr<ComputePipeline> computePipeline = std::make_unique<ComputePipeline>(*device, *computePipelineState, nullptr);

	pipelines.computeParticleIntegrate.pipelineState = std::move(computePipelineState);
	pipelines.computeParticleIntegrate.pipeline = std::move(computePipeline);
}

void VulkrApp::createFramebuffers()
{
	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		std::vector<VkImageView> offscreenAttachments{ outputImageView->getHandle(), copyOutputImageView->getHandle(), velocityImageView->getHandle(), depthImageView->getHandle() };
		frameData.offscreenFramebuffers[i] = std::make_unique<Framebuffer>(*device, *swapchain, *mainRenderPass.renderPass, offscreenAttachments);
		setDebugUtilsObjectName(device->getHandle(), frameData.offscreenFramebuffers[i]->getHandle(), "outputImageFramebuffer for frame #" + std::to_string(i));

		std::vector<VkImageView> postProcessingAttachments{ outputImageView->getHandle(), depthImageView->getHandle() };
		frameData.postProcessFramebuffers[i] = std::make_unique<Framebuffer>(*device, *swapchain, *postRenderPass.renderPass, postProcessingAttachments);
		setDebugUtilsObjectName(device->getHandle(), frameData.postProcessFramebuffers[i]->getHandle(), "postProcessingFramebuffer for frame #" + std::to_string(i));
	}

	// Set clear values
	offscreenFramebufferClearValues.resize(4);
	offscreenFramebufferClearValues[0].color = { 1.0f, 1.0f, 1.0f, 1.0f }; // outputImage
	offscreenFramebufferClearValues[1].color = { 1.0f, 1.0f, 1.0f, 1.0f }; // copyOutputImage
	offscreenFramebufferClearValues[2].color = { 0.0f, 0.0f, 0.0f, 0.0f }; // velocity buffer
	offscreenFramebufferClearValues[3].depthStencil = { 1.0f, 0u };

	postProcessFramebufferClearValues.resize(2);
	postProcessFramebufferClearValues[0].color = { 1.0f, 1.0f, 1.0f, 1.0f }; // outputImage
	postProcessFramebufferClearValues[1].depthStencil = { 1.0f, 0u };
}

void VulkrApp::createCommandPools()
{
	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.commandPools[i] = std::make_unique<CommandPool>(*device, m_graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		setDebugUtilsObjectName(device->getHandle(), frameData.commandPools[i]->getHandle(), "commandPool for frame #" + std::to_string(i));
	}

	for (uint8_t i = 0; i < std::thread::hardware_concurrency(); ++i)
	{
		initCommandPoolIds.push(i);
		initCommandPools.push_back(std::make_unique<CommandPool>(*device, m_graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));
		setDebugUtilsObjectName(device->getHandle(), initCommandPools[i]->getHandle(), "initCommandPool #" + std::to_string(i));
	}
}

uint8_t VulkrApp::getInitCommandPoolId()
{
	std::unique_lock<std::mutex> lock(commandPoolMutex);
	commandPoolCv.wait(lock, [this]()
		{
			return !initCommandPoolIds.empty();
		});

	uint8_t id = initCommandPoolIds.front();
	initCommandPoolIds.pop();
	lock.release();
	commandPoolCv.notify_one();
	return id;
}
void VulkrApp::returnInitCommandPool(uint8_t commandPoolId)
{
	std::unique_lock<std::mutex> lock(commandPoolMutex);
	initCommandPoolIds.push(commandPoolId);
}

void VulkrApp::createCommandBuffers()
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

void VulkrApp::copyBufferToImage(const Buffer &srcBuffer, const Image &dstImage, uint32_t width, uint32_t height)
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

void VulkrApp::createDepthResources()
{
	VkFormat depthFormat = getSupportedDepthFormat(device->getPhysicalDevice().getHandle());

	VkExtent3D extent{ swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1 };
	// If we don't propery synchronize between renderpasses that use the same depth buffer, we could have data hazards
	std::unique_ptr<Image> depthImage = std::make_unique<Image>(*device, depthFormat, extent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY /* default values for remaining params */);
	depthImageView = std::make_unique<ImageView>(std::move(depthImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, depthFormat);

	setDebugUtilsObjectName(device->getHandle(), depthImageView->getImage()->getHandle(), "depthImage");
	setDebugUtilsObjectName(device->getHandle(), depthImageView->getHandle(), "depthImageView");
}

std::unique_ptr<Image> VulkrApp::createTextureImage(const std::string &filename)
{
	int texWidth, texHeight, texChannels;

	stbi_uc *pixels = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	if (!pixels)
	{
		LOGEANDABORT("Failed to load texture image {}!", filename);
	}

	VkDeviceSize imageSize{ static_cast<VkDeviceSize>(texWidth * texHeight * 4) };
	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = imageSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
	stagingBuffer->unmap();

	stbi_image_free(pixels);

	// Create the texture image
	VkExtent3D extent{ to_u32(texWidth), to_u32(texHeight), 1u };
	std::unique_ptr<Image> textureImage = std::make_unique<Image>(*device, VK_FORMAT_R8G8B8A8_SRGB, extent, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY /* default values for remaining params */);

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

	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());

	copyBufferToImage(*stagingBuffer, *textureImage, to_u32(texWidth), to_u32(texHeight));

	// Transition the texture image to be prepared to be read by shaders
	transitionTextureImageLayoutBarrier.pNext = nullptr;
	transitionTextureImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // Wait for transfer to finish as the destination
	transitionTextureImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	transitionTextureImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
	transitionTextureImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
	transitionTextureImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	transitionTextureImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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

	// Use the same submit info
	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());

	return textureImage;
}

void VulkrApp::createTextureSampler()
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

void VulkrApp::copyBufferToBuffer(const Buffer &srcBuffer, const Buffer &dstBuffer, VkDeviceSize size)
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

void VulkrApp::createVertexBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader)
{
	VkDeviceSize bufferSize{ sizeof(objLoader.vertices[0]) * objLoader.vertices.size() };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | m_rayTracingBufferUsageFlags;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	objModelRenderingData.vertexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, objLoader.vertices.data(), static_cast<size_t>(bufferSize));
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *(objModelRenderingData.vertexBuffer), bufferSize);
}

void VulkrApp::createIndexBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader)
{
	VkDeviceSize bufferSize{ sizeof(objLoader.indices[0]) * objLoader.indices.size() };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | m_rayTracingBufferUsageFlags;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	objModelRenderingData.indexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, objLoader.indices.data(), static_cast<size_t>(bufferSize));
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *(objModelRenderingData.indexBuffer), bufferSize);
}

void VulkrApp::createMaterialBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader)
{
	VkDeviceSize bufferSize{ sizeof(objLoader.materials[0]) * objLoader.materials.size() };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | m_rayTracingBufferUsageFlags;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	objModelRenderingData.materialsBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, objLoader.materials.data(), static_cast<size_t>(bufferSize));
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *(objModelRenderingData.materialsBuffer), bufferSize);
}

void VulkrApp::createMaterialBuffer(GltfModelRenderingData &gltfModelRenderingData)
{
	std::vector<GltfMaterial> gltfMaterials{};
	for (auto &material : gltfModelRenderingData.gltfModel.materials)
	{
		GltfMaterial gltfMaterial{};

		gltfMaterial.emissiveFactor = material.emissiveFactor;
		// To save space, availabilty and texture coordinate set are combined
		// -1 = texture not used for this material, >= 0 texture used and index of texture coordinate set
		gltfMaterial.colorTextureSet = material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;
		gltfMaterial.normalTextureSet = material.normalTexture != nullptr ? material.texCoordSets.normal : -1;
		gltfMaterial.occlusionTextureSet = material.occlusionTexture != nullptr ? material.texCoordSets.occlusion : -1;
		gltfMaterial.emissiveTextureSet = material.emissiveTexture != nullptr ? material.texCoordSets.emissive : -1;
		gltfMaterial.alphaMask = static_cast<float>(material.alphaMode == gltf::Material::ALPHAMODE_MASK);
		gltfMaterial.alphaMaskCutoff = material.alphaCutoff;
		gltfMaterial.emissiveStrength = material.emissiveStrength;

		// TODO: glTF specs states that metallic roughness should be preferred, even if specular glosiness is present

		if (material.pbrWorkflows.metallicRoughness)
		{
			// Metallic roughness workflow
			gltfMaterial.workflow = static_cast<float>(PBR_WORKFLOW_METALLIC_ROUGHNESS);
			gltfMaterial.baseColorFactor = material.baseColorFactor;
			gltfMaterial.metallicFactor = material.metallicFactor;
			gltfMaterial.roughnessFactor = material.roughnessFactor;
			gltfMaterial.PhysicalDescriptorTextureSet = material.metallicRoughnessTexture != nullptr ? material.texCoordSets.metallicRoughness : -1;
			gltfMaterial.colorTextureSet = material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;
		}

		if (material.pbrWorkflows.specularGlossiness)
		{
			// Specular glossiness workflow
			gltfMaterial.workflow = static_cast<float>(PBR_WORKFLOW_SPECULAR_GLOSINESS);
			gltfMaterial.PhysicalDescriptorTextureSet = material.extension.specularGlossinessTexture != nullptr ? material.texCoordSets.specularGlossiness : -1;
			gltfMaterial.colorTextureSet = material.extension.diffuseTexture != nullptr ? material.texCoordSets.baseColor : -1;
			gltfMaterial.diffuseFactor = material.extension.diffuseFactor;
			gltfMaterial.specularFactor = glm::vec4(material.extension.specularFactor, 1.0f);
		}

		gltfMaterials.push_back(gltfMaterial);
	}

	VkDeviceSize bufferSize{ gltfMaterials.size() * sizeof(GltfMaterial) };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | m_rayTracingBufferUsageFlags;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	gltfModelRenderingData.materialsBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, gltfMaterials.data(), static_cast<size_t>(bufferSize));
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *(gltfModelRenderingData.materialsBuffer), bufferSize);
}

void VulkrApp::createMaterialIndicesBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader)
{
	VkDeviceSize bufferSize{ sizeof(objLoader.materialIndices[0]) * objLoader.materialIndices.size() };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | m_rayTracingBufferUsageFlags;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	objModelRenderingData.materialsIndexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, objLoader.materialIndices.data(), static_cast<size_t>(bufferSize));
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *(objModelRenderingData.materialsIndexBuffer), bufferSize);
}

void VulkrApp::createUniformBuffers()
{
	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = sizeof(CameraData);
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; // TODO: we should use staging buffers instead

	cameraBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), cameraBuffer->getHandle(), "cameraBuffer");

	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	previousFrameCameraBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), previousFrameCameraBuffer->getHandle(), "previousFrameCameraBuffer");

	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	bufferInfo.size = sizeof(LightData) * maxLightCount;

	lightBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), lightBuffer->getHandle(), "lightBuffer");
}

void VulkrApp::createSSBOs()
{
	VkDeviceSize bufferSize{ sizeof(ObjInstance) * maxInstanceCount };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; // TODO: we should use staging buffers instead

	objectBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), objectBuffer->getHandle(), "objectBuffer");

	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	previousFrameObjectBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), previousFrameObjectBuffer->getHandle(), "previousFrameObjectBuffer");
}

// Setup and fill the compute shader storage buffers containing the particles
void VulkrApp::prepareParticleData()
{
	computeParticlesPushConstant.particleCount = to_u32(attractors.size()) * particlesPerAttractor;

	// Initial particle positions
	allParticleData.resize(computeParticlesPushConstant.particleCount);

	std::default_random_engine      rndEngine((unsigned)time(nullptr));
	std::normal_distribution<float> rndDistribution(0.0f, 1.0f);

	for (uint32_t i = 0; i < to_u32(attractors.size()); i++)
	{
		for (uint32_t j = 0; j < particlesPerAttractor; j++)
		{
			Particle &particle = allParticleData[i * particlesPerAttractor + j];

			// First particle in group as heavy center of gravity
			if (j == 0)
			{
				particle.position = glm::vec4(attractors[i] * 1.5f, 90000.0f);
				particle.velocity = glm::vec4(glm::vec4(0.0f));
			}
			else
			{
				// Position
				glm::vec3 position(attractors[i] + glm::vec3(rndDistribution(rndEngine), rndDistribution(rndEngine), rndDistribution(rndEngine)) * 0.75f);
				float     len = glm::length(glm::normalize(position - attractors[i]));
				position.y *= 2.0f - (len * len);

				// Velocity
				glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
				glm::vec3 velocity = glm::cross((position - attractors[i]), angular) + glm::vec3(rndDistribution(rndEngine), rndDistribution(rndEngine), rndDistribution(rndEngine) * 0.025f);

				float mass = (rndDistribution(rndEngine) * 0.5f + 0.5f) * 75.0f;
				particle.position = glm::vec4(position, mass);
				particle.velocity = glm::vec4(velocity, 0.0f);
			}

			// Color gradient offset
			particle.velocity.w = (float)i * 1.0f / static_cast<uint32_t>(attractors.size());
		}
	}

	particleBufferSize = allParticleData.size() * sizeof(Particle);

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = particleBufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, allParticleData.data(), static_cast<size_t>(particleBufferSize));
	stagingBuffer->unmap();

	// SSBO won't be changed on the host after upload so copy to device local memory
	particleBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), particleBuffer->getHandle(), "particleBuffer");
	copyBufferToBuffer(*stagingBuffer, *(particleBuffer), particleBufferSize);
}

void VulkrApp::createDescriptorPool()
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

void VulkrApp::createDescriptorSets()
{
	// Global Descriptor Set
	VkDescriptorSetAllocateInfo globalDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	globalDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	globalDescriptorSetAllocateInfo.descriptorSetCount = 1;
	globalDescriptorSetAllocateInfo.pSetLayouts = &globalDescriptorSetLayout->getHandle();
	globalDescriptorSet = std::make_unique<DescriptorSet>(*device, globalDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), globalDescriptorSet->getHandle(), "globalDescriptorSet");

	VkDescriptorBufferInfo cameraBufferInfo{};
	cameraBufferInfo.buffer = cameraBuffer->getHandle();
	cameraBufferInfo.offset = 0;
	cameraBufferInfo.range = sizeof(CameraData);

	VkDescriptorBufferInfo lightBufferInfo{};
	lightBufferInfo.buffer = lightBuffer->getHandle();
	lightBufferInfo.offset = 0;
	lightBufferInfo.range = sizeof(LightData) * maxLightCount;
	std::array<VkDescriptorBufferInfo, 2> globalBufferInfos{ cameraBufferInfo, lightBufferInfo };

	VkWriteDescriptorSet writeGlobalDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeGlobalDescriptorSet.dstSet = globalDescriptorSet->getHandle();
	writeGlobalDescriptorSet.dstBinding = 0;
	writeGlobalDescriptorSet.dstArrayElement = 0;
	writeGlobalDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeGlobalDescriptorSet.descriptorCount = to_u32(globalBufferInfos.size());
	writeGlobalDescriptorSet.pBufferInfo = globalBufferInfos.data();
	writeGlobalDescriptorSet.pImageInfo = nullptr;
	writeGlobalDescriptorSet.pTexelBufferView = nullptr;

	// Object Descriptor Set
	VkDescriptorSetAllocateInfo objectDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	objectDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	objectDescriptorSetAllocateInfo.descriptorSetCount = 1;
	objectDescriptorSetAllocateInfo.pSetLayouts = &objectDescriptorSetLayout->getHandle();
	objectDescriptorSet = std::make_unique<DescriptorSet>(*device, objectDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), objectDescriptorSet->getHandle(), "objectDescriptorSet");

	VkDescriptorBufferInfo currentFrameObjectBufferInfo{};
	currentFrameObjectBufferInfo.buffer = objectBuffer->getHandle();
	currentFrameObjectBufferInfo.offset = 0;
	currentFrameObjectBufferInfo.range = sizeof(ObjInstance) * maxInstanceCount;

	VkWriteDescriptorSet writeObjectDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeObjectDescriptorSet.dstSet = objectDescriptorSet->getHandle();
	writeObjectDescriptorSet.dstBinding = 0;
	writeObjectDescriptorSet.dstArrayElement = 0;
	writeObjectDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeObjectDescriptorSet.descriptorCount = 1;
	writeObjectDescriptorSet.pBufferInfo = &currentFrameObjectBufferInfo;
	writeObjectDescriptorSet.pImageInfo = nullptr;
	writeObjectDescriptorSet.pTexelBufferView = nullptr;

	// Post Processing Descriptor Set
	VkDescriptorSetAllocateInfo postProcessingDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	postProcessingDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	postProcessingDescriptorSetAllocateInfo.descriptorSetCount = 1;
	postProcessingDescriptorSetAllocateInfo.pSetLayouts = &postProcessingDescriptorSetLayout->getHandle();
	postProcessingDescriptorSet = std::make_unique<DescriptorSet>(*device, postProcessingDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), postProcessingDescriptorSet->getHandle(), "postProcessingDescriptorSet");

	// Binding 0 is the camera buffer
	std::array<VkDescriptorBufferInfo, 1> postProcessingUniformBufferInfos{ cameraBufferInfo };
	VkWriteDescriptorSet writePostProcessingUniformBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writePostProcessingUniformBufferDescriptorSet.dstSet = postProcessingDescriptorSet->getHandle();
	writePostProcessingUniformBufferDescriptorSet.dstBinding = 0;
	writePostProcessingUniformBufferDescriptorSet.dstArrayElement = 0;
	writePostProcessingUniformBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writePostProcessingUniformBufferDescriptorSet.descriptorCount = to_u32(postProcessingUniformBufferInfos.size());
	writePostProcessingUniformBufferDescriptorSet.pBufferInfo = postProcessingUniformBufferInfos.data();
	writePostProcessingUniformBufferDescriptorSet.pImageInfo = nullptr;
	writePostProcessingUniformBufferDescriptorSet.pTexelBufferView = nullptr;

	// Binding 1 is the currentFrameObjectBuffer
	std::array<VkDescriptorBufferInfo, 1> postProcessingStorageBufferInfos{ currentFrameObjectBufferInfo };
	VkWriteDescriptorSet writePostProcessingStorageBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writePostProcessingStorageBufferDescriptorSet.dstSet = postProcessingDescriptorSet->getHandle();
	writePostProcessingStorageBufferDescriptorSet.dstBinding = 1;
	writePostProcessingStorageBufferDescriptorSet.dstArrayElement = 0;
	writePostProcessingStorageBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writePostProcessingStorageBufferDescriptorSet.descriptorCount = to_u32(postProcessingStorageBufferInfos.size());
	writePostProcessingStorageBufferDescriptorSet.pBufferInfo = postProcessingStorageBufferInfos.data();
	writePostProcessingStorageBufferDescriptorSet.pImageInfo = nullptr;
	writePostProcessingStorageBufferDescriptorSet.pTexelBufferView = nullptr;

	// Bindings 2, 3 and 4 are the history image, velocity image and copy output image respectively
	VkDescriptorImageInfo historyImageInfo{};
	historyImageInfo.sampler = VK_NULL_HANDLE;
	historyImageInfo.imageView = historyImageView->getHandle();
	historyImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkDescriptorImageInfo velocityImageInfo{};
	velocityImageInfo.sampler = VK_NULL_HANDLE;
	velocityImageInfo.imageView = velocityImageView->getHandle();
	velocityImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkDescriptorImageInfo copyOutputImageInfo{};
	copyOutputImageInfo.sampler = VK_NULL_HANDLE;
	copyOutputImageInfo.imageView = copyOutputImageView->getHandle();
	copyOutputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	std::array<VkDescriptorImageInfo, 3> postProcessingStorageImageInfos{ historyImageInfo, velocityImageInfo, copyOutputImageInfo };

	VkWriteDescriptorSet writePostProcessingStorageImageDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writePostProcessingStorageImageDescriptorSet.dstSet = postProcessingDescriptorSet->getHandle();
	writePostProcessingStorageImageDescriptorSet.dstBinding = 2;
	writePostProcessingStorageImageDescriptorSet.dstArrayElement = 0;
	writePostProcessingStorageImageDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writePostProcessingStorageImageDescriptorSet.descriptorCount = to_u32(postProcessingStorageImageInfos.size());
	writePostProcessingStorageImageDescriptorSet.pImageInfo = postProcessingStorageImageInfos.data();
	writePostProcessingStorageImageDescriptorSet.pBufferInfo = nullptr;
	writePostProcessingStorageImageDescriptorSet.pTexelBufferView = nullptr;

	// Taa Descriptor Set
	VkDescriptorSetAllocateInfo taaDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	taaDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	taaDescriptorSetAllocateInfo.descriptorSetCount = 1;
	taaDescriptorSetAllocateInfo.pSetLayouts = &taaDescriptorSetLayout->getHandle();
	taaDescriptorSet = std::make_unique<DescriptorSet>(*device, taaDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), taaDescriptorSet->getHandle(), "taaDescriptorSet");

	// Binding 0 is the previous frame camera buffer
	VkDescriptorBufferInfo previousFrameCameraBufferInfo{};
	previousFrameCameraBufferInfo.buffer = previousFrameCameraBuffer->getHandle();
	previousFrameCameraBufferInfo.offset = 0;
	previousFrameCameraBufferInfo.range = sizeof(CameraData);
	std::array<VkDescriptorBufferInfo, 1> taaUniformBufferInfos{ previousFrameCameraBufferInfo };

	VkWriteDescriptorSet writeTaaUniformBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeTaaUniformBufferDescriptorSet.dstSet = taaDescriptorSet->getHandle();
	writeTaaUniformBufferDescriptorSet.dstBinding = 0;
	writeTaaUniformBufferDescriptorSet.dstArrayElement = 0;
	writeTaaUniformBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeTaaUniformBufferDescriptorSet.descriptorCount = to_u32(taaUniformBufferInfos.size());
	writeTaaUniformBufferDescriptorSet.pBufferInfo = taaUniformBufferInfos.data();
	writeTaaUniformBufferDescriptorSet.pImageInfo = nullptr;
	writeTaaUniformBufferDescriptorSet.pTexelBufferView = nullptr;

	// Binding 1 is the previous frame object buffer
	VkDescriptorBufferInfo previousFrameObjectBufferInfo{};
	previousFrameObjectBufferInfo.buffer = previousFrameObjectBuffer->getHandle();
	previousFrameObjectBufferInfo.offset = 0;
	previousFrameObjectBufferInfo.range = sizeof(ObjInstance) * maxInstanceCount;
	std::array<VkDescriptorBufferInfo, 1> taaStorageImageInfos{ previousFrameObjectBufferInfo };

	VkWriteDescriptorSet writeTaaStorageBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeTaaStorageBufferDescriptorSet.dstSet = taaDescriptorSet->getHandle();
	writeTaaStorageBufferDescriptorSet.dstBinding = 1;
	writeTaaStorageBufferDescriptorSet.dstArrayElement = 0;
	writeTaaStorageBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeTaaStorageBufferDescriptorSet.descriptorCount = to_u32(taaStorageImageInfos.size());
	writeTaaStorageBufferDescriptorSet.pBufferInfo = taaStorageImageInfos.data();
	writeTaaStorageBufferDescriptorSet.pImageInfo = nullptr;
	writeTaaStorageBufferDescriptorSet.pTexelBufferView = nullptr;

	// Particle Buffer Descriptor Set
	VkDescriptorSetAllocateInfo particleComputeDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	particleComputeDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	particleComputeDescriptorSetAllocateInfo.descriptorSetCount = 1;
	particleComputeDescriptorSetAllocateInfo.pSetLayouts = &particleComputeDescriptorSetLayout->getHandle();
	particleComputeDescriptorSet = std::make_unique<DescriptorSet>(*device, particleComputeDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), particleComputeDescriptorSet->getHandle(), "particleComputeDescriptorSet");

	// Binding 0 is the particle buffer
	VkDescriptorBufferInfo particleBufferInfo{};
	particleBufferInfo.buffer = particleBuffer->getHandle();
	particleBufferInfo.offset = 0;
	particleBufferInfo.range = sizeof(Particle) * computeParticlesPushConstant.particleCount;
	std::array<VkDescriptorBufferInfo, 1> particleComputeStorageBufferInfos{ particleBufferInfo };

	VkWriteDescriptorSet writeParticleComputeStorageBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeParticleComputeStorageBufferDescriptorSet.dstSet = particleComputeDescriptorSet->getHandle();
	writeParticleComputeStorageBufferDescriptorSet.dstBinding = 0;
	writeParticleComputeStorageBufferDescriptorSet.dstArrayElement = 0;
	writeParticleComputeStorageBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeParticleComputeStorageBufferDescriptorSet.descriptorCount = to_u32(particleComputeStorageBufferInfos.size());
	writeParticleComputeStorageBufferDescriptorSet.pBufferInfo = particleComputeStorageBufferInfos.data();
	writeParticleComputeStorageBufferDescriptorSet.pImageInfo = nullptr;
	writeParticleComputeStorageBufferDescriptorSet.pTexelBufferView = nullptr;

	// Texture Descriptor Set
	VkDescriptorSetAllocateInfo textureDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	textureDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
	textureDescriptorSetAllocateInfo.descriptorSetCount = 1;
	textureDescriptorSetAllocateInfo.pSetLayouts = &textureDescriptorSetLayout->getHandle();
	textureDescriptorSet = std::make_unique<DescriptorSet>(*device, textureDescriptorSetAllocateInfo);
	setDebugUtilsObjectName(device->getHandle(), textureDescriptorSet->getHandle(), "textureDescriptorSet");

	std::vector<VkDescriptorImageInfo> textureImageInfos;
	for (auto &texture : textureImageViews)
	{
		VkDescriptorImageInfo textureImageInfo{};
		textureImageInfo.sampler = textureSampler->getHandle();
		textureImageInfo.imageView = texture->getHandle();
		textureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureImageInfos.push_back(textureImageInfo);
	}

	// Binding 0 is the texture image
	VkWriteDescriptorSet writeTextureDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeTextureDescriptorSet.dstSet = textureDescriptorSet->getHandle();
	writeTextureDescriptorSet.dstBinding = 0;
	writeTextureDescriptorSet.dstArrayElement = 0;
	writeTextureDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writeTextureDescriptorSet.descriptorCount = to_u32(textureImageInfos.size());
	writeTextureDescriptorSet.pImageInfo = textureImageInfos.data();
	writeTextureDescriptorSet.pBufferInfo = nullptr;
	writeTextureDescriptorSet.pTexelBufferView = nullptr;


	std::array<VkWriteDescriptorSet, 9> writeDescriptorSets
	{
		writeGlobalDescriptorSet,
		writeObjectDescriptorSet,

		writePostProcessingUniformBufferDescriptorSet,
		writePostProcessingStorageBufferDescriptorSet,
		writePostProcessingStorageImageDescriptorSet,

		writeTaaUniformBufferDescriptorSet,
		writeTaaStorageBufferDescriptorSet,

		writeParticleComputeStorageBufferDescriptorSet,
		writeTextureDescriptorSet
	};
	vkUpdateDescriptorSets(device->getHandle(), to_u32(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);


	// Per-Material descriptor sets
	for (auto &gltfModelToRender : gltfModelsToRender)
	{
		for (auto &material : gltfModelToRender.gltfModel.materials)
		{
			VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
			descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			descriptorSetAllocInfo.descriptorPool = descriptorPool->getHandle();
			descriptorSetAllocInfo.pSetLayouts = &gltfMaterialSamplersDescriptorSetLayout->getHandle();
			descriptorSetAllocInfo.descriptorSetCount = 1u;
			VK_CHECK(vkAllocateDescriptorSets(device->getHandle(), &descriptorSetAllocInfo, &material.descriptorSet));

			// TODO currently we're passing in the textureImageViews[0] (could use historyImageView) as the "default" but it shouldn't be used based on the flag in GltfMaterial; 
			// should check if this works, this also assumes that there is atelast one texture
			VkDescriptorImageInfo defaultEmptyDescriptorImageInfo{};
			defaultEmptyDescriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			defaultEmptyDescriptorImageInfo.imageView = textureImageViews[0]->getHandle();
			defaultEmptyDescriptorImageInfo.sampler = textureSampler->getHandle();

			std::vector<VkDescriptorImageInfo> imageDescriptors = {
				defaultEmptyDescriptorImageInfo,
				defaultEmptyDescriptorImageInfo,
				material.normalTexture ? material.normalTexture->descriptor : defaultEmptyDescriptorImageInfo,
				material.occlusionTexture ? material.occlusionTexture->descriptor : defaultEmptyDescriptorImageInfo,
				material.emissiveTexture ? material.emissiveTexture->descriptor : defaultEmptyDescriptorImageInfo
			};

			// TODO: glTF specs states that metallic roughness should be preferred, even if specular glosiness is present

			if (material.pbrWorkflows.metallicRoughness)
			{
				if (material.baseColorTexture)
				{
					imageDescriptors[0] = material.baseColorTexture->descriptor;
				}
				if (material.metallicRoughnessTexture)
				{
					imageDescriptors[1] = material.metallicRoughnessTexture->descriptor;
				}
			}

			if (material.pbrWorkflows.specularGlossiness)
			{
				if (material.extension.diffuseTexture)
				{
					imageDescriptors[0] = material.extension.diffuseTexture->descriptor;
				}
				if (material.extension.specularGlossinessTexture)
				{
					imageDescriptors[1] = material.extension.specularGlossinessTexture->descriptor;
				}
			}

			std::array<VkWriteDescriptorSet, 5> writeDescriptorSets{};
			for (size_t i = 0; i < imageDescriptors.size(); i++)
			{
				writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				writeDescriptorSets[i].descriptorCount = 1u;
				writeDescriptorSets[i].dstSet = material.descriptorSet;
				writeDescriptorSets[i].dstBinding = static_cast<uint32_t>(i);
				writeDescriptorSets[i].pImageInfo = &imageDescriptors[i];
			}

			vkUpdateDescriptorSets(device->getHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}

		// Model node (matrices)
		{
			// Per-Node descriptor set
			for (auto &node : gltfModelToRender.gltfModel.nodes)
			{
				setupNodeDescriptorSet(node);
			}
		}

		// Material Buffer
		{
			VkDescriptorSetAllocateInfo gltfMaterialsDescriptorSetAllocateInfo{};
			gltfMaterialsDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			gltfMaterialsDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
			gltfMaterialsDescriptorSetAllocateInfo.pSetLayouts = &gltfMaterialDescriptorSetLayout->getHandle();
			gltfMaterialsDescriptorSetAllocateInfo.descriptorSetCount = 1u;
			gltfMaterialDescriptorSet = std::make_unique<DescriptorSet>(*device, gltfMaterialsDescriptorSetAllocateInfo);

			VkDescriptorBufferInfo gltfMaterialBufferInfo{};
			gltfMaterialBufferInfo.buffer = gltfModelToRender.materialsBuffer->getHandle();
			gltfMaterialBufferInfo.offset = 0;
			gltfMaterialBufferInfo.range = sizeof(GltfMaterial) * gltfModelToRender.gltfModel.materials.size();

			VkWriteDescriptorSet writeDescriptorSet{};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.dstSet = gltfMaterialDescriptorSet->getHandle();
			writeDescriptorSet.dstBinding = 0;
			writeDescriptorSet.pBufferInfo = &gltfMaterialBufferInfo;
			vkUpdateDescriptorSets(device->getHandle(), 1, &writeDescriptorSet, 0, nullptr);
		}
	}
}

void VulkrApp::setupNodeDescriptorSet(vulkr::gltf::Node *node)
{
	if (node->mesh)
	{
		VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
		descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descriptorSetAllocInfo.descriptorPool = descriptorPool->getHandle();
		descriptorSetAllocInfo.pSetLayouts = &gltfNodeDescriptorSetLayout->getHandle();
		descriptorSetAllocInfo.descriptorSetCount = 1u;
		VK_CHECK(vkAllocateDescriptorSets(device->getHandle(), &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet));

		VkWriteDescriptorSet writeDescriptorSet{};
		writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSet.descriptorCount = 1;
		writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
		writeDescriptorSet.dstBinding = 0;
		writeDescriptorSet.pBufferInfo = &node->mesh->uniformBuffer.descriptor;

		vkUpdateDescriptorSets(device->getHandle(), 1, &writeDescriptorSet, 0, nullptr);
	}

	for (auto &child : node->children)
	{
		setupNodeDescriptorSet(child);
	}
}

void VulkrApp::createSemaphoreAndFencePools()
{
	semaphorePool = std::make_unique<SemaphorePool>(*device);
	fencePool = std::make_unique<FencePool>(*device);
}

void VulkrApp::loadTextureImages(const std::vector<std::string> &textureFiles)
{
	LOGI("Loading {} texture files..", std::to_string(textureFiles.size()));
	int x = 1;
	for (const std::string &textureFile : textureFiles)
	{
		LOGI(std::to_string(x++) + ": processing " + textureFile);

		std::unique_ptr<Image> imageOfTexture = createTextureImage(std::string("../../assets/textures/" + textureFile).c_str());
		VkFormat textureImageFormat = imageOfTexture->getFormat();

		std::unique_ptr<ImageView> texture = std::make_unique<ImageView>(std::move(imageOfTexture), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, textureImageFormat);
		textureImageViews.emplace_back(std::move(texture));
	}
}

void VulkrApp::setupSynchronizationObjects()
{
	imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);

	for (size_t i = 0; i < maxFramesInFlight; ++i)
	{
		frameData.imageAvailableSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.offscreenRenderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.postProcessRenderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.outputImageCopyFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.computeParticlesFinishedSemaphores[i] = semaphorePool->requestSemaphore();
		frameData.inFlightFences[i] = fencePool->requestFence();
	}
}

void VulkrApp::loadObjModel(const std::string &objFilePath)
{
	ObjModelRenderingData objModelRenderingData;
	ObjLoader objLoader;
	objLoader.loadModel(objFilePath.c_str());

	// Applying gamma correction to convert the ambient, diffuse and specular values from srgb non-linear to srgb linear prior to usage in shaders
	for (auto &m : objLoader.materials)
	{
		m.ambient = glm::pow(m.ambient, glm::vec3(2.2f));
		m.diffuse = glm::pow(m.diffuse, glm::vec3(2.2f));
		m.specular = glm::pow(m.specular, glm::vec3(2.2f));
	}

	objModelRenderingData.verticesCount = to_u32(objLoader.vertices.size());
	objModelRenderingData.indicesCount = to_u32(objLoader.indices.size());

#ifdef MULTI_THREAD
	std::scoped_lock<std::mutex> bufferLock(bufferMutex); // Lock
#endif
	createVertexBuffer(objModelRenderingData, objLoader);
	createIndexBuffer(objModelRenderingData, objLoader);
	createMaterialBuffer(objModelRenderingData, objLoader);
	createMaterialIndicesBuffer(objModelRenderingData, objLoader);

	std::string objNb = std::to_string(objModelsToRender.size());
	setDebugUtilsObjectName(device->getHandle(), objModelRenderingData.vertexBuffer->getHandle(), (std::string("vertex_" + objNb).c_str()));
	setDebugUtilsObjectName(device->getHandle(), objModelRenderingData.indexBuffer->getHandle(), (std::string("index_" + objNb).c_str()));
	setDebugUtilsObjectName(device->getHandle(), objModelRenderingData.materialsBuffer->getHandle(), (std::string("mat_" + objNb).c_str()));
	setDebugUtilsObjectName(device->getHandle(), objModelRenderingData.materialsIndexBuffer->getHandle(), (std::string("matIdx_" + objNb).c_str()));

	objModelRenderingData.objFilePath = objFilePath;
	objModelRenderingData.txtOffset = static_cast<uint64_t>(textureImageViews.size());
	loadTextureImages(objLoader.textures);

	objModelsToRender.push_back(std::move(objModelRenderingData));
}

void VulkrApp::loadGltfModel(const std::string &gltfFilePath)
{
	LOGI("Loading scene from {}", gltfFilePath);
	GltfModelRenderingData gltfModelRenderingData;
	animationIndex = 0;
	animationTimer = 0.0f;
	auto tStart = std::chrono::high_resolution_clock::now();
	gltfModelRenderingData.gltfModel.loadFromFile(gltfFilePath, device.get(), frameData.commandPools[0].get(), m_transferQueue->getHandle());
	createMaterialBuffer(gltfModelRenderingData);
	auto tFileLoad = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tStart).count();
	LOGI("Loading took {} ms", tFileLoad);

	gltfModelsToRender.push_back(std::move(gltfModelRenderingData));
}

void VulkrApp::createObjInstance(const std::string &objFilePath, glm::mat4 transform)
{
	uint64_t objModelIndex = getObjModelIndex(objFilePath);
	ObjModelRenderingData &objModelRenderingData = objModelsToRender[objModelIndex];

	ObjInstance instance;
	instance.transform = transform;
	instance.transformIT = glm::transpose(glm::inverse(transform));
	instance.objIndex = objModelIndex;
	instance.textureOffset = objModelRenderingData.txtOffset;

	VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR };
	bufferDeviceAddressInfo.buffer = objModelRenderingData.vertexBuffer->getHandle();
	instance.vertices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
	bufferDeviceAddressInfo.buffer = objModelRenderingData.indexBuffer->getHandle();
	instance.indices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
	bufferDeviceAddressInfo.buffer = objModelRenderingData.materialsBuffer->getHandle();
	instance.materials = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
	bufferDeviceAddressInfo.buffer = objModelRenderingData.materialsIndexBuffer->getHandle();
	instance.materialIndices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

	objInstances.push_back(std::move(instance));
}

void VulkrApp::createSceneLights()
{
	LightData l1;
	l1.rotation = glm::vec2(75.0f, 40.0f);
	LightData l2;
	l2.position = glm::vec3(2.5f, 3.5f, -3.0f);
	l2.color = glm::vec3(1.0f, 1.0f, 0.0f);
	l2.rotation = glm::vec2(160.0f, 40.0f);
	sceneLights.emplace_back(std::move(l1));
	sceneLights.emplace_back(std::move(l2));

	if (sceneLights.size() > maxLightCount)
	{
		LOGEANDABORT("Lights in the scene exceed the maxLightCount of {}", maxLightCount);
	}

	rasterizationPushConstant.lightCount = static_cast<int>(sceneLights.size());
	raytracingPushConstant.lightCount = static_cast<int>(sceneLights.size());
}

void VulkrApp::loadModels()
{
	const std::array<std::string, 4> objModelFiles{
		"plane.obj",
		"Medieval_building.obj",
		"wuson.obj",
		"cube.obj"
		//"monkey_smooth.obj",
		//"lost_empire.obj",
	};

	const std::array<std::string, 1> gltfModelFiles{
		"DamagedHelmet.gltf"
		//"kitchen_room_interior.glb" // TODO gltf binary files are not working for some reason, need to investigate
	};

	// Load obj files
	std::set<std::string> existingModels;
	for (const std::string &objModelFile : objModelFiles)
	{
		std::string filePath = defaultModelFilePath + objModelFile;
		std::ifstream file(filePath);
		if (!file.good())
		{
			LOGE("{} does not exist!", objModelFile);
			std::abort();
		}

		if (existingModels.count(objModelFile) != 0)
		{
			LOGW("Duplicate obj models can not be loaded!");
		}
		else
		{
			existingModels.insert(objModelFile);
			loadObjModel(filePath);
		}
	}

	// Load gltf files
	existingModels.clear();
	for (const std::string &gltfModelFile : gltfModelFiles)
	{
		std::string filePath = defaultModelFilePath + gltfModelFile;
		std::ifstream file(filePath);
		if (!file.good())
		{
			LOGE("{} does not exist!", gltfModelFile);
			std::abort();
		}

		if (existingModels.count(gltfModelFile) != 0)
		{
			LOGW("Duplicate gltf models can not be loaded!");
		}
		else
		{
			existingModels.insert(gltfModelFile);
			loadGltfModel(filePath);
		}
	}

#ifdef MULTI_THREAD
	// TODO: we need to set up separate threads to load the obj and gltf files above (eg. modelLoadThreads.push_back(std::thread(&VulkrApp::loadModel, this, modelFile)) and then run threads.join 
	for (auto &threads : modelLoadThreads)
	{
		threads.join();
	}
#endif
}

void VulkrApp::createSceneInstances()
{
	if (maxInstanceCount > device->getPhysicalDevice().getAccelerationStructureProperties().maxInstanceCount)
	{
		LOGEANDABORT("Max instance count is above the limit supported by the GPU");
	}

	// The sphere instance index is hardcoded in the animate.comp file, so when you add or remove an instance, that must be updated
	createObjInstance(defaultModelFilePath + "plane.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(0, 0, 0)));
	createObjInstance(defaultModelFilePath + "Medieval_building.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 5, 0,0 }));
	// All wuson instances are assumed to be one after another for the transformation matrix calculations
	createObjInstance(defaultModelFilePath + "wuson.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 3)));
	createObjInstance(defaultModelFilePath + "wuson.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 7)));
	createObjInstance(defaultModelFilePath + "wuson.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 10)));
	createObjInstance(defaultModelFilePath + "Medieval_building.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 15, 0,0 }));
	//createObjInstance(defaultModelFilePath +"monkey_smooth.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 3)));
	//createObjInstance(defaultModelFilePath +"lost_empire.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 5,-10,0 }));

	// ALl particle instances are assumed to be grouped together
	computeParticlesPushConstant.startingIndex = static_cast<int>(objInstances.size());
	for (int i = 0; i < computeParticlesPushConstant.particleCount; ++i)
	{
		createObjInstance(defaultModelFilePath + "cube.obj", glm::translate(glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2f, 0.2f, 0.2f)), glm::vec3(allParticleData[i].position.xyz)));
	}

	if (objInstances.size() > maxInstanceCount)
	{
		LOGEANDABORT("There are more obj instances than maxInstanceCount. You need to increase this value to support more instances");
	}
}

uint64_t VulkrApp::getObjModelIndex(const std::string &name)
{
	for (uint64_t i = 0; i < objModelsToRender.size(); ++i)
	{
		if (objModelsToRender[i].objFilePath.compare(name) == 0) return i;
	}

	LOGEANDABORT("Object model {} not found", name);
}

void VulkrApp::initializeImGui()
{
	// Create descriptor pool for imgui
	std::vector<VkDescriptorPoolSize> poolSizes =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};
	imguiPool = std::make_unique<DescriptorPool>(*device, poolSizes, 1000 * 11, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	ImGui::StyleColorsClassic();

	// Initialize imgui for glfw
	ImGui_ImplGlfw_InitForVulkan(platform.getWindow().getHandle(), true);

	// Initilialize imgui for Vulkan
	ImGui_ImplVulkan_InitInfo initInfo = {};
	initInfo.Instance = m_instance->getHandle();
	initInfo.PhysicalDevice = device->getPhysicalDevice().getHandle();
	initInfo.Device = device->getHandle();
	initInfo.QueueFamily = m_graphicsQueue->getFamilyIndex();
	initInfo.Queue = m_graphicsQueue->getHandle();
	initInfo.PipelineCache = VK_NULL_HANDLE;
	initInfo.DescriptorPool = imguiPool->getHandle();
	initInfo.Subpass = 0u;
	initInfo.MinImageCount = swapchain->getProperties().imageCount;
	initInfo.ImageCount = swapchain->getProperties().imageCount;
	initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	initInfo.Allocator = device->getAllocationCallbacks(); // TODO pass VMA here; figure out if this solution works
	initInfo.CheckVkResultFn = checkVkResult;

	ImGui_ImplVulkan_LoadFunctions(loadFunction); // TODO figure out how to integrate with volk
	ImGui_ImplVulkan_Init(&initInfo, postRenderPass.renderPass->getHandle());

	std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	ImGui_ImplVulkan_CreateFontsTexture(commandBuffer->getHandle());

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

	// Clear font data on CPU
	ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void VulkrApp::resetFrameSinceViewChange()
{
	raytracingPushConstant.frameSinceViewChange = -1; // TODO remove
	taaPushConstant.frameSinceViewChange = -1;
}

void VulkrApp::createImageResourcesForFrames()
{
	VkExtent3D extent{ swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1u };
	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0u;
	subresourceRange.levelCount = 1u;
	subresourceRange.baseArrayLayer = 0u;
	subresourceRange.layerCount = 1u;

	// Create textureImageViews
	std::unique_ptr<Image> outputImage = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VkFormat outputImageFormat = outputImage->getFormat();
	outputImageView = std::make_unique<ImageView>(std::move(outputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, outputImageFormat);

	std::unique_ptr<Image> copyOutputImage = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VkFormat copyOutputImageFormat = copyOutputImage->getFormat();
	copyOutputImageView = std::make_unique<ImageView>(std::move(copyOutputImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, copyOutputImageFormat);

	std::unique_ptr<Image> historyImage = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VkFormat historyImageFormat = historyImage->getFormat();
	historyImageView = std::make_unique<ImageView>(std::move(historyImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, historyImageFormat);

	std::unique_ptr<Image> velocityImage = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VkFormat velocityImageFormat = velocityImage->getFormat();
	velocityImageView = std::make_unique<ImageView>(std::move(velocityImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, velocityImageFormat);

	// Set debug markers
	setDebugUtilsObjectName(device->getHandle(), outputImageView->getImage()->getHandle(), "outputImage");
	setDebugUtilsObjectName(device->getHandle(), outputImageView->getHandle(), "outputImageView");
	setDebugUtilsObjectName(device->getHandle(), copyOutputImageView->getImage()->getHandle(), "copyOutputImage");
	setDebugUtilsObjectName(device->getHandle(), copyOutputImageView->getHandle(), "copyOutputImageView");
	setDebugUtilsObjectName(device->getHandle(), historyImageView->getImage()->getHandle(), "historyImage");
	setDebugUtilsObjectName(device->getHandle(), historyImageView->getHandle(), "historyImageView");
	setDebugUtilsObjectName(device->getHandle(), velocityImageView->getImage()->getHandle(), "velocityImage");
	setDebugUtilsObjectName(device->getHandle(), velocityImageView->getHandle(), "velocityImageView");

	// Create memory barriers for layout transition
	VkImageMemoryBarrier2 transitionOutputImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionOutputImageLayoutBarrier.pNext = nullptr;
	transitionOutputImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // No synchronization required on initial setup
	transitionOutputImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionOutputImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionOutputImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionOutputImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	transitionOutputImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionOutputImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionOutputImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionOutputImageLayoutBarrier.image = outputImageView->getImage()->getHandle();
	transitionOutputImageLayoutBarrier.subresourceRange = subresourceRange;

	VkImageMemoryBarrier2 transitionCopyOutputImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionCopyOutputImageLayoutBarrier.pNext = nullptr;
	transitionCopyOutputImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // No synchronization required on initial setup
	transitionCopyOutputImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionCopyOutputImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionCopyOutputImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionCopyOutputImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	transitionCopyOutputImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionCopyOutputImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionCopyOutputImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionCopyOutputImageLayoutBarrier.image = copyOutputImageView->getImage()->getHandle();
	transitionCopyOutputImageLayoutBarrier.subresourceRange = subresourceRange;

	VkImageMemoryBarrier2 transitionHistoryImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionHistoryImageLayoutBarrier.pNext = nullptr;
	transitionHistoryImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // No synchronization required on initial setup
	transitionHistoryImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionHistoryImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionHistoryImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionHistoryImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	transitionHistoryImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionHistoryImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionHistoryImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionHistoryImageLayoutBarrier.image = historyImageView->getImage()->getHandle();
	transitionHistoryImageLayoutBarrier.subresourceRange = subresourceRange;

	VkImageMemoryBarrier2 transitionVelocityImageLayoutBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	transitionVelocityImageLayoutBarrier.pNext = nullptr;
	transitionVelocityImageLayoutBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE; // No synchronization required on initial setup
	transitionVelocityImageLayoutBarrier.srcAccessMask = VK_ACCESS_2_NONE;
	transitionVelocityImageLayoutBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	transitionVelocityImageLayoutBarrier.dstAccessMask = VK_ACCESS_2_NONE;
	transitionVelocityImageLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	transitionVelocityImageLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	transitionVelocityImageLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionVelocityImageLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transitionVelocityImageLayoutBarrier.image = velocityImageView->getImage()->getHandle();
	transitionVelocityImageLayoutBarrier.subresourceRange = subresourceRange;

	std::vector<VkImageMemoryBarrier2> imageTransitionMemoryBarriers;
	imageTransitionMemoryBarriers.push_back(transitionOutputImageLayoutBarrier);
	imageTransitionMemoryBarriers.push_back(transitionCopyOutputImageLayoutBarrier);
	imageTransitionMemoryBarriers.push_back(transitionHistoryImageLayoutBarrier);
	imageTransitionMemoryBarriers.push_back(transitionVelocityImageLayoutBarrier);

	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 0u;
	dependencyInfo.pMemoryBarriers = nullptr;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = to_u32(imageTransitionMemoryBarriers.size());
	dependencyInfo.pImageMemoryBarriers = imageTransitionMemoryBarriers.data();

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
	submitInfo.flags = 0u;
	submitInfo.waitSemaphoreInfoCount = 0u;
	submitInfo.pWaitSemaphoreInfos = nullptr;
	submitInfo.commandBufferInfoCount = 1u;
	submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
	submitInfo.signalSemaphoreInfoCount = 0u;
	submitInfo.pSignalSemaphoreInfos = nullptr;

	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	// TODO evaluate whether we can reduce the number of vkQueueWaitIdle and have just one at the end of the setup phrase before the main rendering loop begins
	vkQueueWaitIdle(m_graphicsQueue->getHandle());
}

// Convert an OBJ model into the ray tracing geometry used to build the BLAS
BlasInput VulkrApp::objectToVkGeometryKHR(size_t objModelIndex)
{
	VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	bufferDeviceAddressInfo.buffer = objModelsToRender[objModelIndex].vertexBuffer->getHandle();
	// We can take advantage of the fact that position is the first member of Vertex
	VkDeviceAddress vertexAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
	bufferDeviceAddressInfo.buffer = objModelsToRender[objModelIndex].indexBuffer->getHandle();
	VkDeviceAddress indexAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

	uint32_t maxPrimitiveCount = objModelsToRender[objModelIndex].indicesCount / 3;

	// Describe buffer as array of VertexObj.
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
	triangles.vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
	triangles.vertexData.deviceAddress = vertexAddress;
	triangles.vertexStride = sizeof(VertexObj);
	// Describe index data (32-bit unsigned int)
	triangles.indexType = VK_INDEX_TYPE_UINT32;
	triangles.indexData.deviceAddress = indexAddress;
	// Indicate identity transform by setting transformData to null device pointer.
	//triangles.transformData = {};
	triangles.maxVertex = objModelsToRender[objModelIndex].verticesCount;

	// Identify the above data as containing opaque triangles.
	VkAccelerationStructureGeometryKHR asGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	asGeometry.geometry.triangles = triangles;

	// The entire array will be used to build the BLAS.
	VkAccelerationStructureBuildRangeInfoKHR offset;
	offset.firstVertex = 0;
	offset.primitiveCount = maxPrimitiveCount;
	offset.primitiveOffset = 0;
	offset.transformOffset = 0;

	// Our blas is made from only one geometry, but could be made of many geometries
	BlasInput blasInput;
	blasInput.asGeometry.emplace_back(asGeometry);
	blasInput.asBuildRangeInfo.emplace_back(offset);

	return blasInput;
}

std::unique_ptr<AccelerationStructure> VulkrApp::createAccelerationStructure(VkAccelerationStructureCreateInfoKHR &accelerationStructureInfo)
{
	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = accelerationStructureInfo.size;
	bufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	std::unique_ptr<AccelerationStructure> newAccelerationStructure = std::make_unique<AccelerationStructure>(*device);
	newAccelerationStructure->buffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo); // Allocating the buffer to hold the acceleration structure
	accelerationStructureInfo.buffer = newAccelerationStructure->buffer->getHandle();

	// Create the acceleration structure
	vkCreateAccelerationStructureKHR(device->getHandle(), &accelerationStructureInfo, nullptr, &newAccelerationStructure->accelerationStuctureKHR);

	return std::move(newAccelerationStructure);
}


void VulkrApp::buildBlas()
{
	// TODO: enable compaction
	VkBuildAccelerationStructureFlagsKHR buildAccelerationStructureFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR/* | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR */;

	std::vector<BlasInput> blasInputs;
	blasInputs.reserve(objModelsToRender.size());
	for (uint32_t i = 0; i < objModelsToRender.size(); ++i)
	{
		BlasInput blasInput = objectToVkGeometryKHR(i);
		blasInputs.emplace_back(blasInput); // We could add more geometry in each BLAS, but we add only one for now
	}

	m_blas = std::vector<BlasEntry>(blasInputs.begin(), blasInputs.end());
	uint32_t blasCount = to_u32(m_blas.size());

	// Preparing the build information array for the acceleration build command.
	// This is mostly just a fancy pointer to the user-passed arrays of VkAccelerationStructureGeometryKHR.
	// dstAccelerationStructure will be filled later once we allocated the acceleration structures.
	std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(blasCount);
	for (uint32_t idx = 0; idx < blasCount; idx++)
	{
		buildInfos[idx].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		buildInfos[idx].flags = buildAccelerationStructureFlags;
		buildInfos[idx].geometryCount = (uint32_t)m_blas[idx].input.asGeometry.size();
		buildInfos[idx].pGeometries = m_blas[idx].input.asGeometry.data();
		buildInfos[idx].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		buildInfos[idx].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildInfos[idx].srcAccelerationStructure = VK_NULL_HANDLE;
	}

	// Finding sizes to create acceleration structures and scratch
	// Keep the largest scratch buffer size, to use only one scratch for all build
	VkDeviceSize              maxScratch{ 0ull };          // Largest scratch buffer for our BLAS
	std::vector<VkDeviceSize> originalSizes(blasCount);  // use for stats

	for (size_t idx = 0; idx < blasCount; idx++)
	{
		// Query both the size of the finished acceleration structure and the  amount of scratch memory
		// needed (both written to sizeInfo). The `vkGetAccelerationStructureBuildSizesKHR` function
		// computes the worst case memory requirements based on the user-reported max number of
		// primitives. Later, compaction can fix this potential inefficiency.
		std::vector<uint32_t> maxPrimCount(m_blas[idx].input.asBuildRangeInfo.size());
		for (auto tt = 0; tt < m_blas[idx].input.asBuildRangeInfo.size(); tt++)
			maxPrimCount[tt] = m_blas[idx].input.asBuildRangeInfo[tt].primitiveCount;  // Number of primitives/triangles
		VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device->getHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfos[idx],
			maxPrimCount.data(), &sizeInfo);

		// Create acceleration structure object. Not yet bound to memory.
		VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		createInfo.size = sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.

		// Actual allocation of buffer and acceleration structure. Note: This relies on createInfo.offset == 0
		// and fills in createInfo.buffer with the buffer allocated to store the BLAS. The underlying
		// vkCreateAccelerationStructureKHR call then consumes the buffer value.
		m_blas[idx].accelerationStructure = createAccelerationStructure(createInfo);

		setDebugUtilsObjectName(device->getHandle(), m_blas[idx].accelerationStructure->accelerationStuctureKHR, std::string("BLAS Acceleration Structure #" + std::to_string(idx)));
		setDebugUtilsObjectName(device->getHandle(), m_blas[idx].accelerationStructure->buffer->getHandle(), std::string("BLAS Acceleration Structure Buffer #" + std::to_string(idx)));
		buildInfos[idx].dstAccelerationStructure = m_blas[idx].accelerationStructure->accelerationStuctureKHR;  // Setting the where the build lands

		// Keeping info
		m_blas[idx].flags = buildAccelerationStructureFlags;
		maxScratch = std::max(maxScratch, sizeInfo.buildScratchSize);

		// Stats - Original size
		originalSizes[idx] = sizeInfo.accelerationStructureSize;
	}

	// Allocate the scratch buffers holding the temporary data of the
	// acceleration structure builder
	VkBufferCreateInfo bufferCreateInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.size = maxScratch;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	std::unique_ptr<Buffer> scratchBuffer = std::make_unique<Buffer>(*device, bufferCreateInfo, memoryInfo);

	VkBufferDeviceAddressInfo bufferDeviceAddressInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	bufferDeviceAddressInfo.buffer = scratchBuffer->getHandle();
	VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

	// Is compaction requested?
	bool doCompaction = (buildAccelerationStructureFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
		== VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

	LOGD("Acceleration structure compaction is {}", doCompaction ? "enabled" : "disabled");

	// Allocate a query pool for storing the needed size for every BLAS compaction.
	VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	qpci.queryCount = blasCount;
	qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
	VkQueryPool queryPool;
	vkCreateQueryPool(device->getHandle(), &qpci, nullptr, &queryPool);
	vkResetQueryPool(device->getHandle(), queryPool, 0, blasCount);

	// Allocate a command pool for queue of given queue index.
	// To avoid timeout, record and submit one command buffer per AS build.
	CommandPool asCommandPool(*device, m_graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	std::vector<std::shared_ptr<CommandBuffer>> allCmdBufs;
	allCmdBufs.reserve(blasCount);

	// Building the acceleration structures
	for (uint32_t idx = 0; idx < blasCount; idx++)
	{
		BlasEntry &blas = m_blas[idx];
		std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		allCmdBufs.push_back(cmdBuf);
		cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

		// All build are using the same scratch buffer
		buildInfos[idx].scratchData.deviceAddress = scratchAddress;

		// Convert user vector of offsets to vector of pointer-to-offset. This defines which (sub)section of the vertex/index arrays will be built into the BLAS.
		std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> pBuildOffset(blas.input.asBuildRangeInfo.size());
		for (size_t infoIdx = 0; infoIdx < blas.input.asBuildRangeInfo.size(); infoIdx++)
			pBuildOffset[infoIdx] = &blas.input.asBuildRangeInfo[infoIdx];

		// Building the AS
		vkCmdBuildAccelerationStructuresKHR(cmdBuf->getHandle(), 1, &buildInfos[idx], pBuildOffset.data());

		// Since the scratch buffer is reused across builds, we need a barrier to ensure one build is finished before starting the next one
		VkMemoryBarrier2 memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
		memoryBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		memoryBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
		memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 1u;
		dependencyInfo.pMemoryBarriers = &memoryBarrier;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = 0u;
		dependencyInfo.pImageMemoryBarriers = nullptr;

		vkCmdPipelineBarrier2KHR(cmdBuf->getHandle(), &dependencyInfo);

		// Write compacted size to query number idx.
		if (doCompaction)
		{
			vkCmdWriteAccelerationStructuresPropertiesKHR(
				cmdBuf->getHandle(), 1, &blas.accelerationStructure->accelerationStuctureKHR,
				VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, idx
			);
		}
	}

	// submit and wait
	std::vector<VkCommandBufferSubmitInfo> commandBufferSubmitInfos;
	for (uint32_t idx = 0; idx < allCmdBufs.size(); idx++)
	{
		allCmdBufs[idx]->end();

		VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
		commandBufferSubmitInfo.pNext = nullptr;
		commandBufferSubmitInfo.commandBuffer = allCmdBufs[idx]->getHandle();
		commandBufferSubmitInfo.deviceMask = 0u;

		commandBufferSubmitInfos.push_back(commandBufferSubmitInfo);
	}

	VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.pNext = nullptr;
	submitInfo.waitSemaphoreInfoCount = 0u;
	submitInfo.pWaitSemaphoreInfos = nullptr;
	submitInfo.commandBufferInfoCount = to_u32(commandBufferSubmitInfos.size());
	submitInfo.pCommandBufferInfos = commandBufferSubmitInfos.data();
	submitInfo.signalSemaphoreInfoCount = 0u;
	submitInfo.pSignalSemaphoreInfos = nullptr;

	VK_CHECK(vkQueueSubmit2KHR(m_graphicsQueue->getHandle(), 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(m_graphicsQueue->getHandle());
	allCmdBufs.clear();

	// Compacting all BLAS
	if (doCompaction)
	{
		std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

		// Get the size result back
		std::vector<VkDeviceSize> compactSizes(blasCount);
		vkGetQueryPoolResults(device->getHandle(), queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
			compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);


		// Compacting
		uint32_t statTotalOriSize{ 0u }, statTotalCompactSize{ 0u };
		for (uint32_t idx = 0; idx < blasCount; idx++)
		{
			// LOGD("Reducing %i, from %d to %d \n", i, originalSizes[i], compactSizes[i]);
			statTotalOriSize += (uint32_t)originalSizes[idx];
			statTotalCompactSize += (uint32_t)compactSizes[idx];

			// Creating a compact version of the AS
			VkAccelerationStructureCreateInfoKHR asCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
			asCreateInfo.size = compactSizes[idx];
			asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
			std::unique_ptr<AccelerationStructure> as = createAccelerationStructure(asCreateInfo);

			// Copy the original BLAS to a compact version
			VkCopyAccelerationStructureInfoKHR copyInfo{ VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
			copyInfo.src = m_blas[idx].accelerationStructure->accelerationStuctureKHR;
			copyInfo.dst = as->accelerationStuctureKHR;
			copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
			vkCmdCopyAccelerationStructureKHR(cmdBuf->getHandle(), &copyInfo);
			// Clean up AS
			vkDestroyAccelerationStructureKHR(device->getHandle(), m_blas[idx].accelerationStructure->accelerationStuctureKHR, nullptr);
			m_blas[idx].accelerationStructure.reset();
			m_blas[idx].accelerationStructure = std::move(as);

			setDebugUtilsObjectName(device->getHandle(), m_blas[idx].accelerationStructure->accelerationStuctureKHR, std::string("BLAS Acceleration Structure #" + std::to_string(idx)));
			setDebugUtilsObjectName(device->getHandle(), m_blas[idx].accelerationStructure->buffer->getHandle(), std::string("BLAS Acceleration Structure Buffer #" + std::to_string(idx)));
		}

		// submitandwaitidle
		cmdBuf->end();

		VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
		commandBufferSubmitInfo.pNext = nullptr;
		commandBufferSubmitInfo.commandBuffer = cmdBuf->getHandle();
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

		LOGD(
			"RT BLAS: reducing from: %u to: %u = %u (%2.2f%s smaller) \n",
			statTotalOriSize,
			statTotalCompactSize,
			statTotalOriSize - statTotalCompactSize,
			(statTotalOriSize - statTotalCompactSize) / float(statTotalOriSize) * 100.f, "%%"
		);
	}

	vkDestroyQueryPool(device->getHandle(), queryPool, nullptr);
}

void VulkrApp::buildTlas(bool update)
{
	// Cannot call buildTlas twice except to update.
	if (m_tlas != nullptr && !update)
	{
		LOGEANDABORT("Cannot call buildTlas twice except to update");
	}

	if (m_tlas == nullptr)
	{
		m_tlas = std::make_unique<Tlas>();
	}

	if (update)
	{
		// Update the acceleration structure instance transformations by obtaining the objectBuffer data since the objInstances data is stale
		void *mappedData = objectBuffer->map();
		ObjInstance *objectSSBO = static_cast<ObjInstance *>(mappedData);
		for (int i = 0; i < objInstances.size(); ++i)
		{
			m_accelerationStructureInstances[i].transform = toTransformMatrixKHR(objectSSBO[i].transform);
		}
		objectBuffer->unmap();
	}
	else
	{
		// First time creation
		m_accelerationStructureInstances.reserve(objInstances.size());
		for (size_t i = 0; i < objInstances.size(); ++i)
		{
			VkAccelerationStructureInstanceKHR accelerationStructureInstance;
			accelerationStructureInstance.transform = toTransformMatrixKHR(objInstances[i].transform);  // Position of the instance
			accelerationStructureInstance.instanceCustomIndex = objInstances[i].objIndex;               // gl_InstanceCustomIndexEXT
			accelerationStructureInstance.mask = 0xFF;
			accelerationStructureInstance.instanceShaderBindingTableRecordOffset = 0;                   // We will use the same hit group for all objects
			accelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			accelerationStructureInstance.accelerationStructureReference = getBlasDeviceAddress(objInstances[i].objIndex);

			m_accelerationStructureInstances.emplace_back(accelerationStructureInstance);
		}
		m_buildAccelerationStructureFlags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	}

	CommandPool asCommandPool(*device, m_graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	m_tlas->flags = m_buildAccelerationStructureFlags;

	// Create a buffer holding the actual instance data for use by the AS builder
	VkDeviceSize instanceDescsSizeInBytes{ m_accelerationStructureInstances.size() * sizeof(VkAccelerationStructureInstanceKHR) };

	// Allocate the instance buffer and copy its contents from host to device memory
	if (update)
	{
		m_instBuffer.reset();
	}

	VkDeviceSize bufferSize{ sizeof(VkAccelerationStructureInstanceKHR) * m_accelerationStructureInstances.size() };
	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	m_instBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, m_accelerationStructureInstances.data(), static_cast<size_t>(bufferSize));
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *m_instBuffer, bufferSize);
	setDebugUtilsObjectName(device->getHandle(), m_instBuffer->getHandle(), "Instance Buffer");
	VkBufferDeviceAddressInfo bufferDeviceAddressInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	bufferDeviceAddressInfo.buffer = m_instBuffer->getHandle();
	VkDeviceAddress instanceAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

	// Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
	VkMemoryBarrier2 memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
	memoryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	memoryBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
	memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

	VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	dependencyInfo.pNext = nullptr;
	dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	dependencyInfo.memoryBarrierCount = 1u;
	dependencyInfo.pMemoryBarriers = &memoryBarrier;
	dependencyInfo.bufferMemoryBarrierCount = 0u;
	dependencyInfo.pBufferMemoryBarriers = nullptr;
	dependencyInfo.imageMemoryBarrierCount = 0u;
	dependencyInfo.pImageMemoryBarriers = nullptr;

	vkCmdPipelineBarrier2KHR(cmdBuf->getHandle(), &dependencyInfo);

	// Create VkAccelerationStructureGeometryInstancesDataKHR; this wraps a device pointer to the above uploaded instances.
	VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
	instancesVk.arrayOfPointers = VK_FALSE;
	instancesVk.data.deviceAddress = instanceAddress;

	// Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
	VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	topASGeometry.geometry.instances = instancesVk;

	// Find sizes
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	buildInfo.flags = m_buildAccelerationStructureFlags;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &topASGeometry;
	buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

	uint32_t instancesCount = to_u32(m_accelerationStructureInstances.size());
	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetAccelerationStructureBuildSizesKHR(device->getHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instancesCount, &sizeInfo);

	// Create TLAS
	if (!update)
	{
		VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationStructureCreateInfo.size = sizeInfo.accelerationStructureSize;

		m_tlas->accelerationStructure = createAccelerationStructure(accelerationStructureCreateInfo);
		setDebugUtilsObjectName(device->getHandle(), m_tlas->accelerationStructure->accelerationStuctureKHR, "TLAS");
	}

	// Allocate the scratch memory
	VkBufferCreateInfo bufferCreateInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.size = sizeInfo.buildScratchSize;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	std::unique_ptr<Buffer> scratchBuffer = std::make_unique<Buffer>(*device, bufferCreateInfo, memoryInfo);
	bufferDeviceAddressInfo.buffer = scratchBuffer->getHandle();
	VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

	// Update build information
	buildInfo.srcAccelerationStructure = update ? m_tlas->accelerationStructure->accelerationStuctureKHR : VK_NULL_HANDLE;
	buildInfo.dstAccelerationStructure = m_tlas->accelerationStructure->accelerationStuctureKHR;
	buildInfo.scratchData.deviceAddress = scratchAddress;

	// Build Offsets info: n instances
	VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{ instancesCount, 0, 0, 0 };
	const VkAccelerationStructureBuildRangeInfoKHR *pBuildOffsetInfo = &buildOffsetInfo;

	// Build the TLAS
	vkCmdBuildAccelerationStructuresKHR(cmdBuf->getHandle(), 1, &buildInfo, &pBuildOffsetInfo);

	// submitandwaitidle
	cmdBuf->end();

	VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	commandBufferSubmitInfo.pNext = nullptr;
	commandBufferSubmitInfo.commandBuffer = cmdBuf->getHandle();
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

VkDeviceAddress VulkrApp::getBlasDeviceAddress(uint64_t blasId)
{
	if (blasId >= objModelsToRender.size()) LOGEANDABORT("Invalid blasId");
	VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
	addressInfo.accelerationStructure = m_blas[blasId].accelerationStructure->accelerationStuctureKHR;
	return vkGetAccelerationStructureDeviceAddressKHR(device->getHandle(), &addressInfo);
}

void VulkrApp::createRaytracingDescriptorPool()
{
	std::vector<VkDescriptorPoolSize> poolSizes{};
	poolSizes.resize(2);
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	poolSizes[0].descriptorCount = 1;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[1].descriptorCount = 1;

	m_rtDescPool = std::make_unique<DescriptorPool>(*device, poolSizes, 2u, 0);
}

void VulkrApp::createRaytracingDescriptorLayout()
{
	VkDescriptorSetLayoutBinding tlasLayoutBinding{};
	tlasLayoutBinding.binding = 0u;
	tlasLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	tlasLayoutBinding.descriptorCount = 1u;
	tlasLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	tlasLayoutBinding.pImmutableSamplers = nullptr; // Optional

	VkDescriptorSetLayoutBinding outputImageLayoutBinding{};
	outputImageLayoutBinding.binding = 1u;
	outputImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	outputImageLayoutBinding.descriptorCount = 1u;
	outputImageLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	outputImageLayoutBinding.pImmutableSamplers = nullptr; // Optional

	std::vector<VkDescriptorSetLayoutBinding> rtDescriptorSetLayoutBindings;
	rtDescriptorSetLayoutBindings.push_back(tlasLayoutBinding);
	rtDescriptorSetLayoutBindings.push_back(outputImageLayoutBinding);

	m_rtDescSetLayout = std::make_unique<DescriptorSetLayout>(*device, rtDescriptorSetLayoutBindings);
}

void VulkrApp::createRaytracingDescriptorSets()
{
	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		VkDescriptorSetAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		allocateInfo.descriptorPool = m_rtDescPool->getHandle();
		allocateInfo.descriptorSetCount = 1u;
		allocateInfo.pSetLayouts = &m_rtDescSetLayout->getHandle();

		raytracingDescriptorSet = std::make_unique<DescriptorSet>(*device, allocateInfo);
		setDebugUtilsObjectName(device->getHandle(), raytracingDescriptorSet->getHandle(), "rtDescriptorSet for frame #" + std::to_string(i));

		VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
		descASInfo.accelerationStructureCount = 1u;
		descASInfo.pAccelerationStructures = &m_tlas->accelerationStructure->accelerationStuctureKHR;

		VkWriteDescriptorSet writeAccelerationStructure{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		writeAccelerationStructure.dstSet = raytracingDescriptorSet->getHandle();
		writeAccelerationStructure.dstBinding = 0u;
		writeAccelerationStructure.dstArrayElement = 0u;
		writeAccelerationStructure.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		writeAccelerationStructure.descriptorCount = 1u;
		writeAccelerationStructure.pNext = &descASInfo;

		VkDescriptorImageInfo outputImageInfo{};
		outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		outputImageInfo.imageView = outputImageView->getHandle();
		outputImageInfo.sampler = {};

		VkWriteDescriptorSet writeOutputImage{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		writeOutputImage.dstSet = raytracingDescriptorSet->getHandle();
		writeOutputImage.dstBinding = 1u;
		writeOutputImage.dstArrayElement = 0u;
		writeOutputImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeOutputImage.descriptorCount = 1u;
		writeOutputImage.pImageInfo = &outputImageInfo;

		std::array<VkWriteDescriptorSet, 2> writeToDescriptorSets{ writeAccelerationStructure, writeOutputImage };
		vkUpdateDescriptorSets(device->getHandle(), to_u32(writeToDescriptorSets.size()), writeToDescriptorSets.data(), 0, nullptr);
	}
}

// Writes the output image to the descriptor set, Required when changing resolution
void VulkrApp::updateRaytracingDescriptorSet()
{
	for (uint32_t i = 0; i < maxFramesInFlight; ++i)
	{
		VkDescriptorImageInfo outputImageInfo{};
		outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		outputImageInfo.imageView = outputImageView->getHandle();
		outputImageInfo.sampler = {};

		VkWriteDescriptorSet writeOutputImage{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		writeOutputImage.dstSet = raytracingDescriptorSet->getHandle();
		writeOutputImage.dstBinding = 1u;
		writeOutputImage.dstArrayElement = 0u;
		writeOutputImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeOutputImage.descriptorCount = 1u;
		writeOutputImage.pImageInfo = &outputImageInfo;
		writeOutputImage.pBufferInfo = nullptr;
		writeOutputImage.pTexelBufferView = nullptr;

		vkUpdateDescriptorSets(device->getHandle(), 1, &writeOutputImage, 0, nullptr);
	}
}

// Pipeline for the ray tracer: all shaders, raygen, chit, miss
void VulkrApp::createRaytracingPipeline()
{
	std::shared_ptr<ShaderSource> rayGenShader = std::make_shared<ShaderSource>("ray_tracing/raytrace.rgen.spv");
	std::shared_ptr<ShaderSource> rayMissShader = std::make_shared<ShaderSource>("ray_tracing/raytrace.rmiss.spv");
	std::shared_ptr<ShaderSource> rayShadowMissShader = std::make_shared<ShaderSource>("ray_tracing/raytraceShadow.rmiss.spv");
	std::shared_ptr<ShaderSource> rayClosestHitShader = std::make_shared<ShaderSource>("ray_tracing/raytrace.rchit.spv");

	struct SpecializationData
	{
		uint32_t maxLightCount;
	} specializationData;
	const VkSpecializationMapEntry entries[] =
	{
		{ 0u, offsetof(SpecializationData, maxLightCount), sizeof(uint32_t) }
	};
	specializationData.maxLightCount = maxLightCount;

	VkSpecializationInfo rayGenSpecializationInfo;
	rayGenSpecializationInfo.mapEntryCount = 0;
	rayGenSpecializationInfo.dataSize = 0;
	VkSpecializationInfo rayMissSpecializationInfo;
	rayMissSpecializationInfo.mapEntryCount = 0;
	rayMissSpecializationInfo.dataSize = 0;
	VkSpecializationInfo rayShadowMissSpecializationInfo;
	rayShadowMissSpecializationInfo.mapEntryCount = 0;
	rayShadowMissSpecializationInfo.dataSize = 0;
	VkSpecializationInfo rayClosestHitSpecializationInfo =
	{
		1u,
		entries,
		to_u32(sizeof(SpecializationData)),
		&specializationData
	};

	std::vector<ShaderModule> raytracingShaderModules;
	raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_RAYGEN_BIT_KHR, rayGenSpecializationInfo, rayGenShader);
	raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_MISS_BIT_KHR, rayMissSpecializationInfo, rayMissShader);
	raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_MISS_BIT_KHR, rayShadowMissSpecializationInfo, rayShadowMissShader);
	raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rayClosestHitSpecializationInfo, rayClosestHitShader);

	// Shader groups
	enum StageIndices
	{
		eRaygen,
		eMiss,
		eMissShadow,
		eClosestHit,
		eShaderGroupCount
	};

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> rtShaderGroups;
	VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	// Raygen
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	rtShaderGroups.push_back(group);

	// Miss
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	rtShaderGroups.push_back(group);

	// Shadow Miss
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMissShadow;
	rtShaderGroups.push_back(group);

	// Closest Hit
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	rtShaderGroups.push_back(group);

	VkPushConstantRange pushConstantRange{ VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 0, sizeof(LightData) };

	std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {
		m_rtDescSetLayout->getHandle(),
		globalDescriptorSetLayout->getHandle(),
		objectDescriptorSetLayout->getHandle(),
		textureDescriptorSetLayout->getHandle()
	};

	std::vector<VkPushConstantRange> pushConstantRangeHandles{ pushConstantRange };
	std::unique_ptr<RayTracingPipelineState> rayTracingPipelineState = std::make_unique<RayTracingPipelineState>(
		std::make_unique<PipelineLayout>(*device, raytracingShaderModules, rtDescSetLayouts, pushConstantRangeHandles), rtShaderGroups
	);
	std::unique_ptr<RayTracingPipeline> rayTracingPipeline = std::make_unique<RayTracingPipeline>(*device, *rayTracingPipelineState, nullptr);

	pipelines.rayTracing.pipelineState = std::move(rayTracingPipelineState);
	pipelines.rayTracing.pipeline = std::move(rayTracingPipeline);
}

// Creating the Shader Binding Table (SBT)
void VulkrApp::createRaytracingShaderBindingTable()
{
	RayTracingPipelineState *rayTracingPipelineState = dynamic_cast<RayTracingPipelineState *>(pipelines.rayTracing.pipelineState.get());
	uint32_t groupCount = to_u32(rayTracingPipelineState->getRayTracingShaderGroups().size());  // 4 shaders: raygen, 2 miss, 1 chit
	uint32_t groupHandleSize = device->getPhysicalDevice().getRayTracingPipelineProperties().shaderGroupHandleSize; // Size of a program identifier
	// Compute the actual size needed per SBT entry (round-up to alignment needed).
	uint32_t groupSizeAligned = align_up(groupHandleSize, device->getPhysicalDevice().getRayTracingPipelineProperties().shaderGroupBaseAlignment);
	// Bytes needed for the SBT.
	VkDeviceSize sbtSize = groupCount * groupSizeAligned;

	// Fetch all the shader handles used in the pipeline. This is opaque data so we store it in a vector of bytes. The order of handles follow the stage entry.
	std::vector<uint8_t> shaderHandleStorage(sbtSize);
	VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device->getHandle(), pipelines.rayTracing.pipeline->getHandle(), 0, groupCount, to_u32(sbtSize), shaderHandleStorage.data()));

	// Allocate a buffer for storing the SBT
	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = sbtSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	m_rtSBTBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
	setDebugUtilsObjectName(device->getHandle(), m_rtSBTBuffer->getHandle(), "SBT Buffer");

	void *mappedData = stagingBuffer->map();
	uint8_t *pData = reinterpret_cast<uint8_t *>(mappedData);
	for (uint32_t g = 0; g < groupCount; g++)
	{
		memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);
		pData += groupSizeAligned;
	}
	stagingBuffer->unmap();
	copyBufferToBuffer(*stagingBuffer, *m_rtSBTBuffer, sbtSize);
}

void VulkrApp::raytrace()
{
	debugUtilBeginLabel(frameData.commandBuffers[currentFrame][0]->getHandle(), "Raytrace");

	std::array<VkDescriptorSet, 4> descSets{
		raytracingDescriptorSet->getHandle(),
		globalDescriptorSet->getHandle(),
		objectDescriptorSet->getHandle(),
		textureDescriptorSet->getHandle()
	};
	vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelines.rayTracing.pipeline->getHandle());
	vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelines.rayTracing.pipelineState->getPipelineLayout().getHandle(), 0,
		to_u32(descSets.size()), descSets.data(), 0, nullptr);
	vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.rayTracing.pipelineState->getPipelineLayout().getHandle(),
		VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
		0, sizeof(RaytracingPushConstant), &raytracingPushConstant);


	// Size of a program identifier
	uint32_t groupSize = align_up(device->getPhysicalDevice().getRayTracingPipelineProperties().shaderGroupHandleSize, device->getPhysicalDevice().getRayTracingPipelineProperties().shaderGroupBaseAlignment);
	uint32_t groupStride = groupSize;

	VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR };
	bufferDeviceAddressInfo.buffer = m_rtSBTBuffer->getHandle();
	VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

	std::array<VkStridedDeviceAddressRegionKHR, 4> strideAddresses{
		VkStridedDeviceAddressRegionKHR{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},  // raygen
		VkStridedDeviceAddressRegionKHR{sbtAddress + 1u * groupSize, groupStride, groupSize * 2},  // miss
		VkStridedDeviceAddressRegionKHR{sbtAddress + 3u * groupSize, groupStride, groupSize * 1},  // hit
		VkStridedDeviceAddressRegionKHR{0u, 0u, 0u}                                                // callable
	};

	vkCmdTraceRaysKHR(frameData.commandBuffers[currentFrame][0]->getHandle(), &strideAddresses[0], &strideAddresses[1], &strideAddresses[2], &strideAddresses[3],
		swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1);

	debugUtilEndLabel(frameData.commandBuffers[currentFrame][0]->getHandle());
}

} // namespace vulkr

int main()
{
	vulkr::Platform platform;
	std::unique_ptr<vulkr::VulkrApp> app = std::make_unique<vulkr::VulkrApp>(platform, "Vulkr");

	platform.initialize(std::move(app));
	platform.prepareApplication();

	platform.runMainProcessingLoop();
	platform.terminate();

	return EXIT_SUCCESS;
}

