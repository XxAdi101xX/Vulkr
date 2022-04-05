/* Copyright (c) 2022 Adithya Venkatarao
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

#include "app.h"
#include "platform/platform.h"

#include <format>

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

MainApp::MainApp(Platform &platform, std::string name) : Application{ platform, name } {}

MainApp::~MainApp()
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

    for (auto &it : objModels)
    {
        it.indexBuffer.reset();
        it.vertexBuffer.reset();
        it.materialsBuffer.reset();
        it.materialsIndexBuffer.reset();
    }
    objModels.clear();
    objInstances.clear();

    for (auto &it : textures)
    {
        it.image.reset();
        it.imageview.reset();
    }

    m_blas.clear();
    m_tlas.reset();
    m_instBuffer.reset();
    m_rtSBTBuffer.reset();

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.outputImages[i].reset();
        frameData.outputImageViews[i].reset();
        frameData.copyOutputImages[i].reset();
        frameData.copyOutputImageViews[i].reset();
        frameData.historyImages[i].reset();
        frameData.historyImageViews[i].reset();
        frameData.velocityImages[i].reset();
        frameData.velocityImageViews[i].reset();
        frameData.commandPools[i].reset();
    }
    imguiPool.reset();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    device.reset();

	if (surface != VK_NULL_HANDLE)
	{
		vkDestroySurfaceKHR(instance->getHandle(), surface, nullptr);
	}

	instance.reset();
}

void MainApp::cleanupSwapchain()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        for (uint32_t j = 0; j < commandBufferCountForFrame; ++j)
        {
            frameData.commandBuffers[i][j].reset();
        }
    }

    depthImage.reset();
    depthImageView.reset();

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.offscreenFramebuffers[i].reset();
        frameData.postProcessFramebuffers[i].reset();
    }

    pipelines.offscreen.pipeline.reset();
    pipelines.offscreen.pipelineState.reset();
    pipelines.postProcess.pipeline.reset();
    pipelines.postProcess.pipelineState.reset();
    pipelines.compute.pipeline.reset();
    pipelines.compute.pipelineState.reset();
    pipelines.computeParticleCalculate.pipeline.reset();
    pipelines.computeParticleCalculate.pipelineState.reset();
    pipelines.computeParticleIntegrate.pipeline.reset();
    pipelines.computeParticleIntegrate.pipelineState.reset();

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

    swapchain.reset();

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        // Descriptor sets
        frameData.globalDescriptorSets[i].reset();
        frameData.objectDescriptorSets[i].reset();
        frameData.postProcessingDescriptorSets[i].reset();
        frameData.taaDescriptorSets[i].reset();
        // Buffers
        frameData.cameraBuffers[i].reset();
        frameData.previousFrameCameraBuffers[i].reset();
        frameData.lightBuffers[i].reset();
        frameData.objectBuffers[i].reset();
        frameData.previousFrameObjectBuffers[i].reset();
        frameData.particleBuffers[i].reset();
    }

    descriptorPool.reset();

#ifndef RENDERDOC_DEBUG
    vkDestroyPipelineLayout(device->getHandle(), m_rtPipelineLayout, nullptr);  // TODO Put this into pipeline layout class
    vkDestroyPipeline(device->getHandle(), m_rtPipeline, nullptr); // TODO Put this into pipeline class
    m_rtDescSetLayout.reset();
    m_rtDescPool.reset();
#endif

    cameraController.reset();
}

void MainApp::prepare()
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
        graphicsQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
        computeQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
        presentQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
        transferQueue = device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
    }
    else
    {
        // TODO: add concurrency with separate transfer queues
        LOGEANDABORT("TODO: Devices where a queue supporting graphics, compute and transfer do not exist are not supported yet");

        // Find a queue that supports graphics and present operations
        int32_t desiredGraphicsQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT, true);
        if (desiredGraphicsQueueFamilyIndex >= 0)
        {
            graphicsQueue = device->getQueue(to_u32(desiredGraphicsQueueFamilyIndex), 0u);
            presentQueue = device->getQueue(to_u32(desiredGraphicsQueueFamilyIndex), 0u);
        }
        else
        {
            int32_t graphicsQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT, false);
            int32_t presentQueueFamilyIndex = device->getQueueFamilyIndexByFlags(0u, true);

            if (graphicsQueueFamilyIndex < 0 || presentQueueFamilyIndex < 0)
            {
                LOGEANDABORT("Unable to find a queue that supports graphics and/or presentation")
            }
            graphicsQueue = device->getQueue(to_u32(graphicsQueueFamilyIndex), 0u);
            presentQueue = device->getQueue(to_u32(presentQueueFamilyIndex), 0u);
        }

        // Find a queue that supports transfer operations
        int32_t desiredTransferQueueFamilyIndex = device->getQueueFamilyIndexByFlags(VK_QUEUE_TRANSFER_BIT, false);
        if (desiredTransferQueueFamilyIndex < 0)
        {
            LOGEANDABORT("Unable to find a queue that supports transfer operations");
            transferQueue = device->getQueue(to_u32(desiredTransferQueueFamilyIndex), 0u);
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
    createComputePipeline();
    createPostProcessingPipeline();
    createTextureSampler();
    createUniformBuffers();
    createSSBOs();
    prepareParticleData();
    createParticleCalculateComputePipeline();
    createParticleIntegrateComputePipeline();
    createDescriptorPool();
    createDescriptorSets();
    setupCamera();
    createScene();
    createSceneLights();

#ifndef RENDERDOC_DEBUG
    createBottomLevelAS();
    buildTlas(false);
    createRtDescriptorPool();
    createRtDescriptorLayout();
    createRtDescriptorSets();
    createRtPipeline();
    createRtShaderBindingTable();
#endif

    createSemaphoreAndFencePools();
    setupSynchronizationObjects();

    initializeBufferData();

    // Ensure that the first tick is with respect to the time after all of the steup
    drawingTimer->tick();
}

void MainApp::update()
{
    fencePool->wait(&frameData.inFlightFences[currentFrame]);
    fencePool->reset(&frameData.inFlightFences[currentFrame]);

    // Now that we are sure that the commands finished executing, we can safely reset the command buffers for the frame to begin recording again
    for (uint32_t i = 0; i < commandBufferCountForFrame; ++i)
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
    animateInstances();
    //updateComputeDescriptorSet(); // TODO: is this even correct, i'm not sure whether we need to update descriptor sets really
    updateBuffersPerFrame();
    drawImGuiInterface();
    updateTaaState(); // must be called after drawingImGui

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
        taaPushConstant.jitter.x = (taaPushConstant.jitter.x / swapchain->getProperties().imageExtent.width) * 5;
        taaPushConstant.jitter.y = (taaPushConstant.jitter.y / swapchain->getProperties().imageExtent.height) * 5;
    }
    else
    {
        taaPushConstant.jitter = glm::vec2(0.0f, 0.0f);
    }

    // Begin command buffer for offscreen pass
    frameData.commandBuffers[currentFrame][0]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    // Compute shader invocations
    computeParticles();
    //animateWithCompute(); // Compute vertices with compute shader TODO: fix the validation errors that happen when this is enabled, might be due to incorrect sytnax with obj buffer
    // in animate.comp or the fact that the objBuffer is readonly? Not totally sure.

    if (raytracingEnabled)
    {
        // Add memory barrier to ensure that the particleIntegrate computer shader has finished writing to the currentFrameObjectBuffer
        VkMemoryBarrier memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            frameData.commandBuffers[currentFrame][0]->getHandle(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        raytrace();
    }
    else
    {
        // Add memory barrier to ensure that the particleIntegrate computer shader has finished writing to the currentFrameObjectBuffer
        VkMemoryBarrier memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            frameData.commandBuffers[currentFrame][0]->getHandle(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        frameData.commandBuffers[currentFrame][0]->beginRenderPass(*mainRenderPass.renderPass, *(frameData.offscreenFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, offscreenFramebufferClearValues, VK_SUBPASS_CONTENTS_INLINE);
        rasterize();
        frameData.commandBuffers[currentFrame][0]->endRenderPass();
    }

    // End command buffer for offscreen pass
    frameData.commandBuffers[currentFrame][0]->end();

    // I have setup a subpass dependency to ensure that the render pass waits for the swapchain to finish reading from the image before accessing it
    // hence I don't need to set the wait stages to VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT 
    std::array<VkPipelineStageFlags, 1> offscreenWaitStages{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    std::array<VkSemaphore, 1> offscreenWaitSemaphores{ frameData.imageAvailableSemaphores[currentFrame] };
    std::array<VkSemaphore, 1> offscreenSignalSemaphores{ frameData.offscreenRenderingFinishedSemaphores[currentFrame] };

    VkSubmitInfo offscreenPassSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    offscreenPassSubmitInfo.waitSemaphoreCount = to_u32(offscreenWaitSemaphores.size());
    offscreenPassSubmitInfo.pWaitSemaphores = offscreenWaitSemaphores.data();
    offscreenPassSubmitInfo.pWaitDstStageMask = offscreenWaitStages.data();
    offscreenPassSubmitInfo.commandBufferCount = 1;
    offscreenPassSubmitInfo.pCommandBuffers = &frameData.commandBuffers[currentFrame][0]->getHandle();
    offscreenPassSubmitInfo.signalSemaphoreCount = to_u32(offscreenSignalSemaphores.size());
    offscreenPassSubmitInfo.pSignalSemaphores = offscreenSignalSemaphores.data();

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
    // hence I don't need to set the wait stages to VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT 
    std::array<VkPipelineStageFlags, 1> postProcessWaitStages{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    std::array<VkSemaphore, 1> postProcessWaitSemaphores{ frameData.offscreenRenderingFinishedSemaphores[currentFrame] };
    std::array<VkSemaphore, 1> postProcessSignalSemaphores{ frameData.postProcessRenderingFinishedSemaphores[currentFrame] };

    VkSubmitInfo postProcessSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    postProcessSubmitInfo.waitSemaphoreCount = to_u32(postProcessWaitSemaphores.size());
    postProcessSubmitInfo.pWaitSemaphores = postProcessWaitSemaphores.data();
    postProcessSubmitInfo.pWaitDstStageMask = postProcessWaitStages.data();
    postProcessSubmitInfo.commandBufferCount = 1;
    postProcessSubmitInfo.pCommandBuffers = &frameData.commandBuffers[currentFrame][1]->getHandle();
    postProcessSubmitInfo.signalSemaphoreCount = to_u32(postProcessSignalSemaphores.size());
    postProcessSubmitInfo.pSignalSemaphores = postProcessSignalSemaphores.data();

    // Begin command buffer for outputImage copy operations to swapchain and history buffer
    frameData.commandBuffers[currentFrame][2]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    // Prepare the current swapchain image and history buffer as transfer destinations
    swapchain->getImages()[swapchainImageIndex]->transitionImageLayout(*frameData.commandBuffers[currentFrame][2], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
    frameData.historyImages[currentFrame]->transitionImageLayout(*frameData.commandBuffers[currentFrame][2], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
    // Note that the layout of the outputImage has been transitioned from VK_IMAGE_LAYOUT_GENERAL to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL as defined in the postProcessingRenderPass configuration

    VkImageCopy outputImageCopyRegion{};
    outputImageCopyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    outputImageCopyRegion.srcOffset = { 0, 0, 0 };
    outputImageCopyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    outputImageCopyRegion.dstOffset = { 0, 0, 0 };
    outputImageCopyRegion.extent = { swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1 };

    // Copy output image to swapchain image and history image (note that they can execute in any order due to lack of barriers)
    vkCmdCopyImage(frameData.commandBuffers[currentFrame][2]->getHandle(), frameData.outputImages[currentFrame]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain->getImages()[swapchainImageIndex]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &outputImageCopyRegion);
    vkCmdCopyImage(frameData.commandBuffers[currentFrame][2]->getHandle(), frameData.outputImages[currentFrame]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frameData.historyImages[currentFrame]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &outputImageCopyRegion);

    // TODO create staging buffer before conducting the two cmdCopyBuffer, see copyBufferToBuffer method as well
    VkBufferCopy cameraBufferCopyRegion{};
    cameraBufferCopyRegion.srcOffset = 0;
    cameraBufferCopyRegion.dstOffset = 0;
    cameraBufferCopyRegion.size = sizeof(CameraData);
    vkCmdCopyBuffer(frameData.commandBuffers[currentFrame][2]->getHandle(), frameData.cameraBuffers[currentFrame]->getHandle(), frameData.previousFrameCameraBuffers[currentFrame]->getHandle(), 1, &cameraBufferCopyRegion);

    VkBufferCopy objectBufferCopyRegion{};
    objectBufferCopyRegion.srcOffset = 0;
    objectBufferCopyRegion.dstOffset = 0;
    objectBufferCopyRegion.size = sizeof(ObjInstance) * maxInstanceCount;
    vkCmdCopyBuffer(frameData.commandBuffers[currentFrame][2]->getHandle(), frameData.objectBuffers[currentFrame]->getHandle(), frameData.previousFrameObjectBuffers[currentFrame]->getHandle(), 1, &objectBufferCopyRegion);

    // Transition the current swapchain image back for presentation
    swapchain->getImages()[swapchainImageIndex]->transitionImageLayout(*frameData.commandBuffers[currentFrame][2], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, subresourceRange);
    // Transition the history image for general purpose operations
    frameData.historyImages[currentFrame]->transitionImageLayout(*frameData.commandBuffers[currentFrame][2], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
    // Transition the output image back to the general layout
    frameData.outputImages[currentFrame]->transitionImageLayout(*frameData.commandBuffers[currentFrame][2], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);

    // End command buffer for copy operations
    frameData.commandBuffers[currentFrame][2]->end();

    std::array<VkPipelineStageFlags, 1> outputImageTransferWaitStages{ VK_PIPELINE_STAGE_TRANSFER_BIT };
    std::array<VkSemaphore, 1> outputImageTransferWaitSemaphores{ frameData.postProcessRenderingFinishedSemaphores[currentFrame] };
    std::array<VkSemaphore, 1> outputImageTransferSignalSemaphores{ frameData.outputImageCopyFinishedSemaphores[currentFrame] };

    VkSubmitInfo outputImageTransferSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    outputImageTransferSubmitInfo.waitSemaphoreCount = to_u32(outputImageTransferWaitSemaphores.size());
    outputImageTransferSubmitInfo.pWaitSemaphores = outputImageTransferWaitSemaphores.data();
    outputImageTransferSubmitInfo.pWaitDstStageMask = outputImageTransferWaitStages.data();
    outputImageTransferSubmitInfo.commandBufferCount = 1;
    outputImageTransferSubmitInfo.pCommandBuffers = &frameData.commandBuffers[currentFrame][2]->getHandle();
    outputImageTransferSubmitInfo.signalSemaphoreCount = to_u32(outputImageTransferSignalSemaphores.size());
    outputImageTransferSubmitInfo.pSignalSemaphores = outputImageTransferSignalSemaphores.data();

    std::array<VkSubmitInfo, 3> submitInfo{ offscreenPassSubmitInfo, postProcessSubmitInfo, outputImageTransferSubmitInfo };
    VK_CHECK(vkQueueSubmit(graphicsQueue->getHandle(), submitInfo.size(), submitInfo.data(), frameData.inFlightFences[currentFrame]));

    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = to_u32(outputImageTransferSignalSemaphores.size());
    presentInfo.pWaitSemaphores = outputImageTransferSignalSemaphores.data();

    std::array<VkSwapchainKHR, 1> swapchains{ swapchain->getHandle() };
    presentInfo.swapchainCount = to_u32(swapchains.size());
    presentInfo.pSwapchains = swapchains.data();

    presentInfo.pImageIndices = &swapchainImageIndex;

    result = vkQueuePresentKHR(presentQueue->getHandle(), &presentInfo);
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

// TODO: handle recreate for raytracing
void MainApp::recreateSwapchain()
{
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
    createComputePipeline();
    createDepthResources();
    createFramebuffers();
    createCommandBuffers();
    createUniformBuffers();
    createSSBOs();
    createDescriptorPool();
    createDescriptorSets();
#ifndef RENDERDOC_DEBUG
    updateRtDescriptorSet();
#endif
    createScene();

    imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);
}

void MainApp::handleInputEvents(const InputEvent &inputEvent)
{
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    cameraController->handleInputEvents(inputEvent);
}

/* Private methods start here */

void MainApp::drawImGuiInterface()
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
                    changed |= ImGui::RadioButton(oss.str().c_str(), &sceneLights[lightIndex].lightType, 0);
                    ImGui::SameLine();
                    oss.str(""); oss << "Infinite" << "##" << lightIndex;
                    changed |= ImGui::RadioButton(oss.str().c_str(), &sceneLights[lightIndex].lightType, 1);

                    oss.str(""); oss << "Position" << "##" << lightIndex;
                    changed |= ImGui::SliderFloat3(oss.str().c_str(), &(sceneLights[lightIndex].lightPosition.x), -50.f, 50.f);
                    oss.str(""); oss << "Intensity" << "##" << lightIndex;
                    changed |= ImGui::SliderFloat(oss.str().c_str(), &sceneLights[lightIndex].lightIntensity, 0.f, 250.f);
                }
            }

            if (changed)
            {
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

void MainApp::animateInstances()
{
    const uint64_t startingIndex{ 2 };
    const int64_t wusonInstanceCount{
        std::count_if(objInstances.begin(), objInstances.end(), [this](ObjInstance i) { return i.objIndex == getObjModelIndex("wuson.obj"); })
    };
    if (wusonInstanceCount == 0)
    {
        LOGW("No wuson instances found");
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
    void *mappedData = frameData.objectBuffers[currentFrame]->map();
    ObjInstance *objectSSBO = static_cast<ObjInstance *>(mappedData);
    for (int i = startingIndex; i < startingIndex + wusonInstanceCount; i++)
    {
        objectSSBO[i].transform = objInstances[i].transform;
        objectSSBO[i].transformIT = objInstances[i].transformIT;
    }
    frameData.objectBuffers[currentFrame]->unmap();
}

void MainApp::animateWithCompute()
{
    const uint64_t wusonModelIndex{ getObjModelIndex("wuson.obj") };

    computePushConstant.indexCount = objModels[wusonModelIndex].indicesCount;
    computePushConstant.time = std::chrono::duration<float, std::chrono::seconds::period>(drawingTimer->elapsed()).count();

    PipelineData &pipelineData = pipelines.compute;
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipeline->getHandle());
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0, 1, &frameData.objectDescriptorSets[currentFrame]->getHandle(), 0, nullptr);
    vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstant), &computePushConstant);
    vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), objModels[wusonModelIndex].indicesCount / workGroupSize, 1u, 1u);
}

void MainApp::computeParticles()
{
    // Acquire
    if (graphicsQueue->getFamilyIndex() != computeQueue->getFamilyIndex())
    {
        VkBufferMemoryBarrier bufferBarrier =
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            graphicsQueue->getFamilyIndex(),
            computeQueue->getFamilyIndex(),
            frameData.particleBuffers[currentFrame]->getHandle(),
            0,
            particleBufferSize
        };

        vkCmdPipelineBarrier(
            frameData.commandBuffers[currentFrame][0]->getHandle(),
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, &bufferBarrier,
            0, nullptr
        );
    }

    // First pass: Calculate particle movement
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleCalculate.pipeline->getBindPoint(), pipelines.computeParticleCalculate.pipeline->getHandle());
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleCalculate.pipeline->getBindPoint(), pipelines.computeParticleCalculate.pipelineState->getPipelineLayout().getHandle(), 0, 1, &frameData.particleComputeDescriptorSets[currentFrame]->getHandle(), 0, nullptr);
    vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleCalculate.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeParticlesPushConstant), &computeParticlesPushConstant);
    vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), computeParticlesPushConstant.particleCount / workGroupSize, 1, 1);

    // Add memory barrier to ensure that the computer shader has finished writing to the buffer
    VkMemoryBarrier memoryBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        frameData.commandBuffers[currentFrame][0]->getHandle(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    // Second pass: Integrate particles
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipeline->getBindPoint(), pipelines.computeParticleIntegrate.pipeline->getHandle());
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipeline->getBindPoint(), pipelines.computeParticleIntegrate.pipelineState->getPipelineLayout().getHandle(), 0, 1, &frameData.particleComputeDescriptorSets[currentFrame]->getHandle(), 0, nullptr);
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipeline->getBindPoint(), pipelines.computeParticleIntegrate.pipelineState->getPipelineLayout().getHandle(), 1, 1, &frameData.objectDescriptorSets[currentFrame]->getHandle(), 0, nullptr);
    vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelines.computeParticleIntegrate.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeParticlesPushConstant), &computeParticlesPushConstant);
    vkCmdDispatch(frameData.commandBuffers[currentFrame][0]->getHandle(), computeParticlesPushConstant.particleCount / workGroupSize, 1, 1);

    // Release
    if (graphicsQueue->getFamilyIndex() != computeQueue->getFamilyIndex())
    {
        VkBufferMemoryBarrier bufferBarrier =
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            0,
            computeQueue->getFamilyIndex(),
            graphicsQueue->getFamilyIndex(),
            frameData.particleBuffers[currentFrame]->getHandle(),
            0,
            particleBufferSize
        };

        vkCmdPipelineBarrier(
            frameData.commandBuffers[currentFrame][0]->getHandle(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0,
            0, nullptr,
            1, &bufferBarrier,
            0, nullptr
        );
    }
}

void MainApp::initializeBufferData()
{
    for (uint32_t frame = 0; frame < maxFramesInFlight; ++frame)
    {
        // Update the camera buffer
        CameraData cameraData{};
        cameraData.view = cameraController->getCamera()->getView();
        cameraData.proj = cameraController->getCamera()->getProjection();

        void *mappedData = frameData.cameraBuffers[frame]->map();
        memcpy(mappedData, &cameraData, sizeof(cameraData));
        frameData.cameraBuffers[frame]->unmap();

        // Update the light buffer
        mappedData = frameData.lightBuffers[frame]->map();
        memcpy(mappedData, sceneLights.data(), sizeof(LightData) * sceneLights.size());
        frameData.lightBuffers[frame]->unmap();

        // Update the object buffer
        mappedData = frameData.objectBuffers[frame]->map();
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
        frameData.objectBuffers[frame]->unmap();
    }
}

void MainApp::updateBuffersPerFrame()
{
    // TODO should we use staging buffers instead and replace all instances of VMA_MEMORY_USAGE_CPU_TO_GPU to VMA_MEMORY_USAGE_GPU_ONLY?

    // Update the camera buffer
    CameraData cameraData{};
    cameraData.view = cameraController->getCamera()->getView();
    cameraData.proj = cameraController->getCamera()->getProjection();

    void *mappedData = frameData.cameraBuffers[currentFrame]->map();
    memcpy(mappedData, &cameraData, sizeof(cameraData));
    frameData.cameraBuffers[currentFrame]->unmap();

    // Update the light buffer
    mappedData = frameData.lightBuffers[currentFrame]->map();
    memcpy(mappedData, sceneLights.data(), sizeof(LightData) * sceneLights.size());
    frameData.lightBuffers[currentFrame]->unmap();
}

// TODO: as opposed to doing slot based binding of descriptor sets which leads to multiple vkCmdBindDescriptorSets calls per drawcall, you can use
// frequency based descriptor sets and use dynamicOffsetCount: see https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/, or just bindless
// decriptors altogether
void MainApp::rasterize()
{
    debugUtilBeginLabel(frameData.commandBuffers[currentFrame][0]->getHandle(), "Rasterize");

    PipelineData &pipelineData = pipelines.offscreen;

    // Bind the pipeline
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipeline->getHandle());

    // Global data descriptor
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0, 1, &frameData.globalDescriptorSets[currentFrame]->getHandle(), 0, nullptr);

    // Object data descriptor
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 1, 1, &frameData.objectDescriptorSets[currentFrame]->getHandle(), 0, nullptr);

    // Texture descriptor
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 2, 1, &textureDescriptorSet->getHandle(), 0, nullptr);

    // Taa data descriptor
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 3, 1, &frameData.taaDescriptorSets[currentFrame]->getHandle(), 0, nullptr);

    // Push constants
    vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(TaaPushConstant), &taaPushConstant);
    vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), pipelineData.pipelineState->getPipelineLayout().getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(TaaPushConstant), sizeof(RasterizationPushConstant), &rasterizationPushConstant);

    // Bind vertices, indices and call the draw method
    int32_t lastObjIndex{ -1 };
    for (int index = 0; index < objInstances.size(); index++)
    {
        ObjModel &objModel = objModels[objInstances[index].objIndex];
        // Bind the vertex and index buffers if the instance model is different from the previous one
        if (objInstances[index].objIndex != lastObjIndex)
        {
            VkBuffer vertexBuffers[] = { objModel.vertexBuffer->getHandle() };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(frameData.commandBuffers[currentFrame][0]->getHandle(), 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(frameData.commandBuffers[currentFrame][0]->getHandle(), objModel.indexBuffer->getHandle(), 0, VK_INDEX_TYPE_UINT32);

            lastObjIndex = objInstances[index].objIndex;
        }

        vkCmdDrawIndexed(frameData.commandBuffers[currentFrame][0]->getHandle(), to_u32(objModel.indicesCount), 1, 0, 0, index);
    }
    
    debugUtilEndLabel(frameData.commandBuffers[currentFrame][0]->getHandle());
}

void MainApp::postProcess()
{
    debugUtilBeginLabel(frameData.commandBuffers[currentFrame][1]->getHandle(), "Post Process");

    PipelineData &pipelineData = pipelines.postProcess;

    // Bind the pipeline
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame][1]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipeline->getHandle());

    // Post processing descriptor
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][1]->getHandle(), pipelineData.pipeline->getBindPoint(), pipelineData.pipelineState->getPipelineLayout().getHandle(), 0, 1, &frameData.postProcessingDescriptorSets[currentFrame]->getHandle(), 0, nullptr);
    
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
    int32_t lastObjIndex{ -1 };
    for (int index = 0; index < objInstances.size(); index++)
    {
        ObjModel &objModel = objModels[objInstances[index].objIndex];
        // Bind the objModel if it's a different one from last one
        if (objInstances[index].objIndex != lastObjIndex)
        {
            VkBuffer vertexBuffers[] = { objModel.vertexBuffer->getHandle() };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(frameData.commandBuffers[currentFrame][1]->getHandle(), 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(frameData.commandBuffers[currentFrame][1]->getHandle(), objModel.indexBuffer->getHandle(), 0, VK_INDEX_TYPE_UINT32);

            lastObjIndex = objInstances[index].objIndex;
        }

        vkCmdDrawIndexed(frameData.commandBuffers[currentFrame][1]->getHandle(), to_u32(objModel.indicesCount), 1, 0, 0, index);
    }

    debugUtilEndLabel(frameData.commandBuffers[currentFrame][1]->getHandle());
}

void MainApp::setupTimer()
{
    drawingTimer = std::make_unique<Timer>();
    drawingTimer->start();
}

float createHaltonSequence(uint32_t index, uint32_t base)
{
    float f = 1;
    float r = 0;
    int current = index;
    do
    {
        f = f / base;
        r = r + f * (current % base);
        current = glm::floor(current / base);
    } while (current > 0);
    return r;
}

void MainApp::initializeHaltonSequenceArray()
{
    for (int i = 0; i < taaDepth; ++i)
    {
        haltonSequence[i] = glm::vec2(createHaltonSequence(i + 1, 2), createHaltonSequence(i + 1, 3));
    }
}

void MainApp::createInstance()
{
    instance = std::make_unique<Instance>(getName());
    g_instance = instance->getHandle();
}

void MainApp::createSurface()
{
    platform.createSurface(instance->getHandle());
    surface = platform.getSurface();
}

void MainApp::createDevice()
{
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.geometryShader = VK_TRUE;
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.shaderInt64 = VK_TRUE;

    std::unique_ptr<PhysicalDevice> physicalDevice = instance->getSuitablePhysicalDevice();
    physicalDevice->setRequestedFeatures(deviceFeatures);

    workGroupSize = std::min(64u, physicalDevice->getProperties().limits.maxComputeWorkGroupSize[0]);
    shadedMemorySize = std::min(1024u, physicalDevice->getProperties().limits.maxComputeSharedMemorySize);

    device = std::make_unique<Device>(std::move(physicalDevice), surface, deviceExtensions);
}

void MainApp::createSwapchain()
{
    const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT };
    swapchain = std::make_unique<Swapchain>(*device, surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags, graphicsQueue->getFamilyIndex(), presentQueue->getFamilyIndex());

    postProcessPushConstant.imageExtent = glm::vec2(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height);
}

void MainApp::setupCamera()
{
    cameraController = std::make_unique<CameraController>(swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height);
    cameraController->getCamera()->setPerspectiveProjection(45.0f, swapchain->getProperties().imageExtent.width / (float)swapchain->getProperties().imageExtent.height, 0.1f, 100.0f);
    cameraController->getCamera()->setView(glm::vec3(-3.5f, 12.5f, 2.0f), glm::vec3(-1.0f, 12.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

void MainApp::createMainRenderPass()
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

    VkAttachmentReference outputImageAttachmentRef{};
    outputImageAttachmentRef.attachment = 0;
    outputImageAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    VkAttachmentReference copyOutputImageAttachmentRef{};
    copyOutputImageAttachmentRef.attachment = 1;
    copyOutputImageAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    VkAttachmentReference velocityImageAttachmentRef{};
    velocityImageAttachmentRef.attachment = 2;
    velocityImageAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 3;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    mainRenderPass.depthStencilAttachments.push_back(depthAttachmentRef);

    mainRenderPass.subpasses.emplace_back(
        mainRenderPass.inputAttachments,
        mainRenderPass.colorAttachments,
        mainRenderPass.resolveAttachments,
        mainRenderPass.depthStencilAttachments,
        mainRenderPass.preserveAttachments,
        VK_PIPELINE_BIND_POINT_GRAPHICS
    );

    std::vector<VkSubpassDependency> dependencies;
    dependencies.resize(2);

    // TODO: verify these subpass dependencies are correct
    // Only need a dependency coming in to ensure that the first layout transition happens at the right time.
    // Second external dependency is implied by having a different finalLayout and subpass layout.
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0; // References the subpass index in the subpasses array
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0; // We don't have anything that we need to flush
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
    // but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
    // which injects an external dependency here with dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. 

    mainRenderPass.renderPass = std::make_unique<RenderPass>(*device, attachments, mainRenderPass.subpasses, dependencies);
    setDebugUtilsObjectName(device->getHandle(), mainRenderPass.renderPass->getHandle(), "mainRenderPass");
}

void MainApp::createPostRenderPass()
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

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
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

    std::vector<VkSubpassDependency> dependencies;
    dependencies.resize(2);

    // TODO: verify these subpass dependencies are correct
    // Only need a dependency coming in to ensure that the first layout transition happens at the right time.
    // Second external dependency is implied by having a different finalLayout and subpass layout.
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0; // References the subpass index in the subpasses array
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0; // We don't have anything that we need to flush
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
    // but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
    // which injects an external dependency here with dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. 

    postRenderPass.renderPass = std::make_unique<RenderPass>(*device, attachments, postRenderPass.subpasses, dependencies);
    setDebugUtilsObjectName(device->getHandle(), postRenderPass.renderPass->getHandle(), "postProcessRenderPass");
}


void MainApp::createDescriptorSetLayouts()
{
    // Global descriptor set layout
    VkDescriptorSetLayoutBinding cameraBufferLayoutBinding{};
    cameraBufferLayoutBinding.binding = 0;
    cameraBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cameraBufferLayoutBinding.descriptorCount = 1;
    cameraBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; // TODO: not used in fragment bit, closest hit shader
    cameraBufferLayoutBinding.pImmutableSamplers = nullptr; // Optional
    VkDescriptorSetLayoutBinding lightBufferLayoutBinding{};
    lightBufferLayoutBinding.binding = 1;
    lightBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightBufferLayoutBinding.descriptorCount = 1;
    lightBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; // TODO not being used in vertex bit
    lightBufferLayoutBinding.pImmutableSamplers = nullptr; // Optional

    std::vector<VkDescriptorSetLayoutBinding> globalDescriptorSetLayoutBindings{ cameraBufferLayoutBinding, lightBufferLayoutBinding };
    globalDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, globalDescriptorSetLayoutBindings);

    // Object descriptor set layout
    VkDescriptorSetLayoutBinding objectBufferLayoutBinding{};
    objectBufferLayoutBinding.binding = 0;
    objectBufferLayoutBinding.descriptorCount = 1;
    objectBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    objectBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
    objectBufferLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> objectDescriptorSetLayoutBindings{ objectBufferLayoutBinding };
    objectDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, objectDescriptorSetLayoutBindings);

    // Post processing descriptor set layout
    VkDescriptorSetLayoutBinding currentFrameCameraBufferLayoutBindingForPost{};
    currentFrameCameraBufferLayoutBindingForPost.binding = 0;
    currentFrameCameraBufferLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    currentFrameCameraBufferLayoutBindingForPost.descriptorCount = 1;
    currentFrameCameraBufferLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    currentFrameCameraBufferLayoutBindingForPost.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutBinding currentFrameObjectBufferLayoutBindingForPost{};
    currentFrameObjectBufferLayoutBindingForPost.binding = 1;
    currentFrameObjectBufferLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    currentFrameObjectBufferLayoutBindingForPost.descriptorCount = 1;
    currentFrameObjectBufferLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    currentFrameObjectBufferLayoutBindingForPost.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutBinding historyImageLayoutBindingForPost{};
    historyImageLayoutBindingForPost.binding = 2;
    historyImageLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    historyImageLayoutBindingForPost.descriptorCount = 1;
    historyImageLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    historyImageLayoutBindingForPost.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutBinding velocityImageLayoutBindingForPost{};
    velocityImageLayoutBindingForPost.binding = 3;
    velocityImageLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    velocityImageLayoutBindingForPost.descriptorCount = 1;
    velocityImageLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    velocityImageLayoutBindingForPost.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutBinding copyOutputImageLayoutBindingForPost{};
    copyOutputImageLayoutBindingForPost.binding = 4;
    copyOutputImageLayoutBindingForPost.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    copyOutputImageLayoutBindingForPost.descriptorCount = 1;
    copyOutputImageLayoutBindingForPost.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    copyOutputImageLayoutBindingForPost.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> postProcessingDescriptorSetLayoutBindings {
        currentFrameCameraBufferLayoutBindingForPost,
        currentFrameObjectBufferLayoutBindingForPost,
        historyImageLayoutBindingForPost,
        velocityImageLayoutBindingForPost,
        copyOutputImageLayoutBindingForPost
    };
    postProcessingDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, postProcessingDescriptorSetLayoutBindings);

    // Taa descriptor set layout
    VkDescriptorSetLayoutBinding previousFrameCameraBufferLayoutBindingForTaa{};
    previousFrameCameraBufferLayoutBindingForTaa.binding = 0;
    previousFrameCameraBufferLayoutBindingForTaa.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    previousFrameCameraBufferLayoutBindingForTaa.descriptorCount = 1;
    previousFrameCameraBufferLayoutBindingForTaa.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    previousFrameCameraBufferLayoutBindingForTaa.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutBinding previousFrameObjectBufferLayoutBindingForTaa{};
    previousFrameObjectBufferLayoutBindingForTaa.binding = 1;
    previousFrameObjectBufferLayoutBindingForTaa.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    previousFrameObjectBufferLayoutBindingForTaa.descriptorCount = 1;
    previousFrameObjectBufferLayoutBindingForTaa.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    previousFrameObjectBufferLayoutBindingForTaa.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> taaDescriptorSetLayoutBindings {
        previousFrameCameraBufferLayoutBindingForTaa,
        previousFrameObjectBufferLayoutBindingForTaa
    };
    taaDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, taaDescriptorSetLayoutBindings);

    // Texture descriptor set layout
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = to_u32(textures.size());
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> textureDescriptorSetLayoutBindings{ samplerLayoutBinding };
    textureDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, textureDescriptorSetLayoutBindings);

    // Particle compute descriptor set layout
    VkDescriptorSetLayoutBinding particleBufferLayoutBinding{};
    particleBufferLayoutBinding.binding = 0;
    particleBufferLayoutBinding.descriptorCount = 1;
    particleBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    particleBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    particleBufferLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> particleComputeDescriptorSetLayoutBindings{ particleBufferLayoutBinding };
    particleComputeDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, particleComputeDescriptorSetLayoutBindings);
}

void MainApp::createMainRasterizationPipeline()
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
    positionAttributeDescription.binding = 0;
    positionAttributeDescription.location = 0;
    positionAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    positionAttributeDescription.offset = offsetof(VertexObj, position);
    // Normal at location 1
    VkVertexInputAttributeDescription normalAttributeDescription;
    normalAttributeDescription.binding = 0;
    normalAttributeDescription.location = 1;
    normalAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    normalAttributeDescription.offset = offsetof(VertexObj, normal);
    // Color at location 2
    VkVertexInputAttributeDescription colorAttributeDescription;
    colorAttributeDescription.binding = 0;
    colorAttributeDescription.location = 2;
    colorAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttributeDescription.offset = offsetof(VertexObj, color);
    // TexCoord at location 3
    VkVertexInputAttributeDescription textureCoordinateAttributeDescription;
    textureCoordinateAttributeDescription.binding = 0;
    textureCoordinateAttributeDescription.location = 3;
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
    VkViewport viewport{ 0.0f, 0.0f, swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 0.0f, 1.0f };
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

    std::shared_ptr<ShaderSource> mainVertexShader = std::make_shared<ShaderSource>("main.vert.spv");
    VkSpecializationInfo mainVertexShaderSpecializationInfo;
    mainVertexShaderSpecializationInfo.mapEntryCount = 0;
    mainVertexShaderSpecializationInfo.dataSize = 0;

    std::shared_ptr<ShaderSource> mainFragmentShader = std::make_shared<ShaderSource>("main.frag.spv");
    struct SpecializationData {
        uint32_t maxLightCount;
    } specializationData;
    const VkSpecializationMapEntry entries[] =
    {
        { 0u, offsetof(SpecializationData, maxLightCount), sizeof(uint32_t) }
    };
    specializationData.maxLightCount = maxLightCount;

    VkSpecializationInfo mainFragmentShaderSpecializationInfo =
    {
        1u,
        entries,
        to_u32(sizeof(SpecializationData)),
        &specializationData
    };

    std::vector<ShaderModule> shaderModules;
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, mainVertexShaderSpecializationInfo, mainVertexShader);
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, mainFragmentShaderSpecializationInfo, mainFragmentShader);

    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles {
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

    std::shared_ptr<GraphicsPipelineState> mainRasterizationPipelineState = std::make_shared<GraphicsPipelineState>(
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
    std::shared_ptr<GraphicsPipeline> mainRasterizationPipeline = std::make_shared<GraphicsPipeline>(*device, *mainRasterizationPipelineState, nullptr);

    pipelines.offscreen.pipelineState = mainRasterizationPipelineState;
    pipelines.offscreen.pipeline = mainRasterizationPipeline;
}

void MainApp::createPostProcessingPipeline()
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
    positionAttributeDescription.binding = 0;
    positionAttributeDescription.location = 0;
    positionAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    positionAttributeDescription.offset = offsetof(VertexObj, position);
    // Normal at location 1
    VkVertexInputAttributeDescription normalAttributeDescription;
    normalAttributeDescription.binding = 0;
    normalAttributeDescription.location = 1;
    normalAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    normalAttributeDescription.offset = offsetof(VertexObj, normal);
    // Color at location 2
    VkVertexInputAttributeDescription colorAttributeDescription;
    colorAttributeDescription.binding = 0;
    colorAttributeDescription.location = 2;
    colorAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttributeDescription.offset = offsetof(VertexObj, color);
    // TexCoord at location 3
    VkVertexInputAttributeDescription textureCoordinateAttributeDescription;
    textureCoordinateAttributeDescription.binding = 0;
    textureCoordinateAttributeDescription.location = 3;
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
    VkViewport viewport{ 0.0f, 0.0f, swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 0.0f, 1.0f };
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

    std::shared_ptr<ShaderSource> postProcessVertexShader = std::make_shared<ShaderSource>("postProcess.vert.spv");
    VkSpecializationInfo postProcessVertexShaderSpecializationInfo;
    postProcessVertexShaderSpecializationInfo.mapEntryCount = 0;
    postProcessVertexShaderSpecializationInfo.dataSize = 0;
    std::shared_ptr<ShaderSource> postProcessFragmentShader = std::make_shared<ShaderSource>("postProcess.frag.spv");
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

    std::shared_ptr<GraphicsPipelineState> postProcessingPipelineState = std::make_shared<GraphicsPipelineState>(
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
    std::shared_ptr<GraphicsPipeline> postProcessingPipeline = std::make_shared<GraphicsPipeline>(*device, *postProcessingPipelineState, nullptr);

    pipelines.postProcess.pipelineState = postProcessingPipelineState;
    pipelines.postProcess.pipeline = postProcessingPipeline;
}

void MainApp::createComputePipeline()
{
    std::shared_ptr<ShaderSource> animationComputeShader = std::make_shared<ShaderSource>("animate.comp.spv");

    const VkSpecializationMapEntry entries[] =
    {
        {
            0u,
            to_u32(0 * sizeof(uint32_t)),
            sizeof(uint32_t)
        }
    };
    const uint32_t data[] = { workGroupSize };

    VkSpecializationInfo specializationInfo =
    {
        1u,
        entries,
        to_u32(1 * sizeof(uint32_t)),
        data
    };

    std::vector<ShaderModule> shaderModules;
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, animationComputeShader);

    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ objectDescriptorSetLayout->getHandle() };

    std::vector<VkPushConstantRange> pushConstantRangeHandles;
    VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstant) };
    pushConstantRangeHandles.push_back(computePushConstantRange);

    std::shared_ptr<ComputePipelineState> animationComputePipelineState = std::make_shared<ComputePipelineState>(
        std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
    );
    std::shared_ptr<ComputePipeline> animationComputePipeline = std::make_shared<ComputePipeline>(*device, *animationComputePipelineState, nullptr);

    pipelines.compute.pipelineState = animationComputePipelineState;
    pipelines.compute.pipeline = animationComputePipeline;
}

void MainApp::createParticleCalculateComputePipeline()
{
    std::shared_ptr<ShaderSource> particleCalculateComputeShader = std::make_shared<ShaderSource>("particleCalculate.comp.spv");

    struct SpecializationData {
        uint32_t workGroupSize;
        uint32_t sharedDataSize;
        float gravity;
        float power;
        float soften;
    } specializationData;
    const VkSpecializationMapEntry entries[] =
    {
        { 0u, offsetof(SpecializationData, workGroupSize), sizeof(uint32_t) },
        { 1u, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t) },
        { 2u, offsetof(SpecializationData, gravity), sizeof(float) },
        { 3u, offsetof(SpecializationData, power), sizeof(float) },
        { 4u, offsetof(SpecializationData, soften), sizeof(float) },
    };
    specializationData.workGroupSize = workGroupSize;
    specializationData.sharedDataSize = shadedMemorySize / sizeof(glm::vec4);
    specializationData.gravity = 0.002f;
    specializationData.power = 0.75f;
    specializationData.soften = 0.05f;

    VkSpecializationInfo specializationInfo =
    {
        5u,
        entries,
        to_u32(sizeof(SpecializationData)),
        &specializationData
    };

    std::vector<ShaderModule> shaderModules;
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, particleCalculateComputeShader);

    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ particleComputeDescriptorSetLayout->getHandle() };

    std::vector<VkPushConstantRange> pushConstantRangeHandles;
    VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeParticlesPushConstant) };
    pushConstantRangeHandles.push_back(computePushConstantRange);

    std::shared_ptr<ComputePipelineState> particleCalculateComputePipelineState = std::make_shared<ComputePipelineState>(
        std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
    );
    std::shared_ptr<ComputePipeline> particleCalculateComputePipeline = std::make_shared<ComputePipeline>(*device, *particleCalculateComputePipelineState, nullptr);

    pipelines.computeParticleCalculate.pipelineState = particleCalculateComputePipelineState;
    pipelines.computeParticleCalculate.pipeline = particleCalculateComputePipeline;
}

void MainApp::createParticleIntegrateComputePipeline()
{
    std::shared_ptr<ShaderSource> particleIntegrateComputeShader = std::make_shared<ShaderSource>("particleIntegrate.comp.spv");

    const VkSpecializationMapEntry entries[] =
    {
        {
            0u,
            to_u32(0 * sizeof(uint32_t)),
            sizeof(uint32_t)
        }
    };
    const uint32_t data[] = { workGroupSize };

    VkSpecializationInfo specializationInfo =
    {
        1u,
        entries,
        to_u32(1 * sizeof(uint32_t)),
        data
    };

    std::vector<ShaderModule> shaderModules;
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_COMPUTE_BIT, specializationInfo, particleIntegrateComputeShader);

    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles{ particleComputeDescriptorSetLayout->getHandle(), objectDescriptorSetLayout->getHandle()};

    std::vector<VkPushConstantRange> pushConstantRangeHandles;
    VkPushConstantRange computePushConstantRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeParticlesPushConstant) };
    pushConstantRangeHandles.push_back(computePushConstantRange);

    std::shared_ptr<ComputePipelineState> particleIntegrateComputePipelineState = std::make_shared<ComputePipelineState>(
        std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles)
    );
    std::shared_ptr<ComputePipeline> particleIntegrateComputePipeline = std::make_shared<ComputePipeline>(*device, *particleIntegrateComputePipelineState, nullptr);

    pipelines.computeParticleIntegrate.pipelineState = particleIntegrateComputePipelineState;
    pipelines.computeParticleIntegrate.pipeline = particleIntegrateComputePipeline;
}

void MainApp::createFramebuffers()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        std::vector<VkImageView> offscreenAttachments{ frameData.outputImageViews[i]->getHandle(), frameData.copyOutputImageViews[i]->getHandle(), frameData.velocityImageViews[i]->getHandle(), depthImageView->getHandle() };
        frameData.offscreenFramebuffers[i] = std::make_unique<Framebuffer>(*device, *swapchain, *mainRenderPass.renderPass, offscreenAttachments);
        setDebugUtilsObjectName(device->getHandle(), frameData.offscreenFramebuffers[i]->getHandle(), "outputImageFramebuffer for frame #" + std::to_string(i));

        std::vector<VkImageView> postProcessingAttachments{ frameData.outputImageViews[i]->getHandle(), depthImageView->getHandle() };
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

void MainApp::createCommandPools()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.commandPools[i] = std::make_unique<CommandPool>(*device, graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        setDebugUtilsObjectName(device->getHandle(), frameData.commandPools[i]->getHandle(), "commandPool for frame #" + std::to_string(i));
    }
}

void MainApp::createCommandBuffers()
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

void MainApp::copyBufferToImage(const Buffer &srcBuffer, const Image &dstImage, uint32_t width, uint32_t height)
{
    std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(commandBuffer->getHandle(), srcBuffer.getHandle(), dstImage.getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    commandBuffer->end();

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer->getHandle();

    vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue->getHandle());
}

void MainApp::createDepthResources()
{
    VkFormat depthFormat = getSupportedDepthFormat(device->getPhysicalDevice().getHandle());

    VkExtent3D extent{ swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1 };
    // TODO: a single depth buffer may not be correct... https://stackoverflow.com/questions/62371266/why-is-a-single-depth-buffer-sufficient-for-this-vulkan-swapchain-render-loop
    depthImage = std::make_unique<Image>(*device, depthFormat, extent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY /* default values for remaining params */);
    depthImageView = std::make_unique<ImageView>(*depthImage, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, depthFormat);
    setDebugUtilsObjectName(device->getHandle(), depthImage->getHandle(), "depthImage");
    setDebugUtilsObjectName(device->getHandle(), depthImageView->getHandle(), "depthImageView");
}

std::unique_ptr<Image> MainApp::createTextureImage(const std::string &filename)
{
    int texWidth, texHeight, texChannels;

    stbi_uc *pixels = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
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

    VkExtent3D extent{ to_u32(texWidth), to_u32(texHeight), 1u };
    std::unique_ptr<Image> textureImage = std::make_unique<Image>(*device, VK_FORMAT_R8G8B8A8_SRGB, extent, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY /* default values for remaining params */);

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;
    // Transition the texture image to be prepared as a destination target
    std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
    textureImage->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
    commandBuffer->end();

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer->getHandle();
    vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue->getHandle());

    copyBufferToImage(*stagingBuffer, *textureImage, to_u32(texWidth), to_u32(texHeight));

    // Transition the texture image to be prepared to be read by shaders
    commandBuffer->reset();
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
    textureImage->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange);
    commandBuffer->end();

    // Use the same submit info
    vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue->getHandle());

    return textureImage;
}

std::unique_ptr<ImageView> MainApp::createTextureImageView(const Image &textureImage)
{
    return std::make_unique<ImageView>(textureImage, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R8G8B8A8_SRGB);
}

void MainApp::createTextureSampler()
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

void MainApp::copyBufferToBuffer(const Buffer &srcBuffer, const Buffer &dstBuffer, VkDeviceSize size)
{
    std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer->getHandle(), srcBuffer.getHandle(), dstBuffer.getHandle(), 1, &copyRegion);

    // Execute a transfer to the compute queue, if necessary
    if (graphicsQueue->getFamilyIndex() != transferQueue->getFamilyIndex())
    {
        LOGEANDABORT("Cases when the graphics and transfer queue are not the same are not supported yet. This logic requires verification as well.");

        VkBufferMemoryBarrier bufferBarrier =
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
            0,
            graphicsQueue->getFamilyIndex(),
            transferQueue->getFamilyIndex(),
            srcBuffer.getHandle(),
            0,
            size
        };

        vkCmdPipelineBarrier(
            commandBuffer->getHandle(),
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, &bufferBarrier,
            0, nullptr);
    }

    commandBuffer->end();

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer->getHandle();

    vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue->getHandle());
}

void MainApp::createVertexBuffer(ObjModel &objModel, const ObjLoader &objLoader)
{
    VkDeviceSize bufferSize{ sizeof(objLoader.vertices[0]) * objLoader.vertices.size() };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags;
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    objModel.vertexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.vertices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel.vertexBuffer), bufferSize);
}

void MainApp::createIndexBuffer(ObjModel &objModel, const ObjLoader &objLoader)
{
    VkDeviceSize bufferSize{ sizeof(objLoader.indices[0]) * objLoader.indices.size() };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags;
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    objModel.indexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.indices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel.indexBuffer), bufferSize);
}

void MainApp::createMaterialBuffer(ObjModel& objModel, const ObjLoader &objLoader)
{
    VkDeviceSize bufferSize{ sizeof(objLoader.materials[0]) * objLoader.materials.size() };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags;
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    objModel.materialsBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.materials.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel.materialsBuffer), bufferSize);
}

void MainApp::createMaterialIndicesBuffer(ObjModel &objModel, const ObjLoader& objLoader)
{
    VkDeviceSize bufferSize{ sizeof(objLoader.materialIndices[0]) * objLoader.materialIndices.size() };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags;
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    objModel.materialsIndexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.materialIndices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel.materialsIndexBuffer), bufferSize);
}

void MainApp::createUniformBuffers()
{
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = sizeof(CameraData);
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; // TODO: we should use staging buffers instead

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.cameraBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.cameraBuffers[i]->getHandle(), "cameraBuffers for frame #" + std::to_string(i));
    }

    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.previousFrameCameraBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.previousFrameCameraBuffers[i]->getHandle(), "previousFrameCameraBuffers for frame #" + std::to_string(i));
    }

    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.size = sizeof(LightData) * maxLightCount;
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.lightBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.lightBuffers[i]->getHandle(), "lightBuffers for frame #" + std::to_string(i));
    }
}

void MainApp::createSSBOs()
{
    VkDeviceSize bufferSize{ sizeof(ObjInstance) * maxInstanceCount };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; // TODO: we should use staging buffers instead

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.objectBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.objectBuffers[i]->getHandle(), "objectBuffers for frame #" + std::to_string(i));
    }

    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.previousFrameObjectBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.previousFrameObjectBuffers[i]->getHandle(), "previousFrameObjectBuffers for frame #" + std::to_string(i));
    }
}

// Setup and fill the compute shader storage buffers containing the particles
void MainApp::prepareParticleData()
{
    computeParticlesPushConstant.particleCount = to_u32(attractors.size()) * particlesPerAttractor;

    // Initial particle positions
    particleBuffer.resize(computeParticlesPushConstant.particleCount);

    std::default_random_engine      rndEngine((unsigned)time(nullptr));
    std::normal_distribution<float> rndDistribution(0.0f, 1.0f);

    for (uint32_t i = 0; i < to_u32(attractors.size()); i++)
    {
        for (uint32_t j = 0; j < particlesPerAttractor; j++)
        {
            Particle &particle = particleBuffer[i * particlesPerAttractor + j];

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

    particleBufferSize = particleBuffer.size() * sizeof(Particle);

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
    memcpy(mappedData, particleBuffer.data(), static_cast<size_t>(particleBufferSize));
    stagingBuffer->unmap();

    // SSBO won't be changed on the host after upload so copy to device local memory
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.particleBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.particleBuffers[i]->getHandle(), "particleBuffers for frame #" + std::to_string(i));
        copyBufferToBuffer(*stagingBuffer, *(frameData.particleBuffers[i]), particleBufferSize);
    }
}

void MainApp::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes{};
    poolSizes.resize(4);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 10;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 10;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[2].descriptorCount = 10;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[3].descriptorCount = 1;

    descriptorPool = std::make_unique<DescriptorPool>(*device, poolSizes, 11u, 0);
}

void MainApp::createDescriptorSets()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        // Global Descriptor Set
        VkDescriptorSetAllocateInfo globalDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        globalDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
        globalDescriptorSetAllocateInfo.descriptorSetCount = 1;
        globalDescriptorSetAllocateInfo.pSetLayouts = &globalDescriptorSetLayout->getHandle();
        frameData.globalDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, globalDescriptorSetAllocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.globalDescriptorSets[i]->getHandle(), "globalDescriptorSet for frame #" + std::to_string(i));

        VkDescriptorBufferInfo cameraBufferInfo{};
        cameraBufferInfo.buffer = frameData.cameraBuffers[i]->getHandle();
        cameraBufferInfo.offset = 0;
        cameraBufferInfo.range = sizeof(CameraData);
        
        VkDescriptorBufferInfo lightBufferInfo{};
        lightBufferInfo.buffer = frameData.lightBuffers[i]->getHandle();
        lightBufferInfo.offset = 0;
        lightBufferInfo.range = sizeof(LightData) * maxLightCount;
        std::array<VkDescriptorBufferInfo, 2> globalBufferInfos{ cameraBufferInfo, lightBufferInfo };

        VkWriteDescriptorSet writeGlobalDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeGlobalDescriptorSet.dstSet = frameData.globalDescriptorSets[i]->getHandle();
        writeGlobalDescriptorSet.dstBinding = 0;
        writeGlobalDescriptorSet.dstArrayElement = 0;
        writeGlobalDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeGlobalDescriptorSet.descriptorCount = globalBufferInfos.size();
        writeGlobalDescriptorSet.pBufferInfo = globalBufferInfos.data();
        writeGlobalDescriptorSet.pImageInfo = nullptr;
        writeGlobalDescriptorSet.pTexelBufferView = nullptr;

        // Object Descriptor Set
        VkDescriptorSetAllocateInfo objectDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        objectDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
        objectDescriptorSetAllocateInfo.descriptorSetCount = 1;
        objectDescriptorSetAllocateInfo.pSetLayouts = &objectDescriptorSetLayout->getHandle();
        frameData.objectDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, objectDescriptorSetAllocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.objectDescriptorSets[i]->getHandle(), "objectDescriptorSet for frame #" + std::to_string(i));

        VkDescriptorBufferInfo currentFrameObjectBufferInfo{};
        currentFrameObjectBufferInfo.buffer = frameData.objectBuffers[i]->getHandle();
        currentFrameObjectBufferInfo.offset = 0;
        currentFrameObjectBufferInfo.range = sizeof(ObjInstance) * maxInstanceCount;

        VkWriteDescriptorSet writeObjectDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeObjectDescriptorSet.dstSet = frameData.objectDescriptorSets[i]->getHandle();
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
        frameData.postProcessingDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, postProcessingDescriptorSetAllocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.postProcessingDescriptorSets[i]->getHandle(), "postProcessingDescriptorSet for frame #" + std::to_string(i));

        // Binding 0 is the camera buffer
        std::array<VkDescriptorBufferInfo, 1> postProcessingUniformBufferInfos{ cameraBufferInfo };
        VkWriteDescriptorSet writePostProcessingUniformBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writePostProcessingUniformBufferDescriptorSet.dstSet = frameData.postProcessingDescriptorSets[i]->getHandle();
        writePostProcessingUniformBufferDescriptorSet.dstBinding = 0;
        writePostProcessingUniformBufferDescriptorSet.dstArrayElement = 0;
        writePostProcessingUniformBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writePostProcessingUniformBufferDescriptorSet.descriptorCount = postProcessingUniformBufferInfos.size();
        writePostProcessingUniformBufferDescriptorSet.pBufferInfo = postProcessingUniformBufferInfos.data();
        writePostProcessingUniformBufferDescriptorSet.pImageInfo = nullptr;
        writePostProcessingUniformBufferDescriptorSet.pTexelBufferView = nullptr;

        // Binding 1 is the currentFrameObjectBuffer
        std::array<VkDescriptorBufferInfo, 1> postProcessingStorageBufferInfos{ currentFrameObjectBufferInfo };
        VkWriteDescriptorSet writePostProcessingStorageBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writePostProcessingStorageBufferDescriptorSet.dstSet = frameData.postProcessingDescriptorSets[i]->getHandle();
        writePostProcessingStorageBufferDescriptorSet.dstBinding = 1;
        writePostProcessingStorageBufferDescriptorSet.dstArrayElement = 0;
        writePostProcessingStorageBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writePostProcessingStorageBufferDescriptorSet.descriptorCount = postProcessingStorageBufferInfos.size();
        writePostProcessingStorageBufferDescriptorSet.pBufferInfo = postProcessingStorageBufferInfos.data();
        writePostProcessingStorageBufferDescriptorSet.pImageInfo = nullptr;
        writePostProcessingStorageBufferDescriptorSet.pTexelBufferView = nullptr;

        // Bindings 2, 3 and 4 are the history image, velocity image and copy output image respectively
        VkDescriptorImageInfo historyImageInfo{};
        historyImageInfo.sampler = VK_NULL_HANDLE;
        historyImageInfo.imageView = frameData.historyImageViews[i]->getHandle();
        historyImageInfo.imageLayout = frameData.historyImageViews[i]->getImage().getLayout();
        VkDescriptorImageInfo velocityImageInfo{};
        velocityImageInfo.sampler = VK_NULL_HANDLE;
        velocityImageInfo.imageView = frameData.velocityImageViews[i]->getHandle();
        velocityImageInfo.imageLayout = frameData.velocityImageViews[i]->getImage().getLayout();
        VkDescriptorImageInfo copyOutputImageInfo{};
        copyOutputImageInfo.sampler = VK_NULL_HANDLE;
        copyOutputImageInfo.imageView = frameData.copyOutputImageViews[i]->getHandle();
        copyOutputImageInfo.imageLayout = frameData.copyOutputImageViews[i]->getImage().getLayout();
        std::array<VkDescriptorImageInfo, 3> postProcessingStorageImageInfos{ historyImageInfo, velocityImageInfo, copyOutputImageInfo };
        
        VkWriteDescriptorSet writePostProcessingStorageImageDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writePostProcessingStorageImageDescriptorSet.dstSet = frameData.postProcessingDescriptorSets[i]->getHandle();
        writePostProcessingStorageImageDescriptorSet.dstBinding = 2;
        writePostProcessingStorageImageDescriptorSet.dstArrayElement = 0;
        writePostProcessingStorageImageDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writePostProcessingStorageImageDescriptorSet.descriptorCount = postProcessingStorageImageInfos.size();
        writePostProcessingStorageImageDescriptorSet.pImageInfo = postProcessingStorageImageInfos.data();
        writePostProcessingStorageImageDescriptorSet.pBufferInfo = nullptr;
        writePostProcessingStorageImageDescriptorSet.pTexelBufferView = nullptr;

        // Taa Descriptor Set
        VkDescriptorSetAllocateInfo taaDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        taaDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
        taaDescriptorSetAllocateInfo.descriptorSetCount = 1;
        taaDescriptorSetAllocateInfo.pSetLayouts = &taaDescriptorSetLayout->getHandle();
        frameData.taaDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, taaDescriptorSetAllocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.taaDescriptorSets[i]->getHandle(), "taaDescriptorSet for frame #" + std::to_string(i));
        
        // Binding 0 is the previous frame camera buffer
        VkDescriptorBufferInfo previousFrameCameraBufferInfo{};
        previousFrameCameraBufferInfo.buffer = frameData.previousFrameCameraBuffers[i]->getHandle();
        previousFrameCameraBufferInfo.offset = 0;
        previousFrameCameraBufferInfo.range = sizeof(CameraData);
        std::array<VkDescriptorBufferInfo, 1> taaUniformBufferInfos{ previousFrameCameraBufferInfo };

        VkWriteDescriptorSet writeTaaUniformBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeTaaUniformBufferDescriptorSet.dstSet = frameData.taaDescriptorSets[i]->getHandle();
        writeTaaUniformBufferDescriptorSet.dstBinding = 0;
        writeTaaUniformBufferDescriptorSet.dstArrayElement = 0;
        writeTaaUniformBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeTaaUniformBufferDescriptorSet.descriptorCount = taaUniformBufferInfos.size();
        writeTaaUniformBufferDescriptorSet.pBufferInfo = taaUniformBufferInfos.data();
        writeTaaUniformBufferDescriptorSet.pImageInfo = nullptr;
        writeTaaUniformBufferDescriptorSet.pTexelBufferView = nullptr;

        // Binding 1 is the previous frame object buffer
        VkDescriptorBufferInfo previousFrameObjectBufferInfo{};
        previousFrameObjectBufferInfo.buffer = frameData.previousFrameObjectBuffers[i]->getHandle();
        previousFrameObjectBufferInfo.offset = 0;
        previousFrameObjectBufferInfo.range = sizeof(ObjInstance) * maxInstanceCount;
        std::array<VkDescriptorBufferInfo, 1> taaStorageImageInfos{ previousFrameObjectBufferInfo };

        VkWriteDescriptorSet writeTaaStorageBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeTaaStorageBufferDescriptorSet.dstSet = frameData.taaDescriptorSets[i]->getHandle();
        writeTaaStorageBufferDescriptorSet.dstBinding = 1;
        writeTaaStorageBufferDescriptorSet.dstArrayElement = 0;
        writeTaaStorageBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeTaaStorageBufferDescriptorSet.descriptorCount = taaStorageImageInfos.size();
        writeTaaStorageBufferDescriptorSet.pBufferInfo = taaStorageImageInfos.data();
        writeTaaStorageBufferDescriptorSet.pImageInfo = nullptr;
        writeTaaStorageBufferDescriptorSet.pTexelBufferView = nullptr;

        // Particle Buffer Descriptor Set
        VkDescriptorSetAllocateInfo particleComputeDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        particleComputeDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
        particleComputeDescriptorSetAllocateInfo.descriptorSetCount = 1;
        particleComputeDescriptorSetAllocateInfo.pSetLayouts = &particleComputeDescriptorSetLayout->getHandle();
        frameData.particleComputeDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, particleComputeDescriptorSetAllocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.particleComputeDescriptorSets[i]->getHandle(), "particleComputeDescriptorSet for frame #" + std::to_string(i));

        // Binding 0 is the particle buffer
        VkDescriptorBufferInfo particleBufferInfo{};
        particleBufferInfo.buffer = frameData.particleBuffers[i]->getHandle();
        particleBufferInfo.offset = 0;
        particleBufferInfo.range = sizeof(Particle) * computeParticlesPushConstant.particleCount;
        std::array<VkDescriptorBufferInfo, 1> particleComputeStorageBufferInfos{ particleBufferInfo };

        VkWriteDescriptorSet writeParticleComputeStorageBufferDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeParticleComputeStorageBufferDescriptorSet.dstSet = frameData.particleComputeDescriptorSets[i]->getHandle();
        writeParticleComputeStorageBufferDescriptorSet.dstBinding = 0;
        writeParticleComputeStorageBufferDescriptorSet.dstArrayElement = 0;
        writeParticleComputeStorageBufferDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeParticleComputeStorageBufferDescriptorSet.descriptorCount = particleComputeStorageBufferInfos.size();
        writeParticleComputeStorageBufferDescriptorSet.pBufferInfo = particleComputeStorageBufferInfos.data();
        writeParticleComputeStorageBufferDescriptorSet.pImageInfo = nullptr;
        writeParticleComputeStorageBufferDescriptorSet.pTexelBufferView = nullptr;

        // Write descriptor sets
        std::array<VkWriteDescriptorSet, 8> writeDescriptorSets {
            writeGlobalDescriptorSet,
            writeObjectDescriptorSet,

            writePostProcessingUniformBufferDescriptorSet,
            writePostProcessingStorageBufferDescriptorSet,
            writePostProcessingStorageImageDescriptorSet,

            writeTaaUniformBufferDescriptorSet,
            writeTaaStorageBufferDescriptorSet,

            writeParticleComputeStorageBufferDescriptorSet
        };
        vkUpdateDescriptorSets(device->getHandle(), to_u32(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
    }

    // Texture Descriptor Set
    VkDescriptorSetAllocateInfo textureDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    textureDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
    textureDescriptorSetAllocateInfo.descriptorSetCount = 1;
    textureDescriptorSetAllocateInfo.pSetLayouts = &textureDescriptorSetLayout->getHandle();
    textureDescriptorSet = std::make_unique<DescriptorSet>(*device, textureDescriptorSetAllocateInfo);
    setDebugUtilsObjectName(device->getHandle(), textureDescriptorSet->getHandle(), "textureDescriptorSet");

    std::vector<VkDescriptorImageInfo> textureImageInfos;
    for (auto &texture : textures)
    {
        VkDescriptorImageInfo textureImageInfo{};
        textureImageInfo.sampler = textureSampler->getHandle();
        textureImageInfo.imageView = texture.imageview->getHandle();
        textureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        textureImageInfos.push_back(textureImageInfo);
    }

    // Binding 0 is the texture image
    VkWriteDescriptorSet writeTextureDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writeTextureDescriptorSet.dstSet = textureDescriptorSet->getHandle();
    writeTextureDescriptorSet.dstBinding = 0;
    writeTextureDescriptorSet.dstArrayElement = 0;
    writeTextureDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeTextureDescriptorSet.descriptorCount = textureImageInfos.size();
    writeTextureDescriptorSet.pImageInfo = textureImageInfos.data();
    writeTextureDescriptorSet.pBufferInfo = nullptr;
    writeTextureDescriptorSet.pTexelBufferView = nullptr;

    std::array<VkWriteDescriptorSet, 1> writeToDescriptorSets{ writeTextureDescriptorSet };
    vkUpdateDescriptorSets(device->getHandle(), to_u32(writeToDescriptorSets.size()), writeToDescriptorSets.data(), 0, nullptr);
}

void MainApp::updateComputeDescriptorSet()
{
    VkDescriptorBufferInfo currentFrameObjectBufferInfo{};
    currentFrameObjectBufferInfo.buffer = frameData.objectBuffers[currentFrame]->getHandle();
    currentFrameObjectBufferInfo.offset = 0;
    currentFrameObjectBufferInfo.range = sizeof(ObjInstance) * maxInstanceCount;
    std::array<VkDescriptorBufferInfo, 1> storageImageInfos{ currentFrameObjectBufferInfo };

    VkWriteDescriptorSet writeObjectDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writeObjectDescriptorSet.dstSet = frameData.objectDescriptorSets[currentFrame]->getHandle();
    writeObjectDescriptorSet.dstBinding = 0;
    writeObjectDescriptorSet.dstArrayElement = 0;
    writeObjectDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeObjectDescriptorSet.descriptorCount = storageImageInfos.size();
    writeObjectDescriptorSet.pBufferInfo = storageImageInfos.data();
    writeObjectDescriptorSet.pImageInfo = nullptr;
    writeObjectDescriptorSet.pTexelBufferView = nullptr;

    std::array<VkWriteDescriptorSet, 1> writes{ writeObjectDescriptorSet };
    vkUpdateDescriptorSets(device->getHandle(), to_u32(writes.size()), writes.data(), 0, nullptr);
}

void MainApp::createSemaphoreAndFencePools()
{
    semaphorePool = std::make_unique<SemaphorePool>(*device);
    fencePool = std::make_unique<FencePool>(*device);
}

void MainApp::loadTextureImages(const std::vector<std::string> &textureFiles)
{
    LOGI("Loading {} texture files..", std::to_string(textureFiles.size()));
    int x = 1;
    for (const std::string &textureFile : textureFiles)
    {
        LOGI(std::to_string(x++) + ": processing " + textureFile);
        Texture texture;
        texture.image = createTextureImage(std::string("../../assets/textures/" + textureFile).c_str());
        texture.imageview = createTextureImageView(*(texture.image));
        textures.emplace_back(std::move(texture));
    }
}

void MainApp::setupSynchronizationObjects()
{
    imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);

    for (size_t i = 0; i < maxFramesInFlight; ++i) {
        frameData.imageAvailableSemaphores[i] = semaphorePool->requestSemaphore();
        frameData.offscreenRenderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
        frameData.postProcessRenderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
        frameData.outputImageCopyFinishedSemaphores[i] = semaphorePool->requestSemaphore();
        frameData.computeParticlesFinishedSemaphores[i] = semaphorePool->requestSemaphore();
        frameData.inFlightFences[i] = fencePool->requestFence();
    }
}

void MainApp::loadModel(const std::string &objFileName)
{
    const std::string modelPath = "../../assets/models/";
    const std::string filePath = modelPath + objFileName;

    ObjModel objModel;
    ObjLoader objLoader;
    objLoader.loadModel(filePath.c_str());

    // Applying gamma correction to convert the ambient, diffuse and specular values from srgb non-linear to srgb linear prior to usage in shaders
    for (auto &m : objLoader.materials)
    {
        m.ambient = glm::pow(m.ambient, glm::vec3(2.2f));
        m.diffuse = glm::pow(m.diffuse, glm::vec3(2.2f));
        m.specular = glm::pow(m.specular, glm::vec3(2.2f));
    }

    objModel.verticesCount = to_u32(objLoader.vertices.size());
    objModel.indicesCount = to_u32(objLoader.indices.size());

    createVertexBuffer(objModel, objLoader);
    createIndexBuffer(objModel, objLoader);
    createMaterialBuffer(objModel, objLoader);
    createMaterialIndicesBuffer(objModel, objLoader);

    std::string objNb = std::to_string(objModels.size());
    setDebugUtilsObjectName(device->getHandle(), objModel.vertexBuffer->getHandle(), (std::string("vertex_" + objNb).c_str()));
    setDebugUtilsObjectName(device->getHandle(), objModel.indexBuffer->getHandle(), (std::string("index_" + objNb).c_str()));
    setDebugUtilsObjectName(device->getHandle(), objModel.materialsBuffer->getHandle(), (std::string("mat_" + objNb).c_str()));
    setDebugUtilsObjectName(device->getHandle(), objModel.materialsIndexBuffer->getHandle(), (std::string("matIdx_" + objNb).c_str()));

    objModel.objFileName = objFileName;
    objModel.txtOffset = static_cast<uint64_t>(textures.size());
    loadTextureImages(objLoader.textures);

    objModels.push_back(std::move(objModel));
}

void MainApp::createInstance(const std::string &objFileName, glm::mat4 transform)
{
    uint64_t objModelIndex = getObjModelIndex(objFileName);
    ObjModel &objModel = objModels[objModelIndex];

    ObjInstance instance;
    instance.transform = transform;
    instance.transformIT = glm::transpose(glm::inverse(transform));
    instance.objIndex = objModelIndex;
    instance.textureOffset = objModel.txtOffset;

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR };
    bufferDeviceAddressInfo.buffer = objModel.vertexBuffer->getHandle();
    instance.vertices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModel.indexBuffer->getHandle();
    instance.indices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModel.materialsBuffer->getHandle();
    instance.materials = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModel.materialsIndexBuffer->getHandle();
    instance.materialIndices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

    objInstances.push_back(std::move(instance));
}

void MainApp::createSceneLights()
{
    LightData l1;
    LightData l2;
    l2.lightPosition = glm::vec3(10.0f, 5.0f, 4.5f);
    sceneLights.emplace_back(std::move(l1));
    sceneLights.emplace_back(std::move(l2));

    rasterizationPushConstant.lightCount = sceneLights.size();
    raytracingPushConstant.lightCount = sceneLights.size();
}

void MainApp::loadModels()
{
    // TODO: note that loading in the sphere models error out for raytracing so that needs to be resolved when it is used
    loadModel("plane.obj");
    loadModel("Medieval_building.obj");
    loadModel("wuson.obj");
    loadModel("cube.obj");
    //loadModel("monkey_smooth.obj");
    //loadModel("lost_empire.obj");

    // Validate that models are only loaded once
    std::set<std::string> existingModels;
    for (int i = 0; i < objModels.size(); ++i)
    {
        if (existingModels.count(objModels[i].objFileName) != 0)
        {
            LOGEANDABORT("Duplicate models have been loaded!");
        }
        existingModels.insert(objModels[i].objFileName);
    }
}

void MainApp::createScene()
{
    if (maxInstanceCount > device->getPhysicalDevice().getAccelerationStructureProperties().maxInstanceCount)
    {
        LOGEANDABORT("Max instance count is above the limit supported by the GPU");
    }

    // The sphere instance index is hardcoded in the animate.comp file, so when you add or remove an instance, that must be updated
    createInstance("plane.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(0, 0, 0)));
    createInstance("Medieval_building.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 5, 0,0 }));
    // All wuson instances are assumed to be one after another for the transformation matrix calculations
    createInstance("wuson.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 3)));
    createInstance("wuson.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 7)));
    createInstance("wuson.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 10)));
    createInstance("Medieval_building.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 15, 0,0 }));
    //createInstance("monkey_smooth.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 3)));
    //createInstance("lost_empire.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 5,-10,0 }));

    // ALl particle instances are assumed to be grouped together
    computeParticlesPushConstant.startingIndex = static_cast<int>(objInstances.size());
    for (int i = 0; i < computeParticlesPushConstant.particleCount; ++i)
    {
        createInstance("cube.obj", glm::translate(glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2f, 0.2f, 0.2f)), glm::vec3(particleBuffer[i].position.xyz)));
    }

    if (objInstances.size() > maxInstanceCount)
    {
        LOGEANDABORT("There are more instances than maxInstanceCount. You need to increase this value to support more instances");
    }
}

uint64_t MainApp::getObjModelIndex(const std::string &name)
{
    for (uint64_t i = 0; i < objModels.size(); ++i)
    {
        if (objModels[i].objFileName.compare(name) == 0) return i;
    }

    LOGEANDABORT("Object model {} not found", name);
}

void MainApp::initializeImGui()
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
    initInfo.Instance = instance->getHandle();
    initInfo.PhysicalDevice = device->getPhysicalDevice().getHandle();
    initInfo.Device = device->getHandle();
    initInfo.QueueFamily = graphicsQueue->getFamilyIndex();
    initInfo.Queue = graphicsQueue->getHandle();
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

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer->getHandle();

    vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue->getHandle());

    // Clear font data on CPU
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void MainApp::resetFrameSinceViewChange()
{
    raytracingPushConstant.frameSinceViewChange = -1; // TODO remove
    taaPushConstant.frameSinceViewChange = -1;
}

void MainApp::updateTaaState()
{
    if (!temporalAntiAliasingEnabled && !raytracingEnabled) // remove false
    {
        raytracingPushConstant.frameSinceViewChange = 0; // TODO remove
        taaPushConstant.frameSinceViewChange = 0;
        taaPushConstant.jitter = glm::vec2(0.0f);
        return;
    }

    // If the camera has updated, we don't want to use the previous frame for anti aliasing
    if (cameraController->getCamera()->isUpdated())
    {
        resetFrameSinceViewChange();
        cameraController->getCamera()->resetUpdatedFlag();
    }
    raytracingPushConstant.frameSinceViewChange += 1;// TODO remove
    taaPushConstant.frameSinceViewChange += 1;
}

void MainApp::createImageResourcesForFrames()
{
    VkExtent3D extent{ swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1 };
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.outputImages[i] = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        frameData.outputImageViews[i] = std::make_unique<ImageView>(*(frameData.outputImages[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, frameData.outputImages[i]->getFormat());
        frameData.copyOutputImages[i] = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        frameData.copyOutputImageViews[i] = std::make_unique<ImageView>(*(frameData.copyOutputImages[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, frameData.copyOutputImages[i]->getFormat());
        frameData.historyImages[i] = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        frameData.historyImageViews[i] = std::make_unique<ImageView>(*(frameData.historyImages[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, frameData.historyImages[i]->getFormat());
        frameData.velocityImages[i] = std::make_unique<Image>(*device, swapchain->getProperties().surfaceFormat.format, extent, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        frameData.velocityImageViews[i] = std::make_unique<ImageView>(*(frameData.velocityImages[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, frameData.velocityImages[i]->getFormat());

        setDebugUtilsObjectName(device->getHandle(), frameData.outputImages[i]->getHandle(), "outputImage for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.outputImageViews[i]->getHandle(), "outputImageView for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.copyOutputImages[i]->getHandle(), "copyOutputImage for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.copyOutputImageViews[i]->getHandle(), "copyOutputImageView for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.historyImages[i]->getHandle(), "historyImage for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.historyImageViews[i]->getHandle(), "historyImageView for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.velocityImages[i]->getHandle(), "velocityImage for frame #" + std::to_string(i));
        setDebugUtilsObjectName(device->getHandle(), frameData.velocityImageViews[i]->getHandle(), "velocityImageView for frame #" + std::to_string(i));

        std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
        frameData.outputImages[i]->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
        frameData.copyOutputImages[i]->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
        frameData.historyImages[i]->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
        frameData.velocityImages[i]->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
        commandBuffer->end();

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer->getHandle();
        vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue->getHandle());
    }
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
BlasInput MainApp::objectToVkGeometryKHR(size_t objModelIndex)
{
    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    bufferDeviceAddressInfo.buffer = objModels[objModelIndex].vertexBuffer->getHandle();
    // We can take advantage of the fact that position is the first member of Vertex
    VkDeviceAddress vertexAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModels[objModelIndex].indexBuffer->getHandle();
    VkDeviceAddress indexAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

    uint32_t maxPrimitiveCount = objModels[objModelIndex].indicesCount / 3;

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
    triangles.maxVertex = objModels[objModelIndex].verticesCount;

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

void MainApp::createBottomLevelAS()
{
    // BLAS - Storing each model in a geometry
    std::vector<BlasInput> allBlas;
    allBlas.reserve(objModels.size());
    for (uint32_t i = 0; i < objModels.size(); ++i)
    {
        auto blas = objectToVkGeometryKHR(i);

        // We could add more geometry in each BLAS, but we add only one for now
        allBlas.emplace_back(blas);
    }
    // TODO BUILD THE BLASENTRY DIRECTLY RATHER THAN DOING THIS AND ENABLE COMPACTION
    buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR/* | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR */);
}

std::unique_ptr<AccelKHR> MainApp::createAcceleration(VkAccelerationStructureCreateInfoKHR &accel)
{
    std::unique_ptr<AccelKHR> resultAccel = std::make_unique<AccelKHR>(*device);
    // Allocating the buffer to hold the acceleration structure
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = accel.size;
    bufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    resultAccel->buffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
    accel.buffer = resultAccel->buffer->getHandle();

    // Create the acceleration structure
    vkCreateAccelerationStructureKHR(device->getHandle(), &accel, nullptr, &resultAccel->accel);

    return std::move(resultAccel);
}


void MainApp::buildBlas(const std::vector<BlasInput> &input, VkBuildAccelerationStructureFlagsKHR buildAccelerationStructureFlags)
{
    m_blas = std::vector<BlasEntry>(input.begin(), input.end());
    uint32_t nbBlas = to_u32(m_blas.size());

    // Preparing the build information array for the acceleration build command.
    // This is mostly just a fancy pointer to the user-passed arrays of VkAccelerationStructureGeometryKHR.
    // dstAccelerationStructure will be filled later once we allocated the acceleration structures.
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(nbBlas);
    for (uint32_t idx = 0; idx < nbBlas; idx++)
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
    VkDeviceSize              maxScratch{ 0 };          // Largest scratch buffer for our BLAS
    std::vector<VkDeviceSize> originalSizes(nbBlas);  // use for stats

    for (size_t idx = 0; idx < nbBlas; idx++)
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
        m_blas[idx].as = createAcceleration(createInfo);

        setDebugUtilsObjectName(device->getHandle(), m_blas[idx].as->accel, std::string("BLAS Acceleration Structure #" + std::to_string(idx)));
        setDebugUtilsObjectName(device->getHandle(), m_blas[idx].as->buffer->getHandle(), std::string("BLAS Acceleration Structure Buffer #" + std::to_string(idx)));
        buildInfos[idx].dstAccelerationStructure = m_blas[idx].as->accel;  // Setting the where the build lands

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
    qpci.queryCount = nbBlas;
    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    VkQueryPool queryPool;
    vkCreateQueryPool(device->getHandle(), &qpci, nullptr, &queryPool);
    vkResetQueryPool(device->getHandle(), queryPool, 0, nbBlas);

    // Allocate a command pool for queue of given queue index.
    // To avoid timeout, record and submit one command buffer per AS build.
    CommandPool asCommandPool(*device, graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::vector<std::shared_ptr<CommandBuffer>> allCmdBufs;
    allCmdBufs.reserve(nbBlas);

    // Building the acceleration structures
    for (uint32_t idx = 0; idx < nbBlas; idx++)
    {
        BlasEntry &blas = m_blas[idx];
        std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        allCmdBufs.push_back(cmdBuf);
        cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

        // All build are using the same scratch buffer
        buildInfos[idx].scratchData.deviceAddress = scratchAddress;

        // Convert user vector of offsets to vector of pointer-to-offset (required by vk).
        // Recall that this defines which (sub)section of the vertex/index arrays
        // will be built into the BLAS.
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> pBuildOffset(blas.input.asBuildRangeInfo.size());
        for (size_t infoIdx = 0; infoIdx < blas.input.asBuildRangeInfo.size(); infoIdx++)
            pBuildOffset[infoIdx] = &blas.input.asBuildRangeInfo[infoIdx];

        // Building the AS
        vkCmdBuildAccelerationStructuresKHR(cmdBuf->getHandle(), 1, &buildInfos[idx], pBuildOffset.data());

        // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
        // is finished before starting the next one
        VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmdBuf->getHandle(), VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Write compacted size to query number idx.
        if (doCompaction)
        {
            vkCmdWriteAccelerationStructuresPropertiesKHR(
                cmdBuf->getHandle(), 1, &blas.as->accel,
                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, idx
            );
        }
    }

    // submit and wait
    std::vector<VkCommandBuffer> handles;
    for (uint32_t idx = 0; idx < allCmdBufs.size(); idx++)
    {
        allCmdBufs[idx]->end();
        handles.push_back(allCmdBufs[idx]->getHandle());
    }
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pCommandBuffers = handles.data();
    submitInfo.commandBufferCount = to_u32(handles.size());
    VK_CHECK(vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(graphicsQueue->getHandle());
    allCmdBufs.clear();

    // Compacting all BLAS
    if (doCompaction)
    {
        std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

        // Get the size result back
        std::vector<VkDeviceSize> compactSizes(nbBlas);
        vkGetQueryPoolResults(device->getHandle(), queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
            compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);


        // Compacting
        uint32_t                    statTotalOriSize{ 0 }, statTotalCompactSize{ 0 };
        for (uint32_t idx = 0; idx < nbBlas; idx++)
        {
            // LOGD("Reducing %i, from %d to %d \n", i, originalSizes[i], compactSizes[i]);
            statTotalOriSize += (uint32_t)originalSizes[idx];
            statTotalCompactSize += (uint32_t)compactSizes[idx];

            // Creating a compact version of the AS
            VkAccelerationStructureCreateInfoKHR asCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
            asCreateInfo.size = compactSizes[idx];
            asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            std::unique_ptr<AccelKHR> as = createAcceleration(asCreateInfo);

            // Copy the original BLAS to a compact version
            VkCopyAccelerationStructureInfoKHR copyInfo{ VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
            copyInfo.src = m_blas[idx].as->accel;
            copyInfo.dst = as->accel;
            copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
            vkCmdCopyAccelerationStructureKHR(cmdBuf->getHandle(), &copyInfo);
            // Clean up AS
            vkDestroyAccelerationStructureKHR(device->getHandle(), m_blas[idx].as->accel, nullptr);
            m_blas[idx].as.reset();
            m_blas[idx].as = std::move(as);

            setDebugUtilsObjectName(device->getHandle(), m_blas[idx].as->accel, std::string("BLAS Acceleration Structure #" + std::to_string(idx)));
            setDebugUtilsObjectName(device->getHandle(), m_blas[idx].as->buffer->getHandle(), std::string("BLAS Acceleration Structure Buffer #" + std::to_string(idx)));
        }

        // submitandwaitidle
        cmdBuf->end();
        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.pCommandBuffers = &cmdBuf->getHandle();
        submitInfo.commandBufferCount = 1u;
        VK_CHECK(vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE));
        vkQueueWaitIdle(graphicsQueue->getHandle());

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

void MainApp::buildTlas(bool update)
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
        void *mappedData = frameData.objectBuffers[currentFrame]->map();
        ObjInstance *objectSSBO = static_cast<ObjInstance *>(mappedData);
        for (int i = 0; i < objInstances.size(); ++i)
        {
            accelerationStructureInstances[i].transform = toTransformMatrixKHR(objectSSBO[i].transform);
        }
        frameData.objectBuffers[currentFrame]->unmap();
    }
    else
    {
        // First time creation
        accelerationStructureInstances.reserve(objInstances.size());
        for (size_t i = 0; i < objInstances.size(); ++i)
        {
            VkAccelerationStructureInstanceKHR accelerationStructureInstance;
            accelerationStructureInstance.transform = toTransformMatrixKHR(objInstances[i].transform);  // Position of the instance
            accelerationStructureInstance.instanceCustomIndex = objInstances[i].objIndex;               // gl_InstanceCustomIndexEXT
            accelerationStructureInstance.mask = 0xFF;
            accelerationStructureInstance.instanceShaderBindingTableRecordOffset = 0;                   // We will use the same hit group for all objects
            accelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            accelerationStructureInstance.accelerationStructureReference = getBlasDeviceAddress(objInstances[i].objIndex);
            accelerationStructureInstances.emplace_back(accelerationStructureInstance);
        }
        rtFlags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    }

    CommandPool asCommandPool(*device, graphicsQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    m_tlas->flags = rtFlags;

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    VkDeviceSize instanceDescsSizeInBytes{ accelerationStructureInstances.size() * sizeof(VkAccelerationStructureInstanceKHR) };

    // Allocate the instance buffer and copy its contents from host to device memory
    if (update)
    {
        m_instBuffer.reset();
    }

    VkDeviceSize bufferSize{ sizeof(VkAccelerationStructureInstanceKHR) * accelerationStructureInstances.size() };
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
    memcpy(mappedData, accelerationStructureInstances.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *m_instBuffer, bufferSize);
    setDebugUtilsObjectName(device->getHandle(), m_instBuffer->getHandle(), "Instance Buffer");
    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    bufferDeviceAddressInfo.buffer = m_instBuffer->getHandle();
    VkDeviceAddress instanceAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

    // Make sure the copy of the instance buffer are copied before triggering the
    // acceleration structure build
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmdBuf->getHandle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    //--------------------------------------------------------------------------------------------------

    // Create VkAccelerationStructureGeometryInstancesDataKHR
    // This wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    instancesVk.arrayOfPointers = VK_FALSE;
    instancesVk.data.deviceAddress = instanceAddress;

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the
    // instances struct in a union and label it as instance data.
    VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    // Find sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.flags = rtFlags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &topASGeometry;
    buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    uint32_t instancesCount = to_u32(accelerationStructureInstances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkGetAccelerationStructureBuildSizesKHR(device->getHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instancesCount, &sizeInfo);

    // Create TLAS
    if (!update)
    {
        VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        accelerationStructureCreateInfo.size = sizeInfo.accelerationStructureSize;

        m_tlas->as = createAcceleration(accelerationStructureCreateInfo);
        setDebugUtilsObjectName(device->getHandle(), m_tlas->as->accel, "TLAS");
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
    buildInfo.srcAccelerationStructure = update ? m_tlas->as->accel : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = m_tlas->as->accel;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{ instancesCount, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR *pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    vkCmdBuildAccelerationStructuresKHR(cmdBuf->getHandle(), 1, &buildInfo, &pBuildOffsetInfo);

    // submitandwaitidle
    cmdBuf->end();
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pCommandBuffers = &cmdBuf->getHandle();
    submitInfo.commandBufferCount = 1u;
    VK_CHECK(vkQueueSubmit(graphicsQueue->getHandle(), 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(graphicsQueue->getHandle());
}

VkDeviceAddress MainApp::getBlasDeviceAddress(uint32_t blasId)
{
    if (blasId >= objModels.size()) LOGEANDABORT("Invalid blasId");
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    addressInfo.accelerationStructure = m_blas[blasId].as->accel;
    return vkGetAccelerationStructureDeviceAddressKHR(device->getHandle(), &addressInfo);
}

void MainApp::createRtDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes{};
    poolSizes.resize(2);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;

    m_rtDescPool = std::make_unique<DescriptorPool>(*device, poolSizes, 2u, 0);
}

void MainApp::createRtDescriptorLayout()
{
    VkDescriptorSetLayoutBinding tlasLayoutBinding{};
    tlasLayoutBinding.binding = 0;
    tlasLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    tlasLayoutBinding.descriptorCount = 1;
    tlasLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    tlasLayoutBinding.pImmutableSamplers = nullptr; // Optional

    VkDescriptorSetLayoutBinding outputImageLayoutBinding{};
    outputImageLayoutBinding.binding = 1;
    outputImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputImageLayoutBinding.descriptorCount = 1;
    outputImageLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    outputImageLayoutBinding.pImmutableSamplers = nullptr; // Optional

    std::vector<VkDescriptorSetLayoutBinding> rtDescriptorSetLayoutBindings;
    rtDescriptorSetLayoutBindings.push_back(tlasLayoutBinding);
    rtDescriptorSetLayoutBindings.push_back(outputImageLayoutBinding);

    m_rtDescSetLayout = std::make_unique<DescriptorSetLayout>(*device, rtDescriptorSetLayoutBindings);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void MainApp::createRtDescriptorSets()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        VkDescriptorSetAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocateInfo.descriptorPool = m_rtDescPool->getHandle();
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &m_rtDescSetLayout->getHandle();

        frameData.rtDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, allocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.rtDescriptorSets[i]->getHandle(), "rtDescriptorSet for frame #" + std::to_string(i));

        VkAccelerationStructureKHR tlas = m_tlas->as->accel;
        VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        descASInfo.accelerationStructureCount = 1;
        descASInfo.pAccelerationStructures = &tlas;

        VkWriteDescriptorSet writeAccelerationStructure{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeAccelerationStructure.dstSet = frameData.rtDescriptorSets[i]->getHandle();
        writeAccelerationStructure.dstBinding = 0;
        writeAccelerationStructure.dstArrayElement = 0;
        writeAccelerationStructure.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writeAccelerationStructure.descriptorCount = 1;
        writeAccelerationStructure.pNext = &descASInfo;

        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        outputImageInfo.imageView = frameData.outputImageViews[i]->getHandle();
        outputImageInfo.sampler = {};

        VkWriteDescriptorSet writeOutputImage{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeOutputImage.dstSet = frameData.rtDescriptorSets[i]->getHandle();
        writeOutputImage.dstBinding = 1;
        writeOutputImage.dstArrayElement = 0;
        writeOutputImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeOutputImage.descriptorCount = 1;
        writeOutputImage.pImageInfo = &outputImageInfo;

        std::array<VkWriteDescriptorSet, 2> writeToDescriptorSets{ writeAccelerationStructure, writeOutputImage };
        vkUpdateDescriptorSets(device->getHandle(), to_u32(writeToDescriptorSets.size()), writeToDescriptorSets.data(), 0, nullptr);
    }
}

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void MainApp::updateRtDescriptorSet()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        outputImageInfo.imageView = frameData.outputImageViews[i]->getHandle();
        outputImageInfo.sampler = {};
        VkWriteDescriptorSet writeOutputImage{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writeOutputImage.dstSet = frameData.rtDescriptorSets[i]->getHandle();
        writeOutputImage.dstBinding = 1;
        writeOutputImage.dstArrayElement = 0;
        writeOutputImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeOutputImage.descriptorCount = 1;
        writeOutputImage.pImageInfo = &outputImageInfo;
        writeOutputImage.pBufferInfo = nullptr;
        writeOutputImage.pTexelBufferView = nullptr;

        vkUpdateDescriptorSets(device->getHandle(), 1, &writeOutputImage, 0, nullptr);
    }
}

//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void MainApp::createRtPipeline()
{
    enum StageIndices
    {
        eRaygen,
        eMiss,
        eMissShadow,
        eClosestHit,
        eShaderGroupCount
    };

    std::shared_ptr<ShaderSource> rayGenShader = std::make_shared<ShaderSource>("raytrace.rgen.spv");
    std::shared_ptr<ShaderSource> rayMissShader = std::make_shared<ShaderSource>("raytrace.rmiss.spv");
    std::shared_ptr<ShaderSource> rayShadowMissShader = std::make_shared<ShaderSource>("raytraceShadow.rmiss.spv");
    std::shared_ptr<ShaderSource> rayClosestHitShader = std::make_shared<ShaderSource>("raytrace.rchit.spv");

    struct SpecializationData {
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

    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_RAYGEN_BIT_KHR, rayGenSpecializationInfo, rayGenShader);
    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_MISS_BIT_KHR, rayMissSpecializationInfo, rayMissShader);
    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_MISS_BIT_KHR, rayShadowMissSpecializationInfo, rayShadowMissShader);
    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rayClosestHitSpecializationInfo, rayClosestHitShader);

    // All stages
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo              shaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    // Since we only need the shadermodule during the creation of the pipeline, we don't keep the information in the shader module class and delete at the end
    VkShaderModuleCreateInfo shaderModuleCreateInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    // Raygen
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[0].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[0].getShaderSource().getData().data());
    shaderStageCreateInfo.pSpecializationInfo = &raytracingShaderModules[0].getSpecializationInfo();
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[0].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[0].getStage();
    stages[eRaygen] = shaderStageCreateInfo;
    // Miss
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[1].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[1].getShaderSource().getData().data());
    shaderStageCreateInfo.pSpecializationInfo = &raytracingShaderModules[1].getSpecializationInfo();
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[1].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[1].getStage();
    stages[eMiss] = shaderStageCreateInfo;
    // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[2].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[2].getShaderSource().getData().data());
    shaderStageCreateInfo.pSpecializationInfo = &raytracingShaderModules[2].getSpecializationInfo();
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[2].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[2].getStage();
    stages[eMissShadow] = shaderStageCreateInfo;
    // Hit Group - Closest Hit
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[3].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[3].getShaderSource().getData().data());
    shaderStageCreateInfo.pSpecializationInfo = &raytracingShaderModules[3].getSpecializationInfo();
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[3].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[3].getStage();
    stages[eClosestHit] = shaderStageCreateInfo;

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    // Raygen
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    m_rtShaderGroups.push_back(group);

    // Miss
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    m_rtShaderGroups.push_back(group);

    // Shadow Miss
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMissShadow;
    m_rtShaderGroups.push_back(group);

    // Closest Hit
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    m_rtShaderGroups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange pushConstant{ VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                     0, sizeof(LightData) };


    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstant;

    // Descriptor sets: one specific to ray tracing and three shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rtDescSetLayouts = { 
        m_rtDescSetLayout->getHandle(),
        globalDescriptorSetLayout->getHandle(),
        objectDescriptorSetLayout->getHandle(),
        textureDescriptorSetLayout->getHandle()
    };
    pipelineLayoutCreateInfo.setLayoutCount = to_u32(rtDescSetLayouts.size());
    pipelineLayoutCreateInfo.pSetLayouts = rtDescSetLayouts.data();

    vkCreatePipelineLayout(device->getHandle(), &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
    rayPipelineInfo.stageCount = to_u32(stages.size());  // Stages are shaders
    rayPipelineInfo.pStages = stages.data();

    // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
    // two miss shader groups, and one hit group.
    rayPipelineInfo.groupCount = to_u32(m_rtShaderGroups.size());
    rayPipelineInfo.pGroups = m_rtShaderGroups.data();

    // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
    // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
    // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
    // in the ray generation to avoid deep recursion.
    rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
    rayPipelineInfo.layout = m_rtPipelineLayout;

    vkCreateRayTracingPipelinesKHR(device->getHandle(), {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


    // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
    if (device->getPhysicalDevice().getRayTracingPipelineProperties().maxRayRecursionDepth <= 1)
    {
        LOGEANDABORT("Device fails to support ray recursion (maxRayRecursionDepth <= 1)");
    }

    for (auto &s : stages)
    {
        vkDestroyShaderModule(device->getHandle(), s.module, nullptr);
    }
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and write them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void MainApp::createRtShaderBindingTable()
{
    uint32_t     groupCount = to_u32(m_rtShaderGroups.size());  // 4 shaders: raygen, 2 miss, chit
    uint32_t groupHandleSize = device->getPhysicalDevice().getRayTracingPipelineProperties().shaderGroupHandleSize;            // Size of a program identifier
    // Compute the actual size needed per SBT entry (round-up to alignment needed).
    uint32_t groupSizeAligned = align_up(groupHandleSize, device->getPhysicalDevice().getRayTracingPipelineProperties().shaderGroupBaseAlignment);
    // Bytes needed for the SBT.
    VkDeviceSize sbtSize = groupCount * groupSizeAligned;

    // Fetch all the shader handles used in the pipeline. This is opaque data,/ so we store it in a vector of bytes.
    // The order of handles follow the stage entry.
    std::vector<uint8_t> shaderHandleStorage(sbtSize);
    VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device->getHandle(), m_rtPipeline, 0, groupCount, to_u32(sbtSize), shaderHandleStorage.data()));

    // Allocate a buffer for storing the SBT. Give it a debug name for NSight.
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

void MainApp::raytrace()
{
    debugUtilBeginLabel(frameData.commandBuffers[currentFrame][0]->getHandle(), "Raytrace");

    std::array<VkDescriptorSet, 4> descSets {
        frameData.rtDescriptorSets[currentFrame]->getHandle(), 
        frameData.globalDescriptorSets[currentFrame]->getHandle(),
        frameData.objectDescriptorSets[currentFrame]->getHandle(),
        textureDescriptorSet->getHandle()
    };
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame][0]->getHandle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame][0]->getHandle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
        to_u32(descSets.size()), descSets.data(), 0, nullptr);
    vkCmdPushConstants(frameData.commandBuffers[currentFrame][0]->getHandle(), m_rtPipelineLayout,
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
    std::unique_ptr<vulkr::MainApp> app = std::make_unique<vulkr::MainApp>(platform, "Vulkr");

    platform.initialize(std::move(app));
    platform.prepareApplication();

    platform.runMainProcessingLoop();
    platform.terminate();

    return EXIT_SUCCESS;
}

