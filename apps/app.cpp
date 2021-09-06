/* Copyright (c) 2021 Adithya Venkatarao
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

    for (auto &it : objModels)
    {
        it.second->indexBuffer.reset();
        it.second->vertexBuffer.reset();
        it.second->materialsBuffer.reset();
        it.second->materialsIndexBuffer.reset();
    }
    objModels.clear();

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
        frameData.outputImageViews[i].reset();
        frameData.outputImages[i].reset();
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
        frameData.commandBuffers[i].reset();
    }

    depthImage.reset();
    depthImageView.reset();

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.outputImageFramebuffers[i].reset();
    }

    for (auto &it : pipelineDataMap)
    {
        it.second->pipeline.reset();
        it.second->pipelineState.reset();
    }
    pipelineDataMap.clear();
    renderables.clear();

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
        frameData.globalDescriptorSets[i].reset();
        frameData.objectDescriptorSets[i].reset();
        frameData.globalBuffers[i].reset();
        frameData.objectBuffers[i].reset();
    }

    descriptorPool.reset();

    vkDestroyPipelineLayout(device->getHandle(), m_rtPipelineLayout, nullptr);  // TODO Put this into pipeline layout class
    vkDestroyPipeline(device->getHandle(), m_rtPipeline, nullptr); // TODO Put this into pipeline class
    m_rtDescSetLayout.reset();
    m_rtDescPool.reset();

    cameraController.reset();
}

void MainApp::prepare()
{
    Application::prepare();

    setupTimer();

    createInstance();
    createSurface();
    createDevice();

    graphicsQueue = device->getOptimalGraphicsQueue().getHandle();

    if (device->getOptimalGraphicsQueue().canSupportPresentation())
    {
        presentQueue = graphicsQueue;
    }
    else
    {
        presentQueue = device->getQueueByPresentation().getHandle();
    }

    createSwapchain();
    createDepthResources();
    createMainRenderPass();
    createPostRenderPass();
    createCommandPools();
    createCommandBuffers();
    createOutputImageAndImageView();
    createFramebuffers();

    initializeImGui();
    loadModels();

    createDescriptorSetLayouts();
    createGraphicsPipelines();
    createTextureSampler();
    createUniformBuffers();
    createSSBOs();
    createDescriptorPool();
    createDescriptorSets();
    setupCamera();
    createScene();

    createBottomLevelAS();
    createTopLevelAS();
    createRtDescriptorPool();
    createRtDescriptorLayout();
    createRtDescriptorSets();
    createRtPipeline();
    createRtShaderBindingTable();

    createSemaphoreAndFencePools();
    setupSynchronizationObjects();
}

void MainApp::update()
{
    fencePool->wait(&frameData.inFlightFences[currentFrame]);
    fencePool->reset(&frameData.inFlightFences[currentFrame]);

    //now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
    frameData.commandBuffers[currentFrame]->reset();

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
        // fencePool->wait(&imagesInFlight[swapchainImageIndex]);
    }
    imagesInFlight[swapchainImageIndex] = frameData.inFlightFences[currentFrame];

    drawImGuiInterface();

    std::vector<VkClearValue> clearValues;
    clearValues.resize(2);
    clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0u };

    frameData.commandBuffers[currentFrame]->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    // Main offscreen renderpass
    frameData.commandBuffers[currentFrame]->beginRenderPass(*mainRenderPass.renderPass, *(frameData.outputImageFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, clearValues, VK_SUBPASS_CONTENTS_INLINE);
    updateBuffersPerFrame();

    if (!raytracingEnabled)
    {
        rasterize();
    }

    frameData.commandBuffers[currentFrame]->endRenderPass();

    if (raytracingEnabled)
    {
        raytrace(swapchainImageIndex);
    }

    // Post offscreen renderpass
    frameData.commandBuffers[currentFrame]->beginRenderPass(*postRenderPass.renderPass, *(frameData.outputImageFramebuffers[currentFrame]), swapchain->getProperties().imageExtent, clearValues, VK_SUBPASS_CONTENTS_INLINE);
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frameData.commandBuffers[currentFrame]->getHandle());
    frameData.commandBuffers[currentFrame]->endRenderPass();

    // Copy the output image contents to the swapchain image
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    // Prepare the current swapchain image as a transfer destination
    swapchain->getImages()[swapchainImageIndex]->transitionImageLayout(*frameData.commandBuffers[currentFrame], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
    // Prepare ray tracing output image as transfer source
    frameData.outputImages[currentFrame]->transitionImageLayout(*frameData.commandBuffers[currentFrame], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange);

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.srcOffset = { 0, 0, 0 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstOffset = { 0, 0, 0 };
    copyRegion.extent = { swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1 };
    vkCmdCopyImage(frameData.commandBuffers[currentFrame]->getHandle(), frameData.outputImages[currentFrame]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain->getImages()[swapchainImageIndex]->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // Transition the current swapchain image back for presentation
    swapchain->getImages()[swapchainImageIndex]->transitionImageLayout(*frameData.commandBuffers[currentFrame], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, subresourceRange);
    // Transition the output image back to the general layout
    frameData.outputImages[currentFrame]->transitionImageLayout(*frameData.commandBuffers[currentFrame], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);

    frameData.commandBuffers[currentFrame]->end();

    // I have setup a subpass dependency to ensure that the render pass waits for the swapchain to finish reading from the image before accessing it
    // hence I don't need to set the wait stages to VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT 
    std::array<VkPipelineStageFlags, 1> waitStages{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    std::array<VkSemaphore, 1> waitSemaphores{ frameData.imageAvailableSemaphores[currentFrame] };
    std::array<VkSemaphore, 1> signalSemaphores{ frameData.renderingFinishedSemaphores[currentFrame] };

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.waitSemaphoreCount = to_u32(waitSemaphores.size());
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frameData.commandBuffers[currentFrame]->getHandle();
    submitInfo.signalSemaphoreCount = to_u32(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, frameData.inFlightFences[currentFrame]));

    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = to_u32(signalSemaphores.size());
    presentInfo.pWaitSemaphores = signalSemaphores.data();

    std::array<VkSwapchainKHR, 1> swapchains{ swapchain->getHandle() };
    presentInfo.swapchainCount = to_u32(swapchains.size());
    presentInfo.pSwapchains = swapchains.data();

    presentInfo.pImageIndices = &swapchainImageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);
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
    createGraphicsPipelines();
    createDepthResources();
    createFramebuffers();
    createCommandBuffers();
    createUniformBuffers();
    createSSBOs();
    createDescriptorPool();
    createDescriptorSets();
    updateRtDescriptorSet();
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
        if (ImGui::BeginTabItem("Camera"))
        {
            ImGui::Checkbox("Raytracing enabled", &raytracingEnabled);

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
            if (ImGui::CollapsingHeader("Light"))
            {
                ImGui::RadioButton("Point", &m_rtPushConstants.lightType, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Infinite", &m_rtPushConstants.lightType, 1);

                ImGui::SliderFloat3("Position", &m_rtPushConstants.lightPosition.x, -50.f, 50.f);
                ImGui::SliderFloat("Intensity", &m_rtPushConstants.lightIntensity, 0.f, 150.f);
            }
            
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    // ImGui::End();

    // ImGui::ShowDemoWindow();
}

void MainApp::updateBuffersPerFrame()
{
    // Update the camera buffer
    CameraData cameraData{};
    cameraData.view = cameraController->getCamera()->getView();
    cameraData.proj = cameraController->getCamera()->getProjection();

    void *mappedData = frameData.globalBuffers[currentFrame]->map();
    memcpy(mappedData, &cameraData, sizeof(cameraData));
    frameData.globalBuffers[currentFrame]->unmap();

    // Update the object buffer
    mappedData = frameData.objectBuffers[currentFrame]->map();
    ObjInstance *objectSSBO = (ObjInstance *)mappedData;
    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR };
    for (int index = 0; index < objInstances.size(); index++)
    {
        objectSSBO[index].transform = objInstances[index].transform;
        objectSSBO[index].transformIT = objInstances[index].transformIT;

        // TODO this doesn't have to be set per frame, we can do this once in the beginning
        objectSSBO[index].objIndex = objInstances[index].objIndex;
        objectSSBO[index].textureOffset = objInstances[index].textureOffset;
        objectSSBO[index].vertices = objInstances[index].vertices;
        objectSSBO[index].indices = objInstances[index].indices;
        objectSSBO[index].materials = objInstances[index].materials;
        objectSSBO[index].materialIndices = objInstances[index].materialIndices;
    }
    frameData.objectBuffers[currentFrame]->unmap();
}

// TODO: as opposed to doing slot based binding of descriptor sets which leads to multiple vkCmdBindDescriptorSets calls per drawcall, you can use
// frequency based descriptor sets and use dynamicOffsetCount: see https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/, or just bindless
// decriptors altogether
void MainApp::rasterize()
{
    debugUtilBeginLabel(frameData.commandBuffers[currentFrame]->getHandle(), "Rasterize");
    // Draw renderables
    std::shared_ptr<ObjModel> lastObjModel = nullptr;
    std::shared_ptr<PipelineData> lastPipeline = nullptr;
    for (int index = 0; index < renderables.size(); index++)
    {
        RenderObject object = renderables[index];

        // Bind the pipeline if it doesn't match with the already bound one
        if (object.pipelineData != lastPipeline)
        {

            vkCmdBindPipeline(frameData.commandBuffers[currentFrame]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, object.pipelineData->pipeline->getHandle());
            lastPipeline = object.pipelineData;

            // Camera data descriptor
            vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, object.pipelineData->pipelineState->getPipelineLayout().getHandle(), 0, 1, &frameData.globalDescriptorSets[currentFrame]->getHandle(), 0, nullptr);

            // Object data descriptor
            vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, object.pipelineData->pipelineState->getPipelineLayout().getHandle(), 1, 1, &frameData.objectDescriptorSets[currentFrame]->getHandle(), 0, nullptr);

            if (object.pipelineData == getPipelineData("texturedmesh"))
            {
                // TODO we only need to bind this once so we should move it out of this method
                // Texture descriptor
                vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, object.pipelineData->pipelineState->getPipelineLayout().getHandle(), 2, 1, &textureDescriptorSet->getHandle(), 0, nullptr);

            }
        }

        // Bind the objModel if it's a different one from last one
        if (object.objModel != lastObjModel)
        {
            VkBuffer vertexBuffers[] = { object.objModel->vertexBuffer->getHandle() };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(frameData.commandBuffers[currentFrame]->getHandle(), 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(frameData.commandBuffers[currentFrame]->getHandle(), object.objModel->indexBuffer->getHandle(), 0, VK_INDEX_TYPE_UINT32);

            lastObjModel = object.objModel;
        }

        vkCmdDrawIndexed(frameData.commandBuffers[currentFrame]->getHandle(), to_u32(object.objModel->indicesCount), 1, 0, 0, index);
    }
    
    debugUtilEndLabel(frameData.commandBuffers[currentFrame]->getHandle());
}

void MainApp::setupTimer()
{
    drawingTimer = std::make_unique<Timer>();
    drawingTimer->start();
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

    device = std::make_unique<Device>(std::move(physicalDevice), surface, deviceExtensions);
}

void MainApp::createSwapchain()
{
    const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT };
    swapchain = std::make_unique<Swapchain>(*device, surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags);
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
    Attachment colorAttachment{}; // outputImage
    colorAttachment.format = swapchain->getProperties().surfaceFormat.format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    attachments.push_back(colorAttachment);

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    mainRenderPass.colorAttachments.push_back(colorAttachmentRef);

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
    dependencies.resize(1);

    // Only need a dependency coming in to ensure that the first layout transition happens at the right time.
    // Second external dependency is implied by having a different finalLayout and subpass layout.
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0; // References the subpass index in the subpasses array
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0; // We don't have anything that we need to flush
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
    // but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
    // which injects an external dependency here with dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. 

    mainRenderPass.renderPass = std::make_unique<RenderPass>(*device, attachments, mainRenderPass.subpasses, dependencies);
}

// TODO check if we need the depth attachment for the post render pass
void MainApp::createPostRenderPass()
{
    std::vector<Attachment> attachments;
    Attachment colorAttachment{}; // swapchainImage
    colorAttachment.format = swapchain->getProperties().surfaceFormat.format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; 
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
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
    dependencies.resize(1);

    // Only need a dependency coming in to ensure that the first layout transition happens at the right time.
    // Second external dependency is implied by having a different finalLayout and subpass layout.
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0; // References the subpass index in the subpasses array
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0; // We don't have anything that we need to flush
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
    // but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
    // which injects an external dependency here with dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. 

    postRenderPass.renderPass = std::make_unique<RenderPass>(*device, attachments, postRenderPass.subpasses, dependencies);
}


void MainApp::createDescriptorSetLayouts()
{
    // Global descriptor set layout
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

    std::vector<VkDescriptorSetLayoutBinding> globalDescriptorSetLayoutBindings{ uboLayoutBinding };
    globalDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, globalDescriptorSetLayoutBindings);

    // Object descriptor set layout
    VkDescriptorSetLayoutBinding objectLayoutBinding{};
    objectLayoutBinding.binding = 0;
    objectLayoutBinding.descriptorCount = 1;
    objectLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    objectLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    objectLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> objectDescriptorSetLayoutBindings{ objectLayoutBinding };
    objectDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, objectDescriptorSetLayoutBindings);

    // Texture descriptor set layout
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = to_u32(textures.size());
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> textureDescriptorSetLayoutBindings{ samplerLayoutBinding };
    textureDescriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, textureDescriptorSetLayoutBindings);
}

std::shared_ptr<PipelineData> MainApp::createPipelineData(std::shared_ptr<GraphicsPipeline> pipeline, std::shared_ptr<PipelineState> pipelineState, const std::string &name)
{
    std::shared_ptr<PipelineData> pipelineData = std::make_shared<PipelineData>();
    pipelineData->pipeline = pipeline;
    pipelineData->pipelineState = pipelineState;
    pipelineDataMap[name] = pipelineData;
    return pipelineData;
}

void MainApp::createGraphicsPipelines()
{
    // Setup pipeline
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
    colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    ColorBlendState colorBlendState{};
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.logicOp = VK_LOGIC_OP_COPY;
    colorBlendState.attachments.emplace_back(colorBlendAttachmentState);
    colorBlendState.blendConstants[0] = 0.0f;
    colorBlendState.blendConstants[1] = 0.0f;
    colorBlendState.blendConstants[2] = 0.0f;
    colorBlendState.blendConstants[3] = 0.0f;

    std::shared_ptr<ShaderSource> vertexShader = std::make_shared<ShaderSource>("main.vert.spv");
    std::shared_ptr<ShaderSource> defaultFragmentShader = std::make_shared<ShaderSource>("default.frag.spv");
    std::shared_ptr<ShaderSource> texturedFragmentShader = std::make_shared<ShaderSource>("textured.frag.spv");

    std::vector<ShaderModule> shaderModules;
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, vertexShader);
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, defaultFragmentShader);

    std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandles {
        globalDescriptorSetLayout->getHandle(),
        objectDescriptorSetLayout->getHandle()
    };
    std::vector<VkPushConstantRange> pushConstantRangeHandles;

    // Create default mesh pipeline
    std::shared_ptr<PipelineState> defaultMeshPipelineState = std::make_shared<PipelineState>(
        std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles),
        *mainRenderPass.renderPass,
        vertexInputState,
        inputAssemblyState,
        viewportState,
        rasterizationState,
        multisampleState,
        depthStencilState,
        colorBlendState
    );

    std::shared_ptr<GraphicsPipeline> defaultMeshPipeline = std::make_shared<GraphicsPipeline>(*device, *defaultMeshPipelineState, nullptr);

    createPipelineData(defaultMeshPipeline, defaultMeshPipelineState, "defaultmesh");

    // Create textured mesh pipeline
    shaderModules.clear();
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, vertexShader);
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, texturedFragmentShader);

    descriptorSetLayoutHandles.clear();
    descriptorSetLayoutHandles.push_back(globalDescriptorSetLayout->getHandle());
    descriptorSetLayoutHandles.push_back(objectDescriptorSetLayout->getHandle());
    descriptorSetLayoutHandles.push_back(textureDescriptorSetLayout->getHandle());

    std::shared_ptr<PipelineState> texturedMeshPipelineState = std::make_shared<PipelineState>(
        std::make_unique<PipelineLayout>(*device, shaderModules, descriptorSetLayoutHandles, pushConstantRangeHandles),
        *mainRenderPass.renderPass,
        vertexInputState,
        inputAssemblyState,
        viewportState,
        rasterizationState,
        multisampleState,
        depthStencilState,
        colorBlendState
        );

    std::shared_ptr<GraphicsPipeline> texturedMeshPipeline = std::make_shared<GraphicsPipeline>(*device, *texturedMeshPipelineState, nullptr);

    createPipelineData(texturedMeshPipeline, texturedMeshPipelineState, "texturedmesh");
}

void MainApp::createFramebuffers()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        std::vector<VkImageView> attachments{ frameData.outputImageViews[i]->getHandle(), depthImageView->getHandle() };
        frameData.outputImageFramebuffers[i] = std::make_unique<Framebuffer>(*device, *swapchain, *mainRenderPass.renderPass, attachments);
        setDebugUtilsObjectName(device->getHandle(), frameData.outputImageFramebuffers[i]->getHandle(), "outputImageFramebuffer for frame #" + std::to_string(i));
    }
}

void MainApp::createCommandPools()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.commandPools[i] = std::make_unique<CommandPool>(*device, device->getOptimalGraphicsQueue().getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        setDebugUtilsObjectName(device->getHandle(), frameData.commandPools[i]->getHandle(), "commandPool for frame #" + std::to_string(i));
    }
}

void MainApp::createCommandBuffers()
{
    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.commandBuffers[i] = std::make_unique<CommandBuffer>(*frameData.commandPools[i], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        setDebugUtilsObjectName(device->getHandle(), frameData.commandBuffers[i]->getHandle(), "commandBuffer for frame #" + std::to_string(i));
    }
}

// TODO create a new command pool and allocate the command buffer using requestCommandBuffer
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

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
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

std::unique_ptr<Image> MainApp::createTextureImage(const char *filename)
{
    int texWidth, texHeight, texChannels;

    stbi_uc *pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        LOGEANDABORT("failed to load texture image!");
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
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    copyBufferToImage(*stagingBuffer, *textureImage, to_u32(texWidth), to_u32(texHeight));

    // Transition the texture image to be prepared to be read by shaders
    commandBuffer->reset();
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
    textureImage->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange);
    commandBuffer->end();

    // Use the same submit info
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

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

    commandBuffer->end();

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer->getHandle();

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
}

void MainApp::createVertexBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader)
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

    objModel->vertexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.vertices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel->vertexBuffer), bufferSize);
}

void MainApp::createIndexBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader)
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

    objModel->indexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.indices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel->indexBuffer), bufferSize);
}

void MainApp::createMaterialBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader)
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

    objModel->materialsBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.materials.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel->materialsBuffer), bufferSize);
}

void MainApp::createMaterialIndicesBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader)
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

    objModel->materialsIndexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, objLoader.materialIndices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBufferToBuffer(*stagingBuffer, *(objModel->materialsIndexBuffer), bufferSize);
}

void MainApp::createUniformBuffers()
{
    VkDeviceSize bufferSize{ sizeof(CameraData) };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.globalBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
    }
}

void MainApp::createSSBOs()
{
    VkDeviceSize bufferSize{ sizeof(ObjInstance) * MAX_OBJECT_COUNT };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        frameData.objectBuffers[i] = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
    }
}

void MainApp::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes{};
    poolSizes.resize(3);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = maxFramesInFlight;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = maxFramesInFlight;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = 1;

    descriptorPool = std::make_unique<DescriptorPool>(*device, poolSizes, 10u, 0);
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
        cameraBufferInfo.buffer = frameData.globalBuffers[i]->getHandle();
        cameraBufferInfo.offset = 0;
        cameraBufferInfo.range = sizeof(CameraData); // can also use VK_WHOLE_SIZE in this case

        VkWriteDescriptorSet descriptorWriteUniformBuffer{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptorWriteUniformBuffer.dstSet = frameData.globalDescriptorSets[i]->getHandle();
        descriptorWriteUniformBuffer.dstBinding = 0;
        descriptorWriteUniformBuffer.dstArrayElement = 0;
        descriptorWriteUniformBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWriteUniformBuffer.descriptorCount = 1;
        descriptorWriteUniformBuffer.pBufferInfo = &cameraBufferInfo;
        descriptorWriteUniformBuffer.pImageInfo = nullptr; // Optional
        descriptorWriteUniformBuffer.pTexelBufferView = nullptr; // Optional

        // Object Descriptor Set
        VkDescriptorSetAllocateInfo objectDescriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        objectDescriptorSetAllocateInfo.descriptorPool = descriptorPool->getHandle();
        objectDescriptorSetAllocateInfo.descriptorSetCount = 1;
        objectDescriptorSetAllocateInfo.pSetLayouts = &objectDescriptorSetLayout->getHandle();
        frameData.objectDescriptorSets[i] = std::make_unique<DescriptorSet>(*device, objectDescriptorSetAllocateInfo);
        setDebugUtilsObjectName(device->getHandle(), frameData.objectDescriptorSets[i]->getHandle(), "objectDescriptorSet for frame #" + std::to_string(i));

        VkDescriptorBufferInfo objectBufferInfo{};
        objectBufferInfo.buffer = frameData.objectBuffers[i]->getHandle();
        objectBufferInfo.offset = 0;
        objectBufferInfo.range = sizeof(ObjInstance) * MAX_OBJECT_COUNT;

        VkWriteDescriptorSet objectWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        objectWrite.dstSet = frameData.objectDescriptorSets[i]->getHandle();
        objectWrite.dstBinding = 0;
        objectWrite.dstArrayElement = 0;
        objectWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        objectWrite.descriptorCount = 1;
        objectWrite.pBufferInfo = &objectBufferInfo;
        objectWrite.pImageInfo = nullptr; // Optional
        objectWrite.pTexelBufferView = nullptr; // Optional

        // Write descriptor sets
        std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{ descriptorWriteUniformBuffer, objectWrite };
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

    VkWriteDescriptorSet descriptorWriteCombinedImageSampler{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    descriptorWriteCombinedImageSampler.dstSet = textureDescriptorSet->getHandle();
    descriptorWriteCombinedImageSampler.dstBinding = 0;
    descriptorWriteCombinedImageSampler.dstArrayElement = 0;
    descriptorWriteCombinedImageSampler.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWriteCombinedImageSampler.descriptorCount = textureImageInfos.size();
    descriptorWriteCombinedImageSampler.pImageInfo = textureImageInfos.data();
    descriptorWriteCombinedImageSampler.pBufferInfo = nullptr; // Optional
    descriptorWriteCombinedImageSampler.pTexelBufferView = nullptr; // Optional

    std::array<VkWriteDescriptorSet, 1> writeDescriptorSets{ descriptorWriteCombinedImageSampler };
    vkUpdateDescriptorSets(device->getHandle(), to_u32(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void MainApp::createSemaphoreAndFencePools()
{
    semaphorePool = std::make_unique<SemaphorePool>(*device);
    fencePool = std::make_unique<FencePool>(*device);
}

void MainApp::loadTextureImages(const std::vector<std::string> &textureFiles)
{
    int x = 0;
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
        frameData.renderingFinishedSemaphores[i] = semaphorePool->requestSemaphore();
        frameData.inFlightFences[i] = fencePool->requestFence();
    }
}

void MainApp::loadModel(const std::string objFileName, glm::mat4 transform)
{
    const std::string modelPath = "../../assets/models/";
    const std::string filePath = modelPath + objFileName;

    std::shared_ptr<ObjModel> objModel = std::make_shared<ObjModel>();
    ObjLoader objLoader;
    objLoader.loadModel(filePath.c_str());

    // TODO: figure out if this is required
    // Converting the ambient, diffuse and specular values from srgb to linear
    //for (auto &m : objLoader.materials)
    //{
    //    m.ambient = glm::pow(m.ambient, glm::vec3(2.2f));
    //    m.diffuse = glm::pow(m.diffuse, glm::vec3(2.2f));
    //    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
    //}

    objModel->verticesCount = to_u32(objLoader.vertices.size());
    objModel->indicesCount = to_u32(objLoader.indices.size());

    createVertexBuffer(objModel, objLoader);
    createIndexBuffer(objModel, objLoader);
    createMaterialBuffer(objModel, objLoader);
    createMaterialIndicesBuffer(objModel, objLoader);

    std::string objNb = std::to_string(objModels.size());
    setDebugUtilsObjectName(device->getHandle(), objModel->vertexBuffer->getHandle(), (std::string("vertex_" + objNb).c_str()));
    setDebugUtilsObjectName(device->getHandle(), objModel->indexBuffer->getHandle(), (std::string("index_" + objNb).c_str()));
    setDebugUtilsObjectName(device->getHandle(), objModel->materialsBuffer->getHandle(), (std::string("mat_" + objNb).c_str()));
    setDebugUtilsObjectName(device->getHandle(), objModel->materialsIndexBuffer->getHandle(), (std::string("matIdx_" + objNb).c_str()));

    uint64_t txtOffset = static_cast<uint64_t>(textures.size());
    loadTextureImages(objLoader.textures);

    ObjInstance instance;
    instance.transform = transform;
    instance.transformIT = glm::transpose(glm::inverse(transform));
    instance.objIndex = static_cast<uint64_t>(objModels.size());
    instance.textureOffset = txtOffset;

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR };
    bufferDeviceAddressInfo.buffer = objModel->vertexBuffer->getHandle();
    instance.vertices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModel->indexBuffer->getHandle();
    instance.indices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModel->materialsBuffer->getHandle();
    instance.materials = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = objModel->materialsIndexBuffer->getHandle();
    instance.materialIndices = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

    objInstances.push_back(instance);
    objModels[objFileName] = objModel;
}

void MainApp::loadModels()
{
    loadModel("monkey_smooth.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3(1, 0, 0)));
    loadModel("lost_empire.obj", glm::translate(glm::mat4{ 1.0 }, glm::vec3{ 5,-10,0 }));
}

void MainApp::createScene()
{
    RenderObject monkey;
    monkey.objModel = getObjModel("monkey_smooth.obj");
    monkey.pipelineData = getPipelineData("defaultmesh");
    renderables.push_back(monkey);

    RenderObject map;
    map.objModel = getObjModel("lost_empire.obj");
    map.pipelineData = getPipelineData("texturedmesh");
    renderables.push_back(map);
}

std::shared_ptr<PipelineData> MainApp::getPipelineData(const std::string &name)
{
    auto it = pipelineDataMap.find(name);
    if (it == pipelineDataMap.end())
    {
        return nullptr;
    }

    return it->second;

}

std::shared_ptr<ObjModel> MainApp::getObjModel(const std::string &name)
{
    auto it = objModels.find(name);
    if (it == objModels.end())
    {
        return nullptr;
    }

    return it->second;
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
    initInfo.QueueFamily = device->getOptimalGraphicsQueue().getFamilyIndex();
    initInfo.Queue = graphicsQueue;
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

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    // Clear font data on CPU
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void MainApp::createOutputImageAndImageView()
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
        
        std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*frameData.commandPools[currentFrame], VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
        frameData.outputImages[i]->transitionImageLayout(*commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
        commandBuffer->end();

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer->getHandle();
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
    }
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
BlasInput MainApp::objectToVkGeometryKHR(size_t renderableIndex)
{
    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    bufferDeviceAddressInfo.buffer = renderables[renderableIndex].objModel->vertexBuffer->getHandle();
    // We can take advantage of the fact that position is the first member of Vertex
    VkDeviceAddress vertexAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = renderables[renderableIndex].objModel->indexBuffer->getHandle();
    VkDeviceAddress indexAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

    uint32_t maxPrimitiveCount = renderables[renderableIndex].objModel->indicesCount / 3;

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
    triangles.maxVertex = renderables[renderableIndex].objModel->verticesCount;

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
    // BLAS - Storing each primitive in a geometry
    std::vector<BlasInput> allBlas;
    allBlas.reserve(renderables.size());
    for (uint32_t i = 0; i < renderables.size(); ++i)
    {
        auto blas = objectToVkGeometryKHR(i);

        // We could add more geometry in each BLAS, but we add only one for now
        allBlas.emplace_back(blas);
    }
    // TODO BUILD THE BLASENTRY DIRECTLY RATHER THAN DOING THIS
    buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
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


void MainApp::buildBlas(const std::vector<BlasInput> &input, VkBuildAccelerationStructureFlagsKHR flags)
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
        buildInfos[idx].flags = flags;
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
        m_blas[idx].flags = flags;
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
    bool doCompaction = (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
        == VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

    LOGI("Compaction is" + doCompaction ? "true" : "false");

    // Allocate a query pool for storing the needed size for every BLAS compaction.
    VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpci.queryCount = nbBlas;
    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    VkQueryPool queryPool;
    vkCreateQueryPool(device->getHandle(), &qpci, nullptr, &queryPool);
    vkResetQueryPool(device->getHandle(), queryPool, 0, nbBlas);

    // Allocate a command pool for queue of given queue index.
    // To avoid timeout, record and submit one command buffer per AS build.
    CommandPool asCommandPool(*device, device->getOptimalGraphicsQueue().getFamilyIndex(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
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
    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(graphicsQueue);
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
            // LOGI("Reducing %i, from %d to %d \n", i, originalSizes[i], compactSizes[i]);
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
        VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
        vkQueueWaitIdle(graphicsQueue);

        LOGI(" RT BLAS: reducing from: %u to: %u = %u (%2.2f%s smaller) \n", statTotalOriSize, statTotalCompactSize,
            statTotalOriSize - statTotalCompactSize,
            (statTotalOriSize - statTotalCompactSize) / float(statTotalOriSize) * 100.f, "%%");
    }

    vkDestroyQueryPool(device->getHandle(), queryPool, nullptr);
}

void MainApp::createTopLevelAS()
{
    std::vector<BlasInstance> tlas;
    tlas.reserve(objInstances.size());
    for (uint32_t i = 0; i < to_u32(objInstances.size()); i++)
    {
        BlasInstance rayInst;
        rayInst.transform = objInstances[i].transform;  // Position of the instance
        rayInst.instanceCustomId = i;                           // gl_InstanceCustomIndexEXT
        rayInst.blasId = i; // TODO: is this correct
        rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
        rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        tlas.emplace_back(rayInst);
    }
    buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

void MainApp::buildTlas(
    const std::vector<BlasInstance> &instances,
    VkBuildAccelerationStructureFlagsKHR flags,
    bool update
) {
    // Cannot call buildTlas twice except to update.
    if (m_tlas != nullptr && !update)
    {
        LOGEANDABORT("Cannot call buildTlas twice except to update");
    }

    if (m_tlas == nullptr)
    {
        m_tlas = std::make_unique<Tlas>();
    }

    CommandPool asCommandPool(*device, device->getOptimalGraphicsQueue().getFamilyIndex(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::shared_ptr<CommandBuffer> cmdBuf = std::make_shared<CommandBuffer>(asCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

    m_tlas->flags = flags;

    // Convert array of our Instances to an array native Vulkan instances.
    std::vector<VkAccelerationStructureInstanceKHR> geometryInstances;
    geometryInstances.reserve(instances.size());
    for (const auto &inst : instances)
    {
        geometryInstances.push_back(instanceToVkGeometryInstanceKHR(inst));
    }

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    VkDeviceSize instanceDescsSizeInBytes = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    // Allocate the instance buffer and copy its contents from host to device memory
    if (update)
    {
        m_instBuffer.reset();
    }

    VkDeviceSize bufferSize{ sizeof(VkAccelerationStructureInstanceKHR) * geometryInstances.size() };
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
    memcpy(mappedData, geometryInstances.data(), static_cast<size_t>(bufferSize));
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
    buildInfo.flags = flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &topASGeometry;
    buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    uint32_t                                 count = (uint32_t)instances.size();
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkGetAccelerationStructureBuildSizesKHR(device->getHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &count, &sizeInfo);


    // Create TLAS
    if (update == false)
    {
        VkAccelerationStructureCreateInfoKHR accellerationStructureCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        accellerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        accellerationStructureCreateInfo.size = sizeInfo.accelerationStructureSize;

        m_tlas->as = createAcceleration(accellerationStructureCreateInfo);
        setDebugUtilsObjectName(device->getHandle(), m_tlas->as->accel, "TLAS");
    }

    // Allocate the scratch memory
    VkBufferCreateInfo bufferCreateInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferCreateInfo.size = sizeInfo.buildScratchSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    //VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    std::unique_ptr<Buffer> scratchBuffer = std::make_unique<Buffer>(*device, bufferCreateInfo, memoryInfo);
    bufferDeviceAddressInfo.buffer = scratchBuffer->getHandle();
    VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(device->getHandle(), &bufferDeviceAddressInfo);

    // Update build information
    buildInfo.srcAccelerationStructure = update ? m_tlas->as->accel : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = m_tlas->as->accel;
    buildInfo.scratchData.deviceAddress = scratchAddress;


    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{ static_cast<uint32_t>(instances.size()), 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR *pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    vkCmdBuildAccelerationStructuresKHR(cmdBuf->getHandle(), 1, &buildInfo, &pBuildOffsetInfo);

    // submitandwaitidle
    cmdBuf->end();
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pCommandBuffers = &cmdBuf->getHandle();
    submitInfo.commandBufferCount = 1u;
    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(graphicsQueue);
}

VkAccelerationStructureInstanceKHR MainApp::instanceToVkGeometryInstanceKHR(const BlasInstance &instance)
{
    assert(size_t(instance.blasId) < m_blas.size());
    BlasEntry &blas{ m_blas[instance.blasId] };

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    addressInfo.accelerationStructure = blas.as->accel;
    VkDeviceAddress blasAddress = vkGetAccelerationStructureDeviceAddressKHR(device->getHandle(), &addressInfo);

    VkAccelerationStructureInstanceKHR gInst{};
    // The matrices for the instance transforms are row-major, instead of
    // column-major in the rest of the application
    glm::mat4 transp = glm::transpose(instance.transform);
    // The gInst.transform value only contains 12 values, corresponding to a 4x3
    // matrix, hence saving the last row that is anyway always (0,0,0,1). Since
    // the matrix is row-major, we simply copy the first 12 values of the
    // original 4x4 matrix
    memcpy(&gInst.transform, &transp, sizeof(gInst.transform));
    gInst.instanceCustomIndex = instance.instanceCustomId;
    gInst.mask = instance.mask;
    gInst.instanceShaderBindingTableRecordOffset = instance.hitGroupId;
    gInst.flags = instance.flags;
    gInst.accelerationStructureReference = blasAddress;
    return gInst;
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

        VkWriteDescriptorSet descriptorWriteAccelerationStructure{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptorWriteAccelerationStructure.dstSet = frameData.rtDescriptorSets[i]->getHandle();
        descriptorWriteAccelerationStructure.dstBinding = 0;
        descriptorWriteAccelerationStructure.dstArrayElement = 0;
        descriptorWriteAccelerationStructure.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        descriptorWriteAccelerationStructure.descriptorCount = 1;
        descriptorWriteAccelerationStructure.pImageInfo = nullptr; // Optional
        descriptorWriteAccelerationStructure.pBufferInfo = nullptr; // Optional
        descriptorWriteAccelerationStructure.pTexelBufferView = nullptr; // Optional
        descriptorWriteAccelerationStructure.pNext = &descASInfo;


        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        outputImageInfo.imageView = frameData.outputImageViews[i]->getHandle();
        outputImageInfo.sampler = {};

        VkWriteDescriptorSet descriptorWriteStorageImage{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptorWriteStorageImage.dstSet = frameData.rtDescriptorSets[i]->getHandle();
        descriptorWriteStorageImage.dstBinding = 1;
        descriptorWriteStorageImage.dstArrayElement = 0;
        descriptorWriteStorageImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWriteStorageImage.descriptorCount = 1;
        descriptorWriteStorageImage.pImageInfo = &outputImageInfo;
        descriptorWriteStorageImage.pBufferInfo = nullptr;
        descriptorWriteStorageImage.pTexelBufferView = nullptr; // Optional

        std::vector<VkWriteDescriptorSet> writes;
        writes.push_back(descriptorWriteAccelerationStructure);
        writes.push_back(descriptorWriteStorageImage);
        vkUpdateDescriptorSets(device->getHandle(), to_u32(writes.size()), writes.data(), 0, nullptr);
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
        // (1) Output buffer
        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        outputImageInfo.imageView = frameData.outputImageViews[i]->getHandle();
        outputImageInfo.sampler = {};
        VkWriteDescriptorSet descriptorWriteCombinedImageSampler{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptorWriteCombinedImageSampler.dstSet = frameData.rtDescriptorSets[currentFrame]->getHandle();
        descriptorWriteCombinedImageSampler.dstBinding = 1;
        descriptorWriteCombinedImageSampler.dstArrayElement = 0;
        descriptorWriteCombinedImageSampler.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWriteCombinedImageSampler.descriptorCount = 1;
        descriptorWriteCombinedImageSampler.pImageInfo = &outputImageInfo;
        descriptorWriteCombinedImageSampler.pBufferInfo = nullptr;
        descriptorWriteCombinedImageSampler.pTexelBufferView = nullptr; // Optional

        vkUpdateDescriptorSets(device->getHandle(), 1, &descriptorWriteCombinedImageSampler, 0, nullptr);
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

    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_RAYGEN_BIT_KHR, rayGenShader);
    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_MISS_BIT_KHR, rayMissShader);
    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_MISS_BIT_KHR, rayShadowMissShader);
    raytracingShaderModules.emplace_back(*device, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rayClosestHitShader);

    // All stages
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo              shaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    // Since we only need the shadermodule during the creation of the pipeline, we don't keep the information in the shader module class and delete at the end
    VkShaderModuleCreateInfo shaderModuleCreateInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    // Raygen
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[0].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[0].getShaderSource().getData().data());
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[0].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[0].getStage();
    stages[eRaygen] = shaderStageCreateInfo;
    // Miss
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[1].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[1].getShaderSource().getData().data());
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[1].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[1].getStage();
    stages[eMiss] = shaderStageCreateInfo;
    // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[2].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[2].getShaderSource().getData().data());
    VK_CHECK(vkCreateShaderModule(device->getHandle(), &shaderModuleCreateInfo, nullptr, &shaderStageCreateInfo.module));
    shaderStageCreateInfo.pName = raytracingShaderModules[2].getEntryPoint().c_str();
    shaderStageCreateInfo.stage = raytracingShaderModules[2].getStage();
    stages[eMissShadow] = shaderStageCreateInfo;
    // Hit Group - Closest Hit
    shaderModuleCreateInfo.codeSize = raytracingShaderModules[3].getShaderSource().getData().size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(raytracingShaderModules[3].getShaderSource().getData().data());
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
                                     0, sizeof(RtPushConstant) };


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

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void MainApp::raytrace(const uint32_t &swapchainImageIndex)
{
    debugUtilBeginLabel(frameData.commandBuffers[currentFrame]->getHandle(), "Raytrace");

    std::vector<VkDescriptorSet> descSets{
        frameData.rtDescriptorSets[currentFrame]->getHandle(), 
        frameData.globalDescriptorSets[currentFrame]->getHandle(),
        frameData.objectDescriptorSets[currentFrame]->getHandle(),
        textureDescriptorSet->getHandle()
    };
    vkCmdBindPipeline(frameData.commandBuffers[currentFrame]->getHandle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdBindDescriptorSets(frameData.commandBuffers[currentFrame]->getHandle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
        to_u32(descSets.size()), descSets.data(), 0, nullptr);
    vkCmdPushConstants(frameData.commandBuffers[currentFrame]->getHandle(), m_rtPipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
        0, sizeof(RtPushConstant), &m_rtPushConstants);


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

    vkCmdTraceRaysKHR(frameData.commandBuffers[currentFrame]->getHandle(), &strideAddresses[0], &strideAddresses[1], &strideAddresses[2], &strideAddresses[3],
        swapchain->getProperties().imageExtent.width, swapchain->getProperties().imageExtent.height, 1);

    debugUtilEndLabel(frameData.commandBuffers[currentFrame]->getHandle());
}

} // namespace vulkr

int main()
{
    vulkr::Platform platform;
    std::unique_ptr<vulkr::MainApp> app = std::make_unique<vulkr::MainApp>(platform, "Vulkan App");

    platform.initialize(std::move(app));
    platform.prepareApplication();

    platform.runMainProcessingLoop();
    platform.terminate();

    return EXIT_SUCCESS;
}

