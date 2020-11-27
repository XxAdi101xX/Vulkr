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

//#include <iostream>

#include "app.h"
#include "platform/platform.h"

namespace vulkr
{

MainApp::MainApp(Platform& platform, std::string name) : Application{ platform, name } {}

 MainApp::~MainApp()
 {
     device->waitIdle();

     semaphorePool.reset();
     fencePool.reset();

     cleanupSwapchain();

     descriptorSetLayout.reset();

     indexBuffer.reset();

     vertexBuffer.reset();

     commandPool.reset();

     device.reset();

     // Must destroy surface before instance
	 if (surface != VK_NULL_HANDLE)
	 {
		 vkDestroySurfaceKHR(instance->getHandle(), surface, nullptr);
	 }

	 instance.reset();
 }

 void MainApp::cleanupSwapchain()
 {
     for (uint32_t i = 0; i < commandBuffers.size(); ++i) {
         commandBuffers[i].reset();
     }
     commandBuffers.clear();

     for (uint32_t i = 0; i < swapchainFramebuffers.size(); ++i) {
         swapchainFramebuffers[i].reset();
     }
     swapchainFramebuffers.clear();

     shaderModules.clear();
     pipeline.reset();

     pipelineState.reset(); // destroys pipeline layout as well

     renderPass.reset();
     subpasses.clear();

     for (uint32_t i = 0; i < swapChainImageViews.size(); ++i) {
         swapChainImageViews[i].reset();
     }
     swapChainImageViews.clear();
     inputAttachments.clear();
     colorAttachments.clear();
     resolveAttachments.clear();
     depthStencilAttachments.clear();
     preserveAttachments.clear();

     swapchain.reset();

     for (uint32_t i = 0; i < uniformBuffers.size(); ++i) {
         uniformBuffers[i].reset();
     }
     uniformBuffers.clear();

     descriptorSets.clear();
     descriptorPool.reset();
 }

void MainApp::prepare()
{
    Application::prepare();

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
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSemaphoreAndFencePools();
    setupSynchronizationObjects();
    setupTimer();
}

void MainApp::update()
{
    fencePool->wait(&inFlightFences[currentFrame]);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device->getHandle(), swapchain->getHandle(), std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        LOGEANDABORT("Failed to acquire swap chain image");
    }

    updateUniformBuffer(imageIndex);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        fencePool->wait(&imagesInFlight[imageIndex]);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];
    
    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

    std::vector<VkSemaphore> waitSemaphores{ imageAvailableSemaphores[currentFrame] };
    std::vector<VkPipelineStageFlags> waitStages{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = waitSemaphores.size();
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex]->getHandle();

    std::vector<VkSemaphore> signalSemaphores{ renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = signalSemaphores.size();
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    fencePool->reset(&inFlightFences[currentFrame]);

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores.data();

    std::vector<VkSwapchainKHR> swapchains{ swapchain->getHandle() };
    presentInfo.swapchainCount = swapchains.size();
    presentInfo.pSwapchains = swapchains.data();

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        recreateSwapchain();
    }
    else if (result != VK_SUCCESS)
    {
        LOGEANDABORT("Failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void MainApp::recreateSwapchain()
{
    device->waitIdle();
    cleanupSwapchain();

    createSwapchain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

void MainApp::updateUniformBuffer(uint32_t currentImage)
{
    float elapsedTime = static_cast<float>(drawingTimer->elapsed<Timer::Seconds>());

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), elapsedTime * glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapchain->getProperties().imageExtent.width / (float)swapchain->getProperties().imageExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    void *mappedData = uniformBuffers[currentImage]->map();
    memcpy(mappedData, &ubo, sizeof(ubo));
    uniformBuffers[currentImage]->unmap();
}

void MainApp::createInstance()
{
    instance = std::make_unique<Instance>(getName());
}

void MainApp::createSurface()
{
    platform.createSurface(instance->getHandle());
    surface = platform.getSurface();
}

void MainApp::createDevice()
{
    device = std::make_unique<Device>(std::move(instance->getSuitablePhysicalDevice()), surface, deviceExtensions);
}

void MainApp::createSwapchain()
{
    const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
    swapchain = std::make_unique<Swapchain>(*device, surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags);
}

void MainApp::createImageViews()
{
    swapChainImageViews.reserve(swapchain->getImages().size());
    for (uint32_t i = 0; i < swapchain->getImages().size(); ++i) {
        swapChainImageViews.emplace_back(std::make_unique<ImageView>(*(swapchain->getImages()[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, swapchain->getProperties().surfaceFormat.format));
    }
}
void MainApp::createRenderPass()
{
    std::vector<Attachment> attachments;
    Attachment colorAttachment{};
    colorAttachment.format = swapchain->getProperties().surfaceFormat.format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments.push_back(colorAttachment);

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments.push_back(colorAttachmentRef);

    subpasses.emplace_back(inputAttachments, colorAttachments, resolveAttachments, depthStencilAttachments, preserveAttachments, VK_PIPELINE_BIND_POINT_GRAPHICS);

    VkSubpassDependency subpassDependency{};
    subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpassDependency.dstSubpass = 0;
    subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependency.srcAccessMask = 0;
    subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    std::vector<VkSubpassDependency> subpassDependencies{ subpassDependency };
    renderPass = std::make_unique<RenderPass>(*device, attachments, subpasses, subpassDependencies);
}

void MainApp::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings{ uboLayoutBinding };

    descriptorSetLayout = std::make_unique<DescriptorSetLayout>(*device, descriptorSetLayoutBindings);
}

void MainApp::createGraphicsPipeline()
{
    // Setup pipeline
    VertexInputState vertexInputState{};
    vertexInputState.bindingDescriptions.reserve(1);
    vertexInputState.attributeDescriptions.reserve(2);

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    vertexInputState.bindingDescriptions.emplace_back(bindingDescription);

    VkVertexInputAttributeDescription posAttributeDescription;
    posAttributeDescription.binding = 0;
    posAttributeDescription.location = 0;
    posAttributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
    posAttributeDescription.offset = offsetof(Vertex, pos);
    VkVertexInputAttributeDescription colorAttributeDescription;
    colorAttributeDescription.binding = 0;
    colorAttributeDescription.location = 1;
    colorAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttributeDescription.offset = offsetof(Vertex, color);

    vertexInputState.attributeDescriptions.emplace_back(posAttributeDescription);
    vertexInputState.attributeDescriptions.emplace_back(colorAttributeDescription);

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

    shaderModules.emplace_back(*device, VK_SHADER_STAGE_VERTEX_BIT, std::make_unique<ShaderSource>("../../../../src/shaders/vert.spv"));
    shaderModules.emplace_back(*device, VK_SHADER_STAGE_FRAGMENT_BIT, std::make_unique<ShaderSource>("../../../../src/shaders/frag.spv"));

    pipelineState = std::make_unique<PipelineState>(
        std::make_unique<PipelineLayout>(*device, shaderModules, *descriptorSetLayout),
        *renderPass,
        vertexInputState,
        inputAssemblyState,
        viewportState,
        rasterizationState,
        multisampleState,
        depthStencilState,
        colorBlendState
    );

    pipeline = std::make_unique<GraphicsPipeline>(*device, *pipelineState, nullptr);
}

void MainApp::createFramebuffers()
{
    swapchainFramebuffers.reserve(swapchain->getImages().size());
    for (uint32_t i = 0; i < swapchain->getImages().size(); ++i)
    {
        std::vector<VkImageView> attachments{ swapChainImageViews[i]->getHandle() };

        swapchainFramebuffers.emplace_back(std::make_unique<Framebuffer>(*device, *swapchain, *renderPass, attachments));
    }
}

void MainApp::createCommandPool()
{
    commandPool = std::make_unique<CommandPool>(*device, device->getOptimalGraphicsQueue().getFamilyIndex(), 0u);
}

void MainApp::copyBuffer(Buffer& srcBuffer, Buffer& dstBuffer, VkDeviceSize size)
{
    // TODO: move this elsewhere?
    std::unique_ptr<CommandBuffer> commandBuffer = std::make_unique<CommandBuffer>(*commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

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

void MainApp::createVertexBuffer()
{
    VkDeviceSize bufferSize{ sizeof(vertices[0]) * vertices.size() };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vertexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, vertices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer->unmap();
    copyBuffer(*stagingBuffer, *vertexBuffer, bufferSize);
}

void MainApp::createIndexBuffer()
{
    VkDeviceSize bufferSize{ sizeof(indices[0]) * indices.size() };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    indexBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

    void *mappedData = stagingBuffer->map();
    memcpy(mappedData, indices.data(), (size_t)bufferSize);
    stagingBuffer->unmap();
    copyBuffer(*stagingBuffer, *indexBuffer, bufferSize);
}

void MainApp::createUniformBuffers()
{
    VkDeviceSize bufferSize{ sizeof(UniformBufferObject) };

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memoryInfo{};
    memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    uniformBuffers.reserve(swapChainImageViews.size());
    for (uint32_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        uniformBuffers.emplace_back(std::make_unique<Buffer>(*device, bufferInfo, memoryInfo));
    }
}

void MainApp::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes{};
    poolSizes.resize(1);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = to_u32(swapChainImageViews.size());

    descriptorPool = std::make_unique<DescriptorPool>(*device, *descriptorSetLayout, poolSizes, to_u32(swapChainImageViews.size()));
}

void MainApp::createDescriptorSets()
{
    descriptorSets.reserve(swapChainImageViews.size());
    for (uint32_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        descriptorSets.emplace_back(std::make_unique<DescriptorSet>(*device, *descriptorSetLayout, *descriptorPool));

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i]->getHandle();
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject); // can also use VK_WHOLE_SIZE in this case

        VkWriteDescriptorSet descriptorWriteUniformBuffer{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptorWriteUniformBuffer.dstSet = descriptorSets[i]->getHandle();
        descriptorWriteUniformBuffer.dstBinding = 0;
        descriptorWriteUniformBuffer.dstArrayElement = 0;
        descriptorWriteUniformBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWriteUniformBuffer.descriptorCount = 1;
        descriptorWriteUniformBuffer.pBufferInfo = &bufferInfo;
        descriptorWriteUniformBuffer.pImageInfo = nullptr; // Optional
        descriptorWriteUniformBuffer.pTexelBufferView = nullptr; // Optional

        std::vector<VkWriteDescriptorSet> writeDescriptorSets{ descriptorWriteUniformBuffer };

        descriptorSets[i]->update(writeDescriptorSets);
    }
}

// TODO allocate command buffers using command_pool's requestCommandBuffer function??
void MainApp::createCommandBuffers()
{
    std::vector<VkClearValue> clearValues;
    VkClearValue clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues.emplace_back(clearColor);

    commandBuffers.reserve(swapchainFramebuffers.size());
    for (uint32_t i = 0; i < swapchainFramebuffers.size(); ++i) {
        commandBuffers.emplace_back(std::make_unique<CommandBuffer>(*commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY));

        commandBuffers[i]->begin(0u, nullptr);

        commandBuffers[i]->beginRenderPass(*renderPass, *(swapchainFramebuffers[i]), swapchain->getProperties().imageExtent, clearValues, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getHandle()); // TODO: put this in command buffer class?

        VkBuffer vertexBuffers[] = { vertexBuffer->getHandle() };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i]->getHandle(), 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffers[i]->getHandle(), indexBuffer->getHandle(), 0, VK_INDEX_TYPE_UINT16);

        vkCmdBindDescriptorSets(commandBuffers[i]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineState->getPipelineLayout().getHandle(), 0, 1, &(descriptorSets[i]->getHandle()), 0, nullptr);

        vkCmdDrawIndexed(commandBuffers[i]->getHandle(), static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffers[i]->endRenderPass();

        commandBuffers[i]->end();
    }
}

void MainApp::createSemaphoreAndFencePools()
{
    semaphorePool = std::make_unique<SemaphorePool>(*device);
    fencePool = std::make_unique<FencePool>(*device);
}

void MainApp::setupSynchronizationObjects()
{
    imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        imageAvailableSemaphores.push_back(semaphorePool->requestSemaphore());
        renderFinishedSemaphores.push_back(semaphorePool->requestSemaphore());
        inFlightFences.push_back(fencePool->requestFence());
    }
}

void MainApp::setupTimer()
{
    drawingTimer = std::make_unique<Timer>();
    drawingTimer->start();
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

