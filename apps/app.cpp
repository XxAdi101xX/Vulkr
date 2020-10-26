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
#include "core/image.h"

namespace vulkr
{

MainApp::MainApp(Platform& platform, std::string name) : Application{ platform, name } {}

 MainApp::~MainApp()
 {
     device->waitIdle();

     semaphorePool.reset();
     fencePool.reset();

     for (uint32_t i = 0; i < commandBuffers.size(); ++i) {
         commandBuffers[i].reset();
     }

     commandPool.reset();

     for (uint32_t i = 0; i < swapchainFramebuffers.size(); ++i) {
         swapchainFramebuffers[i].reset();
     }

     pipeline.reset();

     pipelineState.reset(); // destroys pipeline layout as well

     renderPass.reset();

     for (uint32_t i = 0; i < swapChainImageViews.size(); ++i) {
         swapChainImageViews[i].reset();
     }

     swapchain.reset();

     device.reset();

     // Must destroy surface before instance
	 if (surface != VK_NULL_HANDLE)
	 {
		 vkDestroySurfaceKHR(instance->getHandle(), surface, nullptr);
	 }

	 instance.reset();
 }

void MainApp::prepare()
{
    Application::prepare();

    instance = std::make_unique<Instance>(getName());

    // create surface 
    platform.createSurface(instance->getHandle());
    surface = platform.getSurface();

    // create device
    device = std::make_unique<Device>(std::move(instance->getSuitablePhysicalDevice()), surface, deviceExtensions);

    // create swapchain
    const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
    swapchain = std::make_unique<Swapchain>(*device, surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags);

    // create swapchain image views
    swapChainImageViews.reserve(swapchain->getImages().size());
    for (uint32_t i = 0; i < swapchain->getImages().size(); ++i) {
        swapChainImageViews.emplace_back(std::make_unique<ImageView>(*(swapchain->getImages()[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, swapchain->getProperties().surfaceFormat.format));
    }

    // create attachments
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

    // Setup pipeline
    VertexInputState vertexInputState{};

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
    rasterizationState.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
        std::make_unique<PipelineLayout>(*device, shaderModules),
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

    // Create swapchain framebuffers
    swapchainFramebuffers.reserve(swapchain->getImages().size());
    for (uint32_t i = 0; i < swapchain->getImages().size(); ++i) {
        std::vector<VkImageView> attachments { swapChainImageViews[i]->getHandle() };

        swapchainFramebuffers.emplace_back(std::make_unique<Framebuffer>(*device, *swapchain, *renderPass, attachments));
    }

    // Create command pool
    commandPool = std::make_unique<CommandPool>(*device, device->getOptimalGraphicsQueue().getFamilyIndex(), 0u);

    // Create command buffers
    std::vector<VkClearValue> clearValues;
    VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues.emplace_back(clearColor);

    commandBuffers.reserve(swapchainFramebuffers.size());
    for (uint32_t i = 0; i < swapchainFramebuffers.size(); ++i) {
        commandBuffers.emplace_back(std::make_unique<CommandBuffer>(*commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, *renderPass, *(swapchainFramebuffers[i])));

        commandBuffers[i]->begin(0u, nullptr);
        
        const std::vector<std::unique_ptr<Subpass>> subpasses; // TODO not used
        commandBuffers[i]->beginRenderPass(clearValues, subpasses, swapchain->getProperties().imageExtent, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i]->getHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getHandle()); // TODO: put this in command buffer class?

        vkCmdDraw(commandBuffers[i]->getHandle(), 3, 1, 0, 0); // TODO do we put this in commmand buffer class

        commandBuffers[i]->endRenderPass();

        commandBuffers[i]->end();
    }

    // Create semaphore and fence pools
    semaphorePool = std::make_unique<SemaphorePool>(*device);
    fencePool = std::make_unique<FencePool>(*device);

    imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapchain->getImages().size(), VK_NULL_HANDLE);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        imageAvailableSemaphores.push_back(semaphorePool->requestSemaphore());
        renderFinishedSemaphores.push_back(semaphorePool->requestSemaphore());
        inFlightFences.push_back(fencePool->requestFence());
    }

    graphicsQueue = device->getOptimalGraphicsQueue().getHandle();

    if (device->getOptimalGraphicsQueue().canSupportPresentation())
    {
        presentQueue = graphicsQueue;
    }
    else
    {
        presentQueue = device->getQueueByPresentation().getHandle();
    }

}

void MainApp::update()
{
    fencePool->wait(&inFlightFences[currentFrame]);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(device->getHandle(), swapchain->getHandle(), UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        fencePool->wait(&imagesInFlight[imageIndex]);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex]->getHandle();

    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    fencePool->reset(&inFlightFences[currentFrame]);

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapchains[] = { swapchain->getHandle() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;

    presentInfo.pImageIndices = &imageIndex;

    VK_CHECK(vkQueuePresentKHR(presentQueue, &presentInfo));

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
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

