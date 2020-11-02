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

#include "core/instance.h"
#include "core/device.h"
#include "core/swapchain.h"
#include "core/image_view.h"
#include "core/render_pass.h"
#include "rendering/subpass.h"
#include "rendering/shader_module.h"
#include "rendering/pipeline_state.h"
#include "core/pipeline_layout.h"
#include "core/pipeline.h"
#include "core/framebuffer.h"
#include "core/queue.h"
#include "core/command_pool.h"
#include "core/command_buffer.h"
#include "core/buffer.h"

#include "common/semaphore_pool.h"
#include "common/fence_pool.h"

#include "platform/application.h"

#include <glm/glm.hpp>

namespace vulkr
{

class MainApp : public Application
{
public:
    MainApp(Platform &platform, std::string name);

    ~MainApp();

    virtual void prepare() override;

    virtual void update();

    virtual void recreateSwapchain() override;
private:
    std::unique_ptr<Instance> instance{ nullptr };
    
    VkSurfaceKHR surface{ VK_NULL_HANDLE };

    std::unique_ptr<Device> device{ nullptr };

    std::unique_ptr<Swapchain> swapchain{ nullptr };

    std::vector<std::unique_ptr<ImageView>> swapChainImageViews;

    std::vector<VkAttachmentReference> inputAttachments;
    std::vector<VkAttachmentReference> colorAttachments;
    std::vector<VkAttachmentReference> resolveAttachments;
    std::vector<VkAttachmentReference> depthStencilAttachments;
    std::vector<uint32_t> preserveAttachments;

    std::vector<Subpass> subpasses;
    std::unique_ptr<RenderPass> renderPass{ nullptr };

    std::vector<ShaderModule> shaderModules;
    std::unique_ptr<PipelineState> pipelineState{ nullptr };
    std::unique_ptr<GraphicsPipeline> pipeline{ nullptr };

    std::vector<std::unique_ptr<Framebuffer>> swapchainFramebuffers;

    std::unique_ptr<CommandPool> commandPool{ nullptr };

    std::vector<std::unique_ptr<CommandBuffer>> commandBuffers;

    std::unique_ptr<SemaphorePool> semaphorePool;
    std::unique_ptr<FencePool> fencePool;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    VkQueue graphicsQueue{ VK_NULL_HANDLE };
    VkQueue presentQueue{ VK_NULL_HANDLE };

    std::unique_ptr<Buffer> vertexBuffer{ nullptr };

    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    size_t currentFrame = 0;

    const std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    struct Vertex {
        glm::vec2 pos;
        glm::vec3 color;
    };

    const std::vector<Vertex> vertices = {
        {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
    };

    void cleanupSwapchain();

    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createVertexBuffer();
    void copyBuffer(Buffer &srcBuffer, Buffer &dstBuffer, VkDeviceSize size);
    void createCommandBuffers();
    void createSemaphoreAndFencePools();
    void setupSynchronizationObjects();
};

} // namespace vulkr