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
#include "core/descriptor_set_layout.h"
#include "core/descriptor_pool.h"
#include "core/descriptor_set.h"
#include "rendering/camera_controller.h"
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
#include "core/image.h"
#include "core/sampler.h"

#include "common/semaphore_pool.h"
#include "common/fence_pool.h"
#include "common/helpers.h"
#include "common/timer.h"

#include "platform/application.h"
#include "platform/input_event.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // don't use the OpenGL default depth range of -1.0 to 1.0 and use 0.0 to 1.0
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 textureCoordinate;

    bool operator==(const Vertex &other) const {
    return position == other.position && color == other.color &&
            textureCoordinate == other.textureCoordinate;
    }
};

namespace std {
    template <> struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const {
        return ((hash<glm::vec3>()(vertex.position) ^
                    (hash<glm::vec3>()(vertex.color) << 1)) >>
                1) ^
                (hash<glm::vec2>()(vertex.textureCoordinate) << 1);
        }
    };
} // namespace std

namespace vulkr
{

class MainApp : public Application
{
public:
    MainApp(Platform &platform, std::string name);

    ~MainApp();

    virtual void prepare() override;

    virtual void update() override;

    virtual void recreateSwapchain() override;

    virtual void handleInputEvents(const InputEvent& inputEvent) override;
private:
    struct UniformBufferObject
    {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

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
    std::unique_ptr<DescriptorSetLayout> descriptorSetLayout{ nullptr };
    std::vector<ShaderModule> shaderModules;
    std::unique_ptr<PipelineState> pipelineState{ nullptr };
    std::unique_ptr<GraphicsPipeline> pipeline{ nullptr };

    std::vector<std::unique_ptr<Framebuffer>> swapchainFramebuffers;
    std::unique_ptr<CommandPool> commandPool{ nullptr };
    std::vector<std::unique_ptr<CommandBuffer>> commandBuffers;

    std::unique_ptr<Image> depthImage{ nullptr };
    std::unique_ptr<ImageView> depthImageView{ nullptr };
    std::unique_ptr<Image> textureImage{ nullptr };
    std::unique_ptr<ImageView> textureImageView{ nullptr };
    std::unique_ptr<Sampler> textureSampler{ nullptr };
    std::unique_ptr<Buffer> vertexBuffer{ nullptr };
    std::unique_ptr<Buffer> indexBuffer{ nullptr };
    std::vector<std::unique_ptr<Buffer>> uniformBuffers;

    std::unique_ptr<DescriptorPool> descriptorPool;
    std::vector<std::unique_ptr<DescriptorSet>> descriptorSets;

    std::unique_ptr<SemaphorePool> semaphorePool;
    std::unique_ptr<FencePool> fencePool;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    std::unique_ptr<CameraController> cameraController;

    const std::string MODEL_PATH = "../../../assets/models/viking_room.obj";
    const std::string TEXTURE_PATH = "../../../assets/textures/viking_room.png";

    VkQueue graphicsQueue{ VK_NULL_HANDLE };
    VkQueue presentQueue{ VK_NULL_HANDLE };

    std::unique_ptr<Timer> drawingTimer;

    const uint32_t MAX_FRAMES_IN_FLIGHT{ 2 }; // Explanation on this how we got this number: https://software.intel.com/content/www/us/en/develop/articles/practical-approach-to-vulkan-part-1.html
    size_t currentFrame{ 0 };
    uint32_t currentImageIndex;

    const std::vector<const char *> deviceExtensions {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Subroutines
    void cleanupSwapchain();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createSwapchainImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void createDepthResources();
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void loadModel();
    void copyBuffer(Buffer &srcBuffer, Buffer &dstBuffer, VkDeviceSize size);
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSemaphoreAndFencePools();
    void setupSynchronizationObjects();
    void setupTimer();
    void setupCamera();
    void updateUniformBuffer();
};

} // namespace vulkr