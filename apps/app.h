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
#include "common/obj_loader.h"
#include "common/debug_util.h"

#include "platform/application.h"
#include "platform/input_event.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // don't use the OpenGL default depth range of -1.0 to 1.0 and use 0.0 to 1.0
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

// Required for imgui integration, might be able to remove this if there's an alternate integration with volk
VkInstance g_instance;
PFN_vkVoidFunction loadFunction(const char *function_name, void *user_data) { return vkGetInstanceProcAddr(g_instance, function_name); }

namespace vulkr
{

constexpr uint32_t maxFramesInFlight{ 2 }; // Explanation on this how we got this number: https://software.intel.com/content/www/us/en/develop/articles/practical-approach-to-vulkan-part-1.html
constexpr uint32_t MAX_OBJECT_COUNT{ 10000 };
bool raytracingEnabled{ true }; // Flag to enable ray tracing vs rasterization

/* Structs shared on both the GPU and CPU */
struct CameraData
{
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct ObjInstance
{
    alignas(16) glm::mat4 transform;
    alignas(16) glm::mat4 transformIT;
    alignas(8) uint64_t objIndex;
    alignas(8) uint64_t textureOffset;
    alignas(8) VkDeviceAddress vertices;
    alignas(8) VkDeviceAddress indices;
    alignas(8) VkDeviceAddress materials;
    alignas(8) VkDeviceAddress materialIndices;
};

/* CPU only structs */
struct ObjModel
{
    uint32_t verticesCount;
    uint32_t indicesCount;
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> materialsBuffer;
    std::unique_ptr<Buffer> materialsIndexBuffer;
};

struct PipelineData
{
    std::shared_ptr<GraphicsPipeline> pipeline;
    std::shared_ptr<PipelineState> pipelineState;
};

struct Texture
{
    std::unique_ptr<Image> image;
    std::unique_ptr<ImageView> imageview;
};

struct RenderObject
{
    std::shared_ptr<ObjModel> objModel;

    std::shared_ptr<PipelineData> pipelineData;
};

// Inputs used to build Bottom-level acceleration structure.
// You manage the lifetime of the buffer(s) referenced by the
// VkAccelerationStructureGeometryKHRs within. In particular, you must
// make sure they are still valid and not being modified when the BLAS
// is built or updated.
struct BlasInput
{
    // Data used to build acceleration structure geometry
    std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildRangeInfo;
};

struct AccelKHR
{
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    std::unique_ptr<Buffer> buffer;

    Device &device;
    AccelKHR(Device &device) : device(device)
    {}
    ~AccelKHR()
    {
        buffer.reset();
        vkDestroyAccelerationStructureKHR(device.getHandle(), accel, nullptr);
    }
};

// Bottom-level acceleration structure, along with the information needed to re-build it.
struct BlasEntry
{
    // User-provided input.
    BlasInput input;

    // VkAccelerationStructureKHR plus extra info needed for our memory allocator.
    std::unique_ptr<AccelKHR> as; // This struct should be initialized outside

    // Additional parameters for acceleration structure builds
    VkBuildAccelerationStructureFlagsKHR flags = 0;

    BlasEntry() = default;
    BlasEntry(BlasInput input_) : input(std::move(input_))
    {}
    ~BlasEntry()
    {
        as.reset();
    }
};

// This is an instance of a BLAS
struct BlasInstance
{
    uint32_t                   blasId{ 0 };            // Index of the BLAS in m_blas
    uint32_t                   instanceCustomId{ 0 };  // Instance Index (gl_InstanceCustomIndexEXT)
    uint32_t                   hitGroupId{ 0 };        // Hit group index in the SBT
    uint32_t                   mask{ 0xFF };           // Visibility mask, will be AND-ed with ray mask
    VkGeometryInstanceFlagsKHR flags{ VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR }; // Geometry instance flags
    glm::mat4                  transform{ glm::mat4(1) };  // Identity
};

struct Tlas
{
    std::unique_ptr<AccelKHR> as;
    VkBuildAccelerationStructureFlagsKHR flags = 0;
};

/* MainApp class */
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
    /* 
    IMPORTANT NOTICE: To enable/disable features, it is not adequate to add the extension name to the device extensions array below. You must also go into
    instances.cpp to manually enable these features through VkPhysicalDeviceFeatures2 pNext chain
    */
    const std::vector<const char *> deviceExtensions {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, // Required to have access to the swapchain and render images to the screen
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // Required by VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, // To build acceleration structures
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, // Provides access to vkCmdTraceRaysKHR
        VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME, // Provides access to vkResetQueryPool
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, // Required to use debugPrintfEXT in shaders
    };

    std::unique_ptr<Instance> instance{ nullptr };
    VkSurfaceKHR surface{ VK_NULL_HANDLE };
    std::unique_ptr<Device> device{ nullptr };
    VkQueue graphicsQueue{ VK_NULL_HANDLE };
    VkQueue presentQueue{ VK_NULL_HANDLE };

    std::unique_ptr<Swapchain> swapchain{ nullptr };

    struct RenderPassData
    {
        std::unique_ptr<RenderPass> renderPass{ nullptr };
        std::vector<Subpass> subpasses;
        std::vector<VkAttachmentReference> inputAttachments;
        std::vector<VkAttachmentReference> colorAttachments;
        std::vector<VkAttachmentReference> resolveAttachments;
        std::vector<VkAttachmentReference> depthStencilAttachments;
        std::vector<uint32_t> preserveAttachments;
    };

    RenderPassData mainRenderPass;
    RenderPassData postRenderPass;

    std::unique_ptr<DescriptorSetLayout> globalDescriptorSetLayout{ nullptr };
    std::unique_ptr<DescriptorSetLayout> objectDescriptorSetLayout{ nullptr };
    std::unique_ptr<DescriptorSetLayout> textureDescriptorSetLayout{ nullptr };
    std::unique_ptr<DescriptorPool> descriptorPool;
    std::unique_ptr<DescriptorPool> imguiPool;

    std::unique_ptr<Image> depthImage{ nullptr };
    std::unique_ptr<ImageView> depthImageView{ nullptr };
    std::unique_ptr<Sampler> textureSampler{ nullptr };

    std::unique_ptr<SemaphorePool> semaphorePool;
    std::unique_ptr<FencePool> fencePool;
    std::vector<VkFence> imagesInFlight;

    std::unique_ptr<CameraController> cameraController;
    std::unique_ptr<Timer> drawingTimer;

    struct FrameData
    {
        std::array<std::unique_ptr<Image>, maxFramesInFlight> outputImages;
        std::array<std::unique_ptr<ImageView>, maxFramesInFlight> outputImageViews;
        std::array<std::unique_ptr<Framebuffer>, maxFramesInFlight> outputImageFramebuffers;

        std::array<VkSemaphore, maxFramesInFlight> imageAvailableSemaphores;
        std::array<VkSemaphore, maxFramesInFlight> renderingFinishedSemaphores;
        std::array<VkFence, maxFramesInFlight> inFlightFences;

        std::array<std::unique_ptr<CommandPool>, maxFramesInFlight> commandPools;
        std::array<std::shared_ptr<CommandBuffer>, maxFramesInFlight> commandBuffers;

        std::array<std::unique_ptr<DescriptorSet>, maxFramesInFlight> globalDescriptorSets;
        std::array<std::unique_ptr<DescriptorSet>, maxFramesInFlight> objectDescriptorSets;
        std::array<std::unique_ptr<DescriptorSet>, maxFramesInFlight> rtDescriptorSets;
        std::array<std::unique_ptr<Buffer>, maxFramesInFlight> globalBuffers;
        std::array<std::unique_ptr<Buffer>, maxFramesInFlight> objectBuffers;
    } frameData;

    // Since the texture will be readonly, we don't require a descriptor set per frame
    std::unique_ptr<DescriptorSet> textureDescriptorSet;
    size_t currentFrame{ 0 };

    std::vector<RenderObject> renderables;
    std::unordered_map<std::string, std::shared_ptr<PipelineData>> pipelineDataMap;
    std::unordered_map<std::string, std::shared_ptr<ObjModel>> objModels;
    std::vector<Texture> textures;
    std::vector<ObjInstance> objInstances;

    // Subroutines
    void drawImGuiInterface();
    void updateBuffersPerFrame();
    void rasterize();
    void cleanupSwapchain();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createOutputImageAndImageView();
    void createMainRenderPass();
    void createPostRenderPass();
    void createDescriptorSetLayouts();
    std::shared_ptr<PipelineData> createPipelineData(std::shared_ptr<GraphicsPipeline> pipeline, std::shared_ptr<PipelineState> pipelineState, const std::string &name);
    void createGraphicsPipelines();
    void createFramebuffers();
    void createCommandPools();
    void createCommandBuffers();
    void copyBufferToImage(const Buffer &srcBuffer, const Image &dstImage, uint32_t width, uint32_t height);
    void createDepthResources();
    std::unique_ptr<Image> createTextureImage(const std::string &filename);
    std::unique_ptr<ImageView> createTextureImageView(const Image &image);
    void createTextureSampler();
    void loadTextureImages(const std::vector<std::string> &textureFiles);
    void copyBufferToBuffer(const Buffer &srcBuffer, const Buffer &dstBuffer, VkDeviceSize size);
    void createVertexBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader);
    void createIndexBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader);
    void createMaterialBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader);
    void createMaterialIndicesBuffer(std::shared_ptr<ObjModel> objModel, const ObjLoader &objLoader);
    void createUniformBuffers();
    void createSSBOs();
    void createDescriptorPool();
    void createDescriptorSets();
    void loadModel(const std::string &objFileName, glm::mat4 transform);
    void loadModels();
    void createScene();
    void createSemaphoreAndFencePools();
    void setupSynchronizationObjects();
    void setupTimer();
    void setupCamera();
    void initializeImGui();

    std::shared_ptr<PipelineData> getPipelineData(const std::string &name);
    std::shared_ptr<ObjModel> getObjModel(const std::string &name);

    // Raytracing TODO: cleanup this section
    BlasInput objectToVkGeometryKHR(size_t renderableIndex);
    void createBottomLevelAS();
    std::unique_ptr<AccelKHR> createAcceleration(VkAccelerationStructureCreateInfoKHR &accel);
    void buildBlas(const std::vector<BlasInput> &input, VkBuildAccelerationStructureFlagsKHR flags);
    // Vector containing all the BLASes built in buildBlas (and referenced by the TLAS)
    std::vector<BlasEntry> m_blas;

    void createTopLevelAS();
    void buildTlas(
        const std::vector<BlasInstance> &instances,
        VkBuildAccelerationStructureFlagsKHR flags,
        bool update = false
    );
    // Top-level acceleration structure
    std::unique_ptr<Tlas> m_tlas;
    VkAccelerationStructureInstanceKHR instanceToVkGeometryInstanceKHR(const BlasInstance &instance);
    // Instance buffer containing the matrices and BLAS ids
    std::unique_ptr<Buffer> m_instBuffer;

    void createRtDescriptorPool();
    void createRtDescriptorLayout();
    void createRtDescriptorSets();

    std::unique_ptr<DescriptorPool> m_rtDescPool;
    std::unique_ptr<DescriptorSetLayout> m_rtDescSetLayout;

    VkBufferUsageFlags flag = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; // TODO do we need this
    VkBufferUsageFlags rayTracingFlags = // used also for building acceleration structures 
        flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    void updateRtDescriptorSet();

    struct RtPushConstant
    {
        glm::vec4 clearColor{ 1.0f, 1.0f, 1.0f, 1.0f };
        glm::vec3 lightPosition{ 10.0f, 13.0f, 4.5f };
        float lightIntensity{ 100.0f };
        int lightType{ 0 }; // 0: point, 1: infinite
    } m_rtPushConstants;
    void                                              createRtPipeline();

    std::vector<ShaderModule> raytracingShaderModules; // TODO: cleanup
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
    VkPipelineLayout                                  m_rtPipelineLayout;
    VkPipeline                                        m_rtPipeline;

    // https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways is a great resource on how the SBT works and how we should be organizing our
    // shaders into primary and occlusion hit groups
    void           createRtShaderBindingTable();
    std::unique_ptr<Buffer> m_rtSBTBuffer;

    void raytrace(const uint32_t &swapchainImageIndex);
};

} // namespace vulkr