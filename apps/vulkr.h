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

#pragma once

 // Vulkan Common
#include "common/vulkan_common.h"

// Common files
#include "common/helpers.h"
#include "common/timer.h"
#include "common/debug_util.h"

// Core Files
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
#include "core/fence_pool.h"
#include "core/semaphore_pool.h"

// Platform files
#include "platform/application.h"
#include "platform/input_event.h"

// Rendering
#include "rendering/gltf_loader.h"
#include "rendering/obj_loader.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// STB
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// ImGui
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

// C++ Libraries
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <queue>
#include <mutex>
#include <condition_variable>

// Required for imgui integration, might be able to remove this if there's an alternate integration with volk
VkInstance g_instanceHandle;
PFN_vkVoidFunction loadFunction(const char *function_name, void *user_data) { return vkGetInstanceProcAddr(g_instanceHandle, function_name); }

namespace vulkr
{

constexpr uint32_t maxFramesInFlight{ 2u }; // Explanation on this how we got this number: https://software.intel.com/content/www/us/en/develop/articles/practical-approach-to-vulkan-part-1.html

/*
 * Command buffer 1: Used for setup before core rendering loop; we ensure all the setup is done before we use it again for the compute/offscreen pass
 * Command buffer 2: Used for the post processing
 * Command buffer 3: Used for image transfers
 * Command buffer 4: Used for geometry buffer pass 
 */
constexpr uint32_t commandBufferCountForFrame{ 4u };
constexpr uint32_t taaDepth{ 128u };
constexpr uint32_t maxObjInstanceCount{ 1000u };
constexpr uint32_t maxGltfInstanceCount{ 100u };
constexpr uint32_t maxLightCount{ 100u };

#ifdef VULKR_DEBUG
constexpr uint32_t particlesPerAttractor{ 64u };
#else
constexpr uint32_t particlesPerAttractor{ 128u };
#endif

constexpr std::array<glm::vec3, 6> attractors = {
	glm::vec3(5.0f, 0.0f, 0.0f),
	glm::vec3(-5.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, 0.0f, 5.0f),
	glm::vec3(0.0f, 0.0f, -5.0f),
	glm::vec3(0.0f, 4.0f, 0.0f),
	glm::vec3(0.0f, -8.0f, 0.0f),
};

const std::string defaultObjModelFilePath = "../../assets/obj_models/";
const std::string defaultGltfModelFilePath = "../../assets/gltf_models/";

// #define MULTI_THREAD // TODO: enabling multi-threaded loading tentatively works with rasterization but fails for the raytracing pipeline during the buildTlas second call; still a WIP
enum class RenderingTechnique
{
	FORWARD,
	RAY_TRACING,
	DEFERRED
} activeRenderingTechnique;

bool temporalAntiAliasingEnabled{ false }; // Flag to enable temporal anti-aliasing

/* Structs shared across the GPU and CPU */
struct CameraData
{
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct LightData
{
	glm::vec3 position{ 10.0f, 4.3f, 7.1f }; // Used for point lights
	float intensity{ 35.0f };
	glm::vec3 color{ 1.0f, 1.0f, 1.0f };
	int type{ 0 }; // 0: point, 1: directional (infinite)
	glm::vec2 rotation{ 75.0f, 40.0f }; // Used for directional lights; represents horizontal (azimuth) and vertical (elevation) rotation
	glm::vec2 blank{ 0.0, 0.0 }; // padding
	// TODO: to support area lights, look into vector irradiance (Real time rendering page 379) where you can integrate over the various light vectors that an area light emits and convert it into a directional light source  without introducing any errors
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

	bool operator == (const ObjInstance &other) const
	{
		return
			transform == other.transform &&
			transformIT == other.transformIT &&
			objIndex == other.objIndex &&
			textureOffset == other.textureOffset &&
			vertices == other.vertices &&
			indices == other.indices &&
			materials == other.materials &&
			materialIndices == other.materialIndices;
	}

	bool operator != (const ObjInstance &other) const
	{
		return !(*this == other);
	}
};

struct GltfInstance
{
	alignas(16) glm::mat4 transform;
	alignas(8) uint64_t modelIndex;
	alignas(8) uint64_t blank; // padding; not required??
};

struct Particle
{
	alignas(16) glm::vec4 position; // xyz = position, w = mass
	alignas(16) glm::vec4 velocity; // xyz = velocity, w = gradient texture position
};

struct alignas(16) GltfMaterial
{
	glm::vec4 baseColorFactor;
	glm::vec4 emissiveFactor;
	glm::vec4 diffuseFactor;
	glm::vec4 specularFactor;
	float workflow;
	int colorTextureSet;
	int PhysicalDescriptorTextureSet;
	int normalTextureSet;
	int occlusionTextureSet;
	int emissiveTextureSet;
	float metallicFactor;
	float roughnessFactor;
	float alphaMask;
	float alphaMaskCutoff;
	float emissiveStrength;
};

// Push constants; note that any modifications to push constants must be matched in the shaders and offsets must be set appropriately including when multiple push constants are defined for different stages (see layout(offset = 16))
// ensure push constants fall under the max size (128 bytes is the min size so we shouldn't expect more than this); be careful of implicit padding (eg. vec3 should always be followed by a 4 byte datatype if possible or else it might pad to 16 bytes)
struct RasterizationPushConstants
{
	int lightCount{ 0 };
} rasterizationPushConstants;

struct RaytracingPushConstants
{
	int lightCount{ 0 };
	int frameSinceViewChange{ -1 }; // TODO not used
} raytracingPushConstants;

struct TaaPushConstants
{
	glm::vec2 jitter{ glm::vec2(0.0f) };
	int frameSinceViewChange{ -1 };
	int blank{ 0 }; // alignment
} taaPushConstants;

struct PostProcessPushConstants
{
	glm::vec2 imageExtent;
} postProcessPushConstants;

struct ComputePushConstants
{
	int indexCount;
	float time;
} computePushConstants;

struct ComputeParticlesPushConstants
{
	int startingIndex{ 0 };
	int particleCount{ 0 };
	float deltaTime{ 0.0f };
	int blank{ 0 }; // alignment
} computeParticlesPushConstants;

struct GltfPushConstants
{
	glm::vec3 cameraPos;
	int materialIndex{ 0 };
	int lightCount{ 0 };
} gltfPushConstants;

struct MrtGeometryBufferPushConstants
{
	int materialIndex{ 0 };
} mrtGeometryBufferPushConstants;

struct DeferredShadingPushConstants
{
	glm::vec3 cameraPos;
	int lightCount{ 0 };
} deferredShadingPushConstants;

/* CPU only structs */

// Holds all of the necessary rendering data to display a .obj file model
struct ObjModelRenderingData
{
	std::string objFilePath;
	uint64_t txtOffset;
	uint32_t verticesCount;
	uint32_t indicesCount;
	std::unique_ptr<Buffer> vertexBuffer;
	std::unique_ptr<Buffer> indexBuffer;
	std::unique_ptr<Buffer> materialsBuffer;
	std::unique_ptr<Buffer> materialsIndexBuffer;
};

enum PBRWorkflows { PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1 };

// Holds all of the necessary rendering data to display a .gltf file model
struct GltfModelRenderingData
{
	std::string filePath;
	gltf::Model gltfModel;
	std::unique_ptr<Buffer> materialsBuffer;
};

struct PipelineData
{
	std::unique_ptr<Pipeline> pipeline;
	std::unique_ptr<PipelineState> pipelineState;
};

struct AccelerationStructure
{
	AccelerationStructure(Device &device) : device(device)
	{
	}

	~AccelerationStructure()
	{
		buffer.reset();
		vkDestroyAccelerationStructureKHR(device.getHandle(), accelerationStuctureKHR, nullptr);
	}

	Device &device;
	VkAccelerationStructureKHR accelerationStuctureKHR = VK_NULL_HANDLE;
	std::unique_ptr<Buffer> buffer;
};

// Inputs used to build the bottom-level acceleration structure.
struct BlasInput
{
	// Data used to build acceleration structure geometry
	std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
	std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildRangeInfo;
};

// Bottom-level acceleration structure, along with the information needed to re-build it.
struct BlasEntry
{
	BlasEntry(BlasInput input_) : input(input_)
	{
	}

	~BlasEntry()
	{
		accelerationStructure.reset();
	}

	// User provided input.
	BlasInput input;

	// VkAccelerationStructureKHR plus extra info needed for our memory allocator.
	std::unique_ptr<AccelerationStructure> accelerationStructure; // This struct will be initialized externally

	// Additional parameters for acceleration structure builds
	VkBuildAccelerationStructureFlagsKHR flags = 0;
};

struct Tlas
{
	std::unique_ptr<AccelerationStructure> accelerationStructure;
	VkBuildAccelerationStructureFlagsKHR flags = 0;
};

/* VulkrApp class */
class VulkrApp : public Application
{
public:
	VulkrApp(Platform &platform, std::string name);

	~VulkrApp();

	virtual void prepare() override;

	virtual void update() override;

	virtual void recreateSwapchain() override;

	virtual void handleInputEvents(const InputEvent &inputEvent) override;
private:
	/* IMPORTANT NOTICE: To enable/disable features, depending on whether it's core or packed into another feature extension struct, you might have to go into device.cpp to enabled them through another struct using the pNext chain */
	const std::vector<const char *> deviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME, // Required to have access to the swapchain and render images to the screen
#ifndef RENDERDOC_DEBUG
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // Required by VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, // To build acceleration structures
		VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, // Provides access to vkCmdTraceRaysKHR
#endif
		VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME, // Provides access to vkResetQueryPool
		VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, // Required to use debugPrintfEXT in shaders
		VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
		VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME
	};

	std::unique_ptr<Instance> m_instance{ nullptr };
	VkSurfaceKHR m_surface{ VK_NULL_HANDLE };
	std::unique_ptr<Device> device{ nullptr };
	Queue *m_graphicsQueue{ VK_NULL_HANDLE };
	Queue *m_computeQueue{ VK_NULL_HANDLE };
	Queue *m_presentQueue{ VK_NULL_HANDLE };
	Queue *m_transferQueue{ VK_NULL_HANDLE }; // TODO: currently unused in code
	uint32_t m_workGroupSize;
	uint32_t m_shadedMemorySize;

	std::unique_ptr<Swapchain> swapchain{ nullptr };

	struct RenderPassData
	{
		std::unique_ptr<RenderPass> renderPass{ nullptr };
		std::vector<Subpass> subpasses;
		std::vector<VkAttachmentReference2> inputAttachments;
		std::vector<VkAttachmentReference2> colorAttachments;
		std::vector<VkAttachmentReference2> resolveAttachments;
		std::vector<VkAttachmentReference2> depthStencilAttachments;
		std::vector<uint32_t> preserveAttachments;
	};

	// Forward rendering main render pass
	RenderPassData mainRenderPass;
	// Deferred rendering render passes
	RenderPassData mrtGeometryBufferRenderPass;
	RenderPassData deferredShadingRenderPass;
	// Post processing render pass
	RenderPassData postRenderPass;

	std::unique_ptr<DescriptorSetLayout> globalDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> objectDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> textureDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> postProcessingDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> taaDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> particleComputeDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> gltfMaterialSamplersDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> gltfNodeDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> gltfMaterialDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> geometryBufferDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorPool> descriptorPool;
	std::unique_ptr<DescriptorPool> imguiPool;

	std::unique_ptr<ImageView> depthImageView{ nullptr };
	std::unique_ptr<Sampler> textureSampler{ nullptr };

	std::unique_ptr<SemaphorePool> semaphorePool;
	std::unique_ptr<FencePool> fencePool;
	std::vector<VkFence> imagesInFlight;

	std::unique_ptr<CameraController> cameraController;
	std::unique_ptr<Timer> drawingTimer;

	std::array<glm::vec2, taaDepth> haltonSequence;

	struct FrameData
	{
		std::array<std::unique_ptr<Framebuffer>, maxFramesInFlight> offscreenFramebuffers;
		std::array<std::unique_ptr<Framebuffer>, maxFramesInFlight> postProcessFramebuffers;
		std::array<std::unique_ptr<Framebuffer>, maxFramesInFlight> geometryBufferFramebuffers;
		std::array<std::unique_ptr<Framebuffer>, maxFramesInFlight> deferredShadingFramebuffers;

		std::array<VkSemaphore, maxFramesInFlight> imageAvailableSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> offscreenRenderingFinishedSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> postProcessRenderingFinishedSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> outputImageCopyFinishedSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> computeParticlesFinishedSemaphores;
		std::array<VkFence, maxFramesInFlight> inFlightFences;

		std::array<std::unique_ptr<CommandPool>, maxFramesInFlight> commandPools;
		std::array<std::array<std::shared_ptr<CommandBuffer>, commandBufferCountForFrame>, maxFramesInFlight> commandBuffers;
	} frameData;

	// Clear values
	std::vector<VkClearValue> offscreenFramebufferClearValues;
	std::vector<VkClearValue> postProcessFramebufferClearValues;
	std::vector<VkClearValue> geometryBufferFramebufferClearValues;
	std::vector<VkClearValue> deferredShadingFramebufferClearValues;

	// Buffers
	std::unique_ptr<Buffer> lightBuffer;
	std::unique_ptr<Buffer> cameraBuffer;
	std::unique_ptr<Buffer> previousFrameCameraBuffer;
	std::unique_ptr<Buffer> gltfInstanceBuffer;
	std::unique_ptr<Buffer> objInstanceBuffer;
	std::unique_ptr<Buffer> previousFrameObjInstanceBuffer;
	std::unique_ptr<Buffer> particleBuffer;
	std::unique_ptr<Buffer> gltfMaterialsBuffer;

	// TAA related textures
	std::unique_ptr<ImageView> outputImageView;
	std::unique_ptr<ImageView> copyOutputImageView;
	std::unique_ptr<ImageView> historyImageView;
	std::unique_ptr<ImageView> velocityImageView;

	// Deferred shading geometry buffers
	std::unique_ptr<ImageView> positionImageView;
	std::unique_ptr<ImageView> normalImageView;
	std::unique_ptr<ImageView> uv0ImageView;
	std::unique_ptr<ImageView> uv1ImageView;
	std::unique_ptr<ImageView> color0ImageView;
	std::unique_ptr<ImageView> materialIndexImageView;

	// Descriptor sets
	std::unique_ptr<DescriptorSet> globalDescriptorSet;
	std::unique_ptr<DescriptorSet> objectDescriptorSet;
	std::unique_ptr<DescriptorSet> postProcessingDescriptorSet;
	std::unique_ptr<DescriptorSet> taaDescriptorSet;
	std::unique_ptr<DescriptorSet> particleComputeDescriptorSet;
	std::unique_ptr<DescriptorSet> textureDescriptorSet; // This is currently only read by shaders
	std::unique_ptr<DescriptorSet> raytracingDescriptorSet;
	std::unique_ptr<DescriptorSet> gltfMaterialDescriptorSet;
	std::unique_ptr<DescriptorSet> geometryBufferDescriptorSet; // Although this says geometry buffers, they're actually filled with ImageView rather than Buffer

	size_t currentFrame{ 0 };

	// TODO: Currently unused; used when MULTI_THREAD is defined
	std::mutex bufferMutex;
	std::mutex commandPoolMutex;
	std::condition_variable commandPoolCv;
	std::vector<std::unique_ptr<CommandPool>> initCommandPools;
	std::queue<uint8_t> initCommandPoolIds;

	struct Pipelines
	{
		PipelineData pbr;
		PipelineData offscreen;
		PipelineData postProcess;
		PipelineData computeModelAnimation;
		PipelineData computeParticleCalculate;
		PipelineData computeParticleIntegrate;
		PipelineData rayTracing;
		PipelineData mrtGeometryBuffer;
		PipelineData deferredShading;
	} pipelines;

	std::vector<GltfModelRenderingData> gltfModelRenderingDataList;
	std::vector<ObjModelRenderingData> objModelRenderingDataList;
	std::vector<std::unique_ptr<ImageView>> textureImageViews;
	std::vector<ObjInstance> objInstances;
	std::vector<GltfInstance> gltfInstances;
	std::vector<LightData> sceneLights;

	uint32_t animationIndex = 0;
	float animationTimer = 0.0f;

	bool haveLightsUpdated{ false };

	VkDeviceSize particleBufferSize{ 0 };
	std::vector<Particle> allParticleData;

	// Subroutines
	void drawImGuiInterface();
	void animateInstances();
	void animateWithCompute();
	void computeParticles();
	void renderNode(PipelineData *pipelineData, vulkr::gltf::Node *node, uint32_t instanceIndex);
	void dataUpdatePerFrame();
	void rasterizeObj();
	void rasterizeGltf();
	void initiateDeferredRenderingPass();
	void postProcess();
	void cleanupSwapchain();
	void createInstance();
	void createSurface();
	void createDevice();
	void createSwapchain();
	void createImageResourcesForFrames();
	void createMainRenderPass();
	void createPostRenderPass();
	void createMrtGeometryBufferRenderPass();
	void createDeferredShadingRenderPass();
	void createDescriptorSetLayouts();
	void createMainRasterizationPipeline();
	void createPbrRasterizationPipeline();
	void createMrtGeometryBufferPipeline();
	void createDeferredShadingPipeline();
	void createPostProcessingPipeline();
	void createModelAnimationComputePipeline();
	void createParticleCalculateComputePipeline();
	void createParticleIntegrateComputePipeline();
	void createFramebuffers();
	void createCommandPools();
	void createCommandBuffers();
	void copyBufferToImage(const Buffer &srcBuffer, const Image &dstImage, uint32_t width, uint32_t height);
	void createDepthResources();
	std::unique_ptr<Image> createTextureImage(const std::string &filename); // Reads a texture file and populate the texture image with the contents
	void createTextureSampler();
	void loadTextureImages(const std::vector<std::string> &textureFiles);
	void copyBufferToBuffer(const Buffer &srcBuffer, const Buffer &dstBuffer, VkDeviceSize size);
	void createVertexBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader);
	void createIndexBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader);
	void createMaterialBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader);
	void createMaterialBuffer(GltfModelRenderingData &gltfModelRenderingData);
	void createMaterialIndicesBuffer(ObjModelRenderingData &objModelRenderingData, const ObjLoader &objLoader);
	void createUniformBuffers();
	void createSSBOs();
	void prepareParticleData();
	void createDescriptorPool();
	void createDescriptorSets();
	void setupNodeDescriptorSet(vulkr::gltf::Node *node);
	void createSceneLights();
	void loadGltfModel(const std::string &gltfFilePath);
	void loadObjModel(const std::string &objFilePath);
	void createObjInstance(const std::string &objFilePath, glm::mat4 transform);
	void createGltfInstance(const std::string &gltfFilePath, glm::mat4 transform);
	void loadModels();
	uint64_t getObjModelIndex(const std::string &name);
	uint64_t getGltfModelIndex(const std::string &name);
	void createSceneInstances();
	void createSemaphoreAndFencePools();
	void setupSynchronizationObjects();
	void setupTimer();
	void initializeHaltonSequenceArray();
	void setupCamera();
	void initializeImGui();
	void resetFrameSinceViewChange();
	void initializeBufferData();

	// TODO: These methods to allow command pool access for multiple threads are not in use at the moment and will need to be used when expanding multi threading capabilities
	uint8_t getInitCommandPoolId();
	void returnInitCommandPool(uint8_t commandPoolId);

	// Raytracing member variables
	std::vector<BlasEntry> m_blas; // Vector containing all the BLASes built in buildBlas (and referenced by the TLAS)
	std::unique_ptr<Tlas> m_tlas; // Top-level acceleration structure
	std::vector<VkAccelerationStructureInstanceKHR> m_accelerationStructureInstances;
	std::unique_ptr<Buffer> m_instBuffer; // Instance buffer containing the matrices and BLAS ids

	std::unique_ptr<DescriptorPool> m_rtDescPool;
	std::unique_ptr<DescriptorSetLayout> m_rtDescSetLayout;

	VkBuildAccelerationStructureFlagsKHR m_buildAccelerationStructureFlags = 0;
	const VkBufferUsageFlags m_rayTracingBufferUsageFlags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	std::unique_ptr<Buffer> m_rtSBTBuffer; // The raytracing shader binding table

	// Raytracing helpers
	VkDeviceAddress getBlasDeviceAddress(uint64_t blasId);
	BlasInput objectToVkGeometryKHR(size_t objModelIndex);
	std::unique_ptr<AccelerationStructure> createAccelerationStructure(VkAccelerationStructureCreateInfoKHR &accelerationStructureInfo);

	// Raytracing core subroutines
	void buildBlas();
	void buildTlas(bool update);

	void createRaytracingDescriptorPool();
	void createRaytracingDescriptorLayout();
	void createRaytracingDescriptorSets();

	void updateRaytracingDescriptorSet();
	void createRaytracingPipeline();
	void createRaytracingShaderBindingTable(); // https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways is a great resource on how the SBT works and how we should be organizing our shaders into primary and occlusion hit groups

	void raytrace();
}; // class VulkrApp

} // namespace vulkr