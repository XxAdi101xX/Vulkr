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
#include "core/queue.h"
#include "core/command_pool.h"
#include "core/command_buffer.h"
#include "core/buffer.h"
#include "core/image.h"
#include "core/sampler.h"
#include "core/semaphore_pool.h"
#include "core/fence_pool.h"

// Platform files
#include "platform/application.h"
#include "platform/input_event.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// C++ Libraries
#include <iostream>
#include <algorithm>

namespace vulkr
{
constexpr uint32_t maxFramesInFlight{ 2u }; // Explanation on this how we got this number: https://software.intel.com/content/www/us/en/develop/articles/practical-approach-to-vulkan-part-1.html
constexpr uint32_t commandBufferCountForFrame{ 3u };

/* Structs shared across the GPU and CPU */
// Push constants; note that any modifications to push constants must be matched in the shaders and offsets must be set appropriately including when multiple push constants are defined for different stages (see layout(offset = 16))
// ensure push constants fall under the max size (128 bytes is the min size so we shouldn't expect more than this); be careful of implicit padding (eg. vec3 should always be followed by a 4 byte datatype if possible or else it might pad to 16 bytes)
struct FluidSimulationPushConstant
{
	glm::vec2 gridSize{ 1280.0f, 720.0f };
	float gridScale{ 1.0f };
	float timestep{ 1.0f };
	glm::vec3 splatForce{ glm::vec3(0.0f) };
	float splatRadius{ 0.60f };
	glm::vec2 splatPosition{ glm::vec2(0.0f) };
	float dissipation{ 0.97f };
	int blank{ 0 }; // padding
} fluidSimulationPushConstant;

struct PipelineData
{
	std::unique_ptr<Pipeline> pipeline;
	std::unique_ptr<PipelineState> pipelineState;
};


/* FluidSimulation class */
class FluidSimulation : public Application
{
public:
	FluidSimulation(Platform &platform, std::string name);

	~FluidSimulation();

	virtual void prepare() override;

	virtual void update() override;

	virtual void recreateSwapchain() override;

	virtual void handleInputEvents(const InputEvent &inputEvent) override;
private:
	/* IMPORTANT NOTICE: To enable/disable features, depending on whether it's core or packed into another feature extension struct, you might have to go into device.cpp to enabled them through another struct using the pNext chain */
	const std::vector<const char *> deviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME, // Required to have access to the swapchain and render images to the screen
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

	std::unique_ptr<DescriptorSetLayout> fluidSimulationInputDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorSetLayout> fluidSimulationOutputDescriptorSetLayout{ nullptr };
	std::unique_ptr<DescriptorPool> descriptorPool;

	std::unique_ptr<Sampler> textureSampler{ nullptr };

	std::unique_ptr<SemaphorePool> semaphorePool;
	std::unique_ptr<FencePool> fencePool;
	std::vector<VkFence> imagesInFlight;

	std::unique_ptr<CameraController> cameraController;

	struct FrameData
	{
		std::array<VkSemaphore, maxFramesInFlight> imageAvailableSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> offscreenRenderingFinishedSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> postProcessRenderingFinishedSemaphores;
		std::array<VkSemaphore, maxFramesInFlight> outputImageCopyFinishedSemaphores;
		std::array<VkFence, maxFramesInFlight> inFlightFences;

		std::array<std::unique_ptr<CommandPool>, maxFramesInFlight> commandPools;
		std::array<std::array<std::shared_ptr<CommandBuffer>, commandBufferCountForFrame>, maxFramesInFlight> commandBuffers;
	} frameData;

	// Fluid velocity textures
	std::unique_ptr<ImageView> fluidVelocityInputImageView;
	std::unique_ptr<ImageView> fluidVelocityDivergenceInputImageView;
	std::unique_ptr<ImageView> fluidPressureInputImageView;
	std::unique_ptr<ImageView> fluidDensityInputImageView;
	std::unique_ptr<ImageView> fluidSimulationOutputImageView; // Generic backbuffer for all of the fluid simulation stages

	// Descriptor sets
	std::unique_ptr<DescriptorSet> fluidSimulationInputDescriptorSet;
	std::unique_ptr<DescriptorSet> fluidSimulationOutputDescriptorSet;

	size_t currentFrame{ 0 };

	struct Pipelines
	{
		PipelineData computeVelocityAdvection;
		PipelineData computeDensityAdvection;
		PipelineData computeVelocityGaussianSplat;
		PipelineData computeDensityGaussianSplat;
		PipelineData computeFluidVelocityDivergence;
		PipelineData computeJacobi;
		PipelineData computeGradientSubtraction;
	} pipelines;

	// Used to draw fluids onto screen
	MouseInput activeMouseInput{ MouseInput::None };
	glm::vec2 lastMousePosition{ glm::vec2(0.0f) };

	// Subroutines
	void computeFluidSimulation();
	void copyFluidOutputTextureToInputTexture(const Image *imageToCopyTo);
	void cleanupSwapchain();
	void createInstance();
	void createSurface();
	void createDevice();
	void createSwapchain();
	void createDescriptorSetLayouts();
	void createVelocityAdvectionComputePipeline();
	void createDensityAdvectionComputePipeline();
	void createVelocityGaussianSplatComputePipeline();
	void createDensityGaussianSplatComputePipeline();
	void createFluidVelocityDivergenceComputePipeline();
	void createJacobiComputePipeline();
	void createGradientSubtractionComputePipeline();
	void createCommandPools();
	void createCommandBuffers();
	std::unique_ptr<Image> createTextureImageWithInitialValue(uint32_t texWidth, uint32_t texHeight, VkImageUsageFlags imageUsageFlags); // Create an empty texture image
	void copyBufferToImage(const Buffer &srcBuffer, const Image &dstImage, uint32_t width, uint32_t height);
	void copyBufferToBuffer(const Buffer &srcBuffer, const Buffer &dstBuffer, VkDeviceSize size);
	void initializeFluidSimulationResources();
	void createDescriptorPool();
	void createDescriptorSets();
	void createTextureSampler();
	void createSemaphoreAndFencePools();
	void setupSynchronizationObjects();
	void setupCamera();
}; // class FluidSimulation

} // namespace vulkr