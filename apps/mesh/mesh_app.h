/* Copyright (c) 2026 Adithya Venkatarao
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

// STL Libraries
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// Platform files
#include "platform/application.h"

#include "gltf_mesh.h"

namespace vulkr
{

/* Forward declaration */
class Buffer;
class CameraController;
class CommandBuffer;
class CommandPool;
class DescriptorPool;
class DescriptorSet;
class DescriptorSetLayout;
class Device;
class FencePool;
class ImageView;
class Instance;
class Queue;
class Sampler;
class SemaphorePool;
class SlangShaderCompiler;
class Swapchain;

/* Mesh shader application: dynamic rendering, synchronization2, buffer device address,
   descriptor indexing and mesh shaders. The overrides of prepare(), update() and finish()
   hold the rendering setup and are implemented in mesh_app.cpp */
class MeshApp : public Application
{
public:
	MeshApp(Platform &platform, std::string name);

	~MeshApp();

	virtual void prepare() override;

	virtual void update() override;

	virtual void handleInputEvents(const InputEvent &inputEvent) override;

	virtual void handleFocusChange(bool isFocused) override;

	virtual void recreateSwapchain() override;
private:
	/* Number of frames that can be recorded and submitted while an earlier frame is still in flight */
	static constexpr uint32_t maxFramesInFlight{ 2u };

	/* Per frame recording and synchronisation objects; the acquire semaphore belongs to a frame, while the render
	   finished semaphores belong to a swapchain image so that they are never signalled while still being presented */
	struct FrameData
	{
		std::unique_ptr<CommandPool> commandPool;
		std::shared_ptr<CommandBuffer> commandBuffer;
		VkFence inFlightFence{ VK_NULL_HANDLE };
		VkSemaphore imageAvailableSemaphore{ VK_NULL_HANDLE };
	};

	/* IMPORTANT NOTICE: Device features are enabled through the pNext chain in src/core/device.cpp; the mesh
	   shader features are only chained in when VK_EXT_mesh_shader is requested through this list */
	const std::vector<const char *> deviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
	};

	void createSwapchainImageViews();

	void createFrameData();

	void createModels();

	void createTextures();

	void createDepthResources();

	void createPipeline();

	std::unique_ptr<Buffer> createBuffer(const void *data, VkDeviceSize size, VkBufferUsageFlags usage);

	std::unique_ptr<ImageView> createTexture(CommandBuffer &commandBuffer, const GltfTextureData &texture,
		std::vector<std::unique_ptr<Buffer>> &stagingBuffers);

	void submitAndWait(CommandBuffer &commandBuffer);

	void recordCommandBuffer();

	void cleanup();

	std::unique_ptr<Instance> m_instance{ nullptr };
	VkSurfaceKHR m_surface{ VK_NULL_HANDLE };
	std::unique_ptr<Device> m_device{ nullptr };
	Queue *m_generalQueue{ nullptr };
	std::unique_ptr<Swapchain> m_swapchain{ nullptr };

	/* The Swapchain owns the swapchain images, the app owns their views and the per image present semaphores */
	std::vector<VkImageView> m_swapchainImageViews;
	std::vector<VkSemaphore> m_renderFinishedSemaphores;

	/* One loaded model: the meshlet buffers live in device memory and the shader reaches them by address, so a model is
	   four addresses plus the number of workgroups that render it */
	struct MeshModel
	{
		std::string name;
		glm::vec3 translation{ 0.0f };
		std::unique_ptr<Buffer> vertexBuffer{ nullptr };
		std::unique_ptr<Buffer> meshletBuffer{ nullptr };
		std::unique_ptr<Buffer> meshletVertexBuffer{ nullptr };
		std::unique_ptr<Buffer> meshletTriangleBuffer{ nullptr };
		VkDeviceAddress vertexAddress{ 0u };
		VkDeviceAddress meshletAddress{ 0u };
		VkDeviceAddress meshletVertexAddress{ 0u };
		VkDeviceAddress meshletTriangleAddress{ 0u };
		uint32_t meshletCount{ 0u };
	};

	/* The scene: as many models as the asset list provides, each rendered with one mesh task dispatch */
	std::vector<MeshModel> m_models;

	/* Bindless texture array: one descriptor per texture of every model, indexed by whatever slot the vertices carry */
	std::vector<std::unique_ptr<ImageView>> m_textureViews;
	std::unique_ptr<Sampler> m_textureSampler{ nullptr };
	std::unique_ptr<DescriptorSetLayout> m_textureSetLayout{ nullptr };
	std::unique_ptr<DescriptorPool> m_texturePool{ nullptr };
	std::unique_ptr<DescriptorSet> m_textureSet{ nullptr };

	/* Decoded images waiting to be uploaded; slot 0 is a white texel for vertices whose material has no texture */
	std::vector<GltfTextureData> m_textureData;

	/* The scene is rendered with a depth buffer for the triangle order to be correct */
	std::unique_ptr<ImageView> m_depthImageView{ nullptr };

	/* Everything the mesh shader needs, matrices first and then the device address of each meshlet buffer in the
	   order the shader declares its pointers */
	struct PushConstants
	{
		glm::mat4 viewProjection;
		glm::mat4 model;
		VkDeviceAddress vertices;
		VkDeviceAddress meshlets;
		VkDeviceAddress meshletVertices;
		VkDeviceAddress meshletTriangles;
	};

	/* The shader declares the same block with std430 offsets, so the layout is pinned here */
	static_assert(sizeof(PushConstants) == sizeof(glm::mat4) * 2 + sizeof(VkDeviceAddress) * 4);
	static_assert(offsetof(PushConstants, vertices) == sizeof(glm::mat4) * 2);

	/* Same orbit/pan/zoom controls as the main app, plus smooth delta-time keyboard movement */
	std::unique_ptr<CameraController> m_cameraController{ nullptr };
	glm::mat4 m_viewProjection{ 1.0f };
	float m_rotation{ 0.0f };

	/* Runtime compiled shaders and the mesh shader pipeline that uses them */
	std::unique_ptr<SlangShaderCompiler> m_slangCompiler{ nullptr };
	VkPipelineLayout m_pipelineLayout{ VK_NULL_HANDLE };
	VkPipeline m_pipeline{ VK_NULL_HANDLE };

	std::unique_ptr<FencePool> m_fencePool{ nullptr };
	std::unique_ptr<SemaphorePool> m_semaphorePool{ nullptr };
	std::array<FrameData, maxFramesInFlight> m_frames{};

	uint32_t m_currentFrame{ 0u };
	uint32_t m_swapchainImageIndex{ 0u };
};

} // namespace vulkr
