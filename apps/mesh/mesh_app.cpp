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

#include "mesh_app.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "common/logger.h"
#include "common/helpers.h"
#include "platform/platform.h"

#include "rendering/camera_controller.h"
#include "rendering/slang_shader_compiler.h"

#include "core/buffer.h"
#include "core/command_buffer.h"
#include "core/descriptor_pool.h"
#include "core/descriptor_set.h"
#include "core/descriptor_set_layout.h"
#include "core/command_pool.h"
#include "core/device.h"
#include "core/fence_pool.h"
#include "core/image.h"
#include "core/image_view.h"
#include "core/instance.h"
#include "core/queue.h"
#include "core/sampler.h"
#include "core/semaphore_pool.h"
#include "core/swapchain.h"

#include "gltf_mesh.h"
#include "meshlet_builder.h"

namespace vulkr
{

// The models the app renders; the main vulkr app draws these too. Every image they declare becomes one slot of the
// bindless texture array, and each model gets its own set of meshlet buffers
static constexpr const char *kModelPaths[]{
	"../../assets/gltf_models/SciFiHelmet/SciFiHelmet.gltf",
	"../../assets/gltf_models/DamagedHelmet/DamagedHelmet.gltf",
	"../../assets/gltf_models/WaterBottle/WaterBottle.glb",
	"../../assets/gltf_models/floor_stones_tilleable/floor_stones_tilleable.gltf",
	"../../assets/gltf_models/the_odd_sphere/the_odd_sphere.gltf",
	"../../assets/gltf_models/CesiumMan/CesiumMan.glb",
};

// Upper bound the descriptor set layout declares for the texture array; the allocation asks for what the asset provides
static constexpr uint32_t kBindlessTextureLimit = 64u;

static VkDeviceAddress getBufferAddress(Device &device, const Buffer &buffer)
{
	// The buffer has to be created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT before an address can be queried
	const VkBufferDeviceAddressInfo bufferDeviceAddressInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer.getHandle(),
	};
	return vkGetBufferDeviceAddress(device.getHandle(), &bufferDeviceAddressInfo);
}

MeshApp::MeshApp(Platform &platform, std::string name) :
	Application(platform, std::move(name))
{
}

MeshApp::~MeshApp()
{
	cleanup();

	m_device.reset();

	if (m_surface != VK_NULL_HANDLE)
	{
		vkDestroySurfaceKHR(m_instance->getHandle(), m_surface, nullptr);
		m_surface = VK_NULL_HANDLE;
	}

	m_instance.reset();
}

void MeshApp::prepare()
{
	Application::prepare();

	// Create the Vulkan instance and surface for the application
	m_instance = std::make_unique<Instance>(getName());

	platform.createSurface(m_instance->getHandle());
	m_surface = platform.getSurface();

	// Create the Vulkan physical and logical device with the required extensions
	std::unique_ptr<PhysicalDevice> physicalDevice = m_instance->getSuitablePhysicalDevice();

	m_device = std::make_unique<Device>(std::move(physicalDevice), m_surface, deviceExtensions);

	// Find general queue for graphics, compute, transfer and presentation
	int32_t desiredQueueFamilyIndex = m_device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, true);

	if (desiredQueueFamilyIndex >= 0)
	{
		m_generalQueue = m_device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
	}
	else
	{
		LOGEANDABORT("TODO: Devices where a queue supporting graphics, compute and transfer do not exist are not supported yet");
	}

	// Create swapchain
	const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT };
	m_swapchain = std::make_unique<Swapchain>(*m_device, m_surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags, m_generalQueue->getFamilyIndex(), m_generalQueue->getFamilyIndex());

	createSwapchainImageViews();
	createFrameData();
	createModels();
	createTextures();
	createDepthResources();
	createPipeline();
}

void MeshApp::createModels()
{
	static constexpr size_t kModelCount = std::size(kModelPaths);

	// Slot 0 is a white texel, so a vertex whose material has no base colour map still samples something valid
	m_textureData.emplace_back(GltfTextureData{ .pixels = std::vector<unsigned char>(4u, 255u), .width = 1u, .height = 1u });

	m_models.reserve(kModelCount);

	for (const char *modelPath : kModelPaths)
	{
		GltfAsset asset = loadGltfAsset(modelPath);

		// Slots are shared by the whole scene, so this model's images start where the previous model's ended
		const uint32_t firstTextureSlot = to_u32(m_textureData.size());
		const uint32_t textureCount = to_u32(asset.textures.size());

		for (GltfVertex &vertex : asset.mesh.vertices)
		{
			vertex.textureIndex = textureCount > 0u ? firstTextureSlot + vertex.textureIndex : 0u;
		}

		const MeshletData meshletData = buildMeshlets(asset.mesh.vertices, asset.mesh.indices);

		MeshModel model;
		model.name = std::filesystem::path(modelPath).stem().string();
		model.meshletCount = to_u32(meshletData.meshlets.size());

		// A vertex is a packed position, a texture coordinate and a texture slot, 24 bytes including the std430 padding
		model.vertexBuffer = createBuffer(meshletData.vertices.data(), meshletData.vertices.size() * sizeof(MeshletVertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		model.meshletBuffer = createBuffer(meshletData.meshlets.data(), meshletData.meshlets.size() * sizeof(meshopt_Meshlet), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		model.meshletVertexBuffer = createBuffer(meshletData.meshletVertices.data(), meshletData.meshletVertices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		model.meshletTriangleBuffer = createBuffer(meshletData.meshletTriangles.data(), meshletData.meshletTriangles.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		model.boundsBuffer = createBuffer(meshletData.bounds.data(), meshletData.bounds.size() * sizeof(MeshletBounds), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		// The task workgroup count lives in memory instead of in the draw call, so the indirect path stays usable once a
		// compute pass starts writing it
		const VkDrawMeshTasksIndirectCommandEXT indirectCommand{
			.groupCountX = model.meshletCount,
			.groupCountY = 1u,
			.groupCountZ = 1u,
		};
		model.dispatchBuffer = createBuffer(&indirectCommand, sizeof(indirectCommand),
			VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		// Buffer device address replaces the descriptor set: the mesh shader dereferences these addresses straight out of
		// its push constants, so nothing but the texture array has to be bound before the draw
		model.vertexAddress = getBufferAddress(*m_device, *model.vertexBuffer);
		model.meshletAddress = getBufferAddress(*m_device, *model.meshletBuffer);
		model.meshletVertexAddress = getBufferAddress(*m_device, *model.meshletVertexBuffer);
		model.meshletTriangleAddress = getBufferAddress(*m_device, *model.meshletTriangleBuffer);
		model.boundsAddress = getBufferAddress(*m_device, *model.boundsBuffer);

		// The images are uploaded once every model is loaded, so the asset can be released here
		for (GltfTextureData &texture : asset.textures)
		{
			m_textureData.emplace_back(std::move(texture));
		}

		m_models.emplace_back(std::move(model));
	}

	// Lay the models out on a grid so any count fits, then pull the camera back far enough to frame it
	const uint32_t columns = to_u32(std::ceil(std::sqrt(static_cast<float>(m_models.size()))));
	const uint32_t rows = to_u32((m_models.size() + columns - 1u) / columns);
	constexpr float kSpacing = 1.6f;

	for (size_t index = 0; index < m_models.size(); ++index)
	{
		const uint32_t column = to_u32(index) % columns;
		const uint32_t row = to_u32(index) / columns;
		m_models[index].translation = glm::vec3(
			(static_cast<float>(column) - static_cast<float>(columns - 1u) * 0.5f) * kSpacing,
			(static_cast<float>(rows - 1u) * 0.5f - static_cast<float>(row)) * kSpacing,
			0.0f);
	}

	const VkExtent2D extent = m_swapchain->getProperties().imageExtent;
	const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
	const float halfVertical = std::tan(glm::radians(45.0f) * 0.5f);
	const float halfHorizontal = halfVertical * aspect;
	const float halfWidth = static_cast<float>(columns) * kSpacing * 0.5f + 0.8f;
	const float halfHeight = static_cast<float>(rows) * kSpacing * 0.5f + 0.8f;
	const float distance = glm::max(halfWidth / halfHorizontal, halfHeight / halfVertical) * 1.15f;

	// The camera starts on the +Z axis far enough back to frame the whole grid, with the pivot at its centre
	m_cameraController = std::make_unique<CameraController>(extent.width, extent.height);
	std::shared_ptr<Camera> camera = m_cameraController->getCamera();
	camera->setPerspectiveProjection(45.0f, aspect, 0.1f, 100.0f);
	camera->setView(glm::vec3(0.0f, 0.0f, distance), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	m_viewProjection = camera->getProjection() * camera->getView();
	camera->resetUpdatedFlag();

	createCullCamera();
	updateCullCamera(*camera);

	LOGI("Loaded {} models into a {}x{} grid, camera at z = {:.2f}", m_models.size(), columns, rows, distance);
}

void MeshApp::createCullCamera()
{
	// Host visible so a moving camera can rewrite it every frame; the task shader reads it through its device address
	VkBufferCreateInfo bufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(CullCamera),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocationCreateInfo{ .usage = VMA_MEMORY_USAGE_CPU_ONLY };
	m_cameraBuffer = std::make_unique<Buffer>(*m_device, bufferCreateInfo, allocationCreateInfo);
	m_cameraAddress = getBufferAddress(*m_device, *m_cameraBuffer);
}

void MeshApp::updateCullCamera(const Camera &camera)
{
	// Frustum planes come from the view projection matrix: each one is a row combination, and the engine uses a 0 to 1
	// depth range, so the near plane is the third row on its own
	const glm::mat4 &viewProjection = m_viewProjection;
	const glm::vec4 rows[4]{
		glm::vec4(viewProjection[0][0], viewProjection[1][0], viewProjection[2][0], viewProjection[3][0]),
		glm::vec4(viewProjection[0][1], viewProjection[1][1], viewProjection[2][1], viewProjection[3][1]),
		glm::vec4(viewProjection[0][2], viewProjection[1][2], viewProjection[2][2], viewProjection[3][2]),
		glm::vec4(viewProjection[0][3], viewProjection[1][3], viewProjection[2][3], viewProjection[3][3]),
	};

	const glm::vec4 rawPlanes[6]{ rows[3] + rows[0], rows[3] - rows[0], rows[3] + rows[1], rows[3] - rows[1], rows[2], rows[3] - rows[2] };

	CullCamera cullCamera{};
	for (size_t planeIndex = 0; planeIndex < 6u; ++planeIndex)
	{
		const float length = glm::length(glm::vec3(rawPlanes[planeIndex]));
		cullCamera.planes[planeIndex] = rawPlanes[planeIndex] / length;
	}

	cullCamera.eyePosition = glm::vec4(camera.getPosition(), 1.0f);

	void *mappedData = m_cameraBuffer->map();
	memcpy(mappedData, &cullCamera, sizeof(CullCamera));
	m_cameraBuffer->unmap();
	m_cameraBuffer->flush();
}

void MeshApp::createTextures()
{
	if (m_textureData.empty() || m_textureData.size() > kBindlessTextureLimit)
	{
		LOGEANDABORT("The scene provides {} textures, the bindless array holds between 1 and {}", m_textureData.size(), kBindlessTextureLimit);
	}

	// One submit for every texture: staging copy into device memory, then the layout the shader samples from
	CommandBuffer commandBuffer(*m_frames[0].commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	commandBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	// The staging buffers have to stay alive until the submission has finished
	std::vector<std::unique_ptr<Buffer>> stagingBuffers;
	stagingBuffers.reserve(m_textureData.size());
	m_textureViews.reserve(m_textureData.size());

	for (const GltfTextureData &texture : m_textureData)
	{
		m_textureViews.emplace_back(createTexture(commandBuffer, texture, stagingBuffers));
	}

	commandBuffer.end();
	submitAndWait(commandBuffer);

	// One sampler for every slot, so the array holds images only and the sampler is a separate binding
	const VkSamplerCreateInfo samplerCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = 0.0f,
	};
	m_textureSampler = std::make_unique<Sampler>(*m_device, samplerCreateInfo);

	// The layout declares how large the array may become, the set is allocated with the count the asset provides, and the
	// slots above that count stay unbound, which is exactly what the partially bound flag allows
	// The variable descriptor count has to sit on the highest binding of the layout, so the sampler comes first
	const std::vector<VkDescriptorSetLayoutBinding> bindings{
		{ .binding = 0u, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1u, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 1u, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = kBindlessTextureLimit, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const std::vector<VkDescriptorBindingFlags> bindingFlags{
		VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
		VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
	};
	m_textureSetLayout = std::make_unique<DescriptorSetLayout>(*m_device, bindings, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, bindingFlags);

	std::vector<VkDescriptorPoolSize> poolSizes{
		{ .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = kBindlessTextureLimit },
		{ .type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1u },
	};
	m_texturePool = std::make_unique<DescriptorPool>(*m_device, poolSizes, 1u, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT);

	const uint32_t textureCount = to_u32(m_textureViews.size());
	const VkDescriptorSetLayout textureSetLayoutHandle = m_textureSetLayout->getHandle();
	VkDescriptorSetVariableDescriptorCountAllocateInfo variableDescriptorCountInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
		.descriptorSetCount = 1u,
		.pDescriptorCounts = &textureCount,
	};
	const VkDescriptorSetAllocateInfo allocateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = &variableDescriptorCountInfo,
		.descriptorPool = m_texturePool->getHandle(),
		.descriptorSetCount = 1u,
		.pSetLayouts = &textureSetLayoutHandle,
	};
	m_textureSet = std::make_unique<DescriptorSet>(*m_device, allocateInfo);

	std::vector<VkDescriptorImageInfo> imageInfos;
	imageInfos.reserve(m_textureViews.size());
	for (const std::unique_ptr<ImageView> &textureView : m_textureViews)
	{
		imageInfos.emplace_back(VkDescriptorImageInfo{
			.sampler = VK_NULL_HANDLE,
			.imageView = textureView->getHandle(),
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		});
	}
	const VkDescriptorImageInfo samplerInfo{ .sampler = m_textureSampler->getHandle() };

	std::array<VkWriteDescriptorSet, 2> writes{
		VkWriteDescriptorSet{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = m_textureSet->getHandle(),
			.dstBinding = 0u,
			.descriptorCount = 1u,
			.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
			.pImageInfo = &samplerInfo,
		},
		VkWriteDescriptorSet{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = m_textureSet->getHandle(),
			.dstBinding = 1u,
			.descriptorCount = textureCount,
			.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.pImageInfo = imageInfos.data(),
		},
	};
	vkUpdateDescriptorSets(m_device->getHandle(), to_u32(writes.size()), writes.data(), 0u, nullptr);

	LOGI("Bound {} textures from {} models to the bindless array, the layout allows {}", textureCount, m_models.size(), kBindlessTextureLimit);

	// The CPU copies are no longer needed once every slot is written
	m_textureData.clear();
	m_textureData.shrink_to_fit();
}

std::unique_ptr<ImageView> MeshApp::createTexture(CommandBuffer &commandBuffer, const GltfTextureData &texture,
	std::vector<std::unique_ptr<Buffer>> &stagingBuffers)
{
	const VkExtent3D extent{ texture.width, texture.height, 1u };
	auto image = std::make_unique<Image>(*m_device, VK_FORMAT_R8G8B8A8_UNORM, extent,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// The decoded texels come from the CPU visible heap
	VkBufferCreateInfo stagingBufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = texture.pixels.size(),
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo stagingAllocationCreateInfo{ .usage = VMA_MEMORY_USAGE_CPU_ONLY };
	auto stagingBuffer = std::make_unique<Buffer>(*m_device, stagingBufferCreateInfo, stagingAllocationCreateInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, texture.pixels.data(), texture.pixels.size());
	stagingBuffer->unmap();
	stagingBuffer->flush();
	stagingBuffers.emplace_back(std::move(stagingBuffer));

	const VkImageSubresourceRange subresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u };
	VkImageMemoryBarrier2 toTransferBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
		.srcAccessMask = VK_ACCESS_2_NONE,
		.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image->getHandle(),
		.subresourceRange = subresourceRange,
	};
	VkDependencyInfo toTransferDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toTransferBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer.getHandle(), &toTransferDependencyInfo);

	const VkBufferImageCopy copyRegion{
		.bufferOffset = 0u,
		.bufferRowLength = 0u,
		.bufferImageHeight = 0u,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u },
		.imageOffset = { 0, 0, 0 },
		.imageExtent = extent,
	};
	vkCmdCopyBufferToImage(commandBuffer.getHandle(), stagingBuffers.back()->getHandle(), image->getHandle(),
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &copyRegion);

	VkImageMemoryBarrier2 toSampleBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image->getHandle(),
		.subresourceRange = subresourceRange,
	};
	VkDependencyInfo toSampleDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toSampleBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer.getHandle(), &toSampleDependencyInfo);

	return std::make_unique<ImageView>(std::move(image), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R8G8B8A8_UNORM);
}

void MeshApp::createDepthResources()
{
	const VkFormat depthFormat = getSupportedDepthFormat(m_device->getPhysicalDevice().getHandle(), true);
	const VkExtent2D extent = m_swapchain->getProperties().imageExtent;
	const VkExtent3D depthExtent{ extent.width, extent.height, 1u };

	auto depthImage = std::make_unique<Image>(*m_device, depthFormat, depthExtent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	m_depthImageView = std::make_unique<ImageView>(std::move(depthImage), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, depthFormat);
}

std::unique_ptr<Buffer> MeshApp::createBuffer(const void *data, VkDeviceSize size, VkBufferUsageFlags usage)
{
	// Staging buffer on the CPU visible heap
	VkBufferCreateInfo stagingBufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo stagingAllocationCreateInfo{ .usage = VMA_MEMORY_USAGE_CPU_ONLY };
	auto stagingBuffer = std::make_unique<Buffer>(*m_device, stagingBufferCreateInfo, stagingAllocationCreateInfo);

	void *mappedData = stagingBuffer->map();
	memcpy(mappedData, data, static_cast<size_t>(size));
	stagingBuffer->unmap();
	stagingBuffer->flush();

	// Device local buffer the mesh shader reads from
	VkBufferCreateInfo bufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocationCreateInfo{ .usage = VMA_MEMORY_USAGE_GPU_ONLY };
	auto buffer = std::make_unique<Buffer>(*m_device, bufferCreateInfo, allocationCreateInfo);

	// One time submit that copies the staging buffer into device memory and waits for it to finish
	CommandBuffer commandBuffer(*m_frames[0].commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	commandBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	const VkBufferCopy copyRegion{ .srcOffset = 0u, .dstOffset = 0u, .size = size };
	vkCmdCopyBuffer(commandBuffer.getHandle(), stagingBuffer->getHandle(), buffer->getHandle(), 1u, &copyRegion);

	commandBuffer.end();
	submitAndWait(commandBuffer);

	return buffer;
}

void MeshApp::submitAndWait(CommandBuffer &commandBuffer)
{
	VkCommandBufferSubmitInfo commandBufferSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.commandBuffer = commandBuffer.getHandle(),
	};
	VkSubmitInfo2 submitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.commandBufferInfoCount = 1u,
		.pCommandBufferInfos = &commandBufferSubmitInfo,
	};

	VkFenceCreateInfo fenceCreateInfo{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence{ VK_NULL_HANDLE };
	VK_CHECK(vkCreateFence(m_device->getHandle(), &fenceCreateInfo, nullptr, &fence));
	VK_CHECK(vkQueueSubmit2(m_generalQueue->getHandle(), 1u, &submitInfo, fence));
	VK_CHECK(vkWaitForFences(m_device->getHandle(), 1u, &fence, VK_TRUE, UINT64_MAX));
	vkDestroyFence(m_device->getHandle(), fence, nullptr);
}

void MeshApp::createPipeline()
{
	// Compile the mesh and fragment entry points with Slang at runtime
	m_slangCompiler = std::make_unique<SlangShaderCompiler>();

	const std::vector<std::pair<std::string, VkShaderStageFlagBits>> entryPoints{
		{ "taskMain", VK_SHADER_STAGE_TASK_BIT_EXT },
		{ "meshMain", VK_SHADER_STAGE_MESH_BIT_EXT },
		{ "fragMain", VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = m_slangCompiler->compile(*m_device, "../../shaders/mesh/meshlets.slang", entryPoints);

	// One descriptor set holds the bindless texture array; the meshlet buffers never need one because their device
	// addresses travel in the push constant block
	const VkDescriptorSetLayout textureSetLayoutHandle = m_textureSetLayout->getHandle();
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
		.offset = 0u,
		.size = sizeof(PushConstants),
	};

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1u,
		.pSetLayouts = &textureSetLayoutHandle,
		.pushConstantRangeCount = 1u,
		.pPushConstantRanges = &pushConstantRange,
	};
	VK_CHECK(vkCreatePipelineLayout(m_device->getHandle(), &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout));

	// Dynamic rendering replaces the render pass of the pipeline; the depth format has to match the attachment
	VkPipelineRenderingCreateInfo renderingCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1u,
		.pColorAttachmentFormats = &m_swapchain->getProperties().surfaceFormat.format,
		.depthAttachmentFormat = m_depthImageView->getImage()->getFormat(),
	};

	VkPipelineViewportStateCreateInfo viewportState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1u,
		.scissorCount = 1u,
	};

	VkPipelineRasterizationStateCreateInfo rasterizationState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo multisampleState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// The projection maps the near plane to 0 and the far plane to 1, so closer fragments have smaller depth values
	VkPipelineDepthStencilStateCreateInfo depthStencilState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState colorBlendAttachment{
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	VkPipelineColorBlendStateCreateInfo colorBlendState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1u,
		.pAttachments = &colorBlendAttachment,
	};

	const VkDynamicState dynamicStates[]{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = to_u32(std::size(dynamicStates)),
		.pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo pipelineCreateInfo{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &renderingCreateInfo,
		.stageCount = to_u32(shaderStages.size()),
		.pStages = shaderStages.data(),
		// pVertexInputState and pInputAssemblyState stay null since a mesh shader pipeline has no vertex buffers
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizationState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = &depthStencilState,
		.pColorBlendState = &colorBlendState,
		.pDynamicState = &dynamicState,
		.layout = m_pipelineLayout,
		.renderPass = VK_NULL_HANDLE,
	};
	VK_CHECK(vkCreateGraphicsPipelines(m_device->getHandle(), VK_NULL_HANDLE, 1u, &pipelineCreateInfo, nullptr, &m_pipeline));

	LOGI("Created the mesh shader pipeline");
}

void MeshApp::createSwapchainImageViews()
{
	m_swapchainImageViews.reserve(m_swapchain->getImages().size());

	for (const std::unique_ptr<Image> &swapchainImage : m_swapchain->getImages())
	{
		VkImageViewCreateInfo imageViewCreateInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchainImage->getHandle(),
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = m_swapchain->getProperties().surfaceFormat.format,
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u },
		};

		VkImageView imageView{ VK_NULL_HANDLE };
		VK_CHECK(vkCreateImageView(m_device->getHandle(), &imageViewCreateInfo, nullptr, &imageView));

		m_swapchainImageViews.emplace_back(imageView);
	}
}

void MeshApp::createFrameData()
{
	m_fencePool = std::make_unique<FencePool>(*m_device);
	m_semaphorePool = std::make_unique<SemaphorePool>(*m_device);

	for (FrameData &frame : m_frames)
	{
		frame.commandPool = std::make_unique<CommandPool>(*m_device, m_generalQueue->getFamilyIndex(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		frame.commandBuffer = std::make_shared<CommandBuffer>(*frame.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		frame.inFlightFence = m_fencePool->requestFence();
		frame.imageAvailableSemaphore = m_semaphorePool->requestSemaphore();
	}

	// One render finished semaphore per swapchain image: presentation may still be using an image by the time the frame
	// that signalled its semaphore comes around again, so the semaphore cannot belong to a frame
	m_renderFinishedSemaphores.resize(m_swapchainImageViews.size(), VK_NULL_HANDLE);
	for (VkSemaphore &renderFinishedSemaphore : m_renderFinishedSemaphores)
	{
		renderFinishedSemaphore = m_semaphorePool->requestSemaphore();
	}
}

void MeshApp::update()
{
	m_cameraController->update(getDeltaTime());

	// The camera marks itself updated when keyboard or mouse input moved it; rebuilding the matrix only then keeps the
	// per-frame cost off the render thread
	std::shared_ptr<Camera> camera = m_cameraController->getCamera();
	if (camera->isUpdated())
	{
		m_viewProjection = camera->getProjection() * camera->getView();
		updateCullCamera(*camera);
		camera->resetUpdatedFlag();
	}

	FrameData &frame = m_frames[m_currentFrame];

	// Nothing may be recorded into this frame's command buffer until the GPU is done with it
	m_fencePool->wait(&frame.inFlightFence);
	m_fencePool->reset(&frame.inFlightFence);

	VkResult result = vkAcquireNextImageKHR(m_device->getHandle(), m_swapchain->getHandle(), std::numeric_limits<uint64_t>::max(), frame.imageAvailableSemaphore, VK_NULL_HANDLE, &m_swapchainImageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		LOGEANDABORT("Failed to acquire the next swapchain image");
	}

	recordCommandBuffer();

	// The acquire semaphore is waited on at the color attachment stage because that is the first stage that touches the image
	VkSemaphoreSubmitInfo imageAvailableSemaphoreSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = frame.imageAvailableSemaphore,
		.value = 0u, // Ignored for binary semaphores
		.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.deviceIndex = 0u,
	};
	VkSemaphoreSubmitInfo renderFinishedSemaphoreSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = m_renderFinishedSemaphores[m_swapchainImageIndex],
		.value = 0u, // Ignored for binary semaphores
		.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.deviceIndex = 0u,
	};
	VkCommandBufferSubmitInfo commandBufferSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.commandBuffer = frame.commandBuffer->getHandle(),
		.deviceMask = 0u,
	};
	VkSubmitInfo2 submitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.waitSemaphoreInfoCount = 1u,
		.pWaitSemaphoreInfos = &imageAvailableSemaphoreSubmitInfo,
		.commandBufferInfoCount = 1u,
		.pCommandBufferInfos = &commandBufferSubmitInfo,
		.signalSemaphoreInfoCount = 1u,
		.pSignalSemaphoreInfos = &renderFinishedSemaphoreSubmitInfo,
	};
	VK_CHECK(vkQueueSubmit2(m_generalQueue->getHandle(), 1u, &submitInfo, frame.inFlightFence));

	const VkSwapchainKHR swapchainHandle = m_swapchain->getHandle();
	VkPresentInfoKHR presentInfo{
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1u,
		.pWaitSemaphores = &m_renderFinishedSemaphores[m_swapchainImageIndex],
		.swapchainCount = 1u,
		.pSwapchains = &swapchainHandle,
		.pImageIndices = &m_swapchainImageIndex,
	};

	result = vkQueuePresentKHR(m_generalQueue->getHandle(), &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		recreateSwapchain();
	}
	else if (result != VK_SUCCESS)
	{
		LOGEANDABORT("Failed to present the swapchain image");
	}

	m_rotation += 0.01f;

	m_currentFrame = (m_currentFrame + 1u) % maxFramesInFlight;
}

void MeshApp::handleInputEvents(const InputEvent &inputEvent)
{
	m_cameraController->handleInputEvents(inputEvent);
}

void MeshApp::handleFocusChange(bool isFocused)
{
	Application::handleFocusChange(isFocused);

	if (!isFocused && m_cameraController)
	{
		m_cameraController->resetInput();
	}
}

void MeshApp::recordCommandBuffer()
{
	FrameData &frame = m_frames[m_currentFrame];
	VkCommandBuffer commandBuffer = frame.commandBuffer->getHandle();
	frame.commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	const VkImage swapchainImage = m_swapchain->getImages()[m_swapchainImageIndex]->getHandle();
	const VkImageSubresourceRange subresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u };

	// Take ownership of the image from the presentation engine. The acquire semaphore is waited on at the color attachment
	// stage, so the transition has to be ordered against that stage for the layout transition not to race with the acquire
	VkImageMemoryBarrier2 toColorAttachmentBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_NONE,
		.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchainImage,
		.subresourceRange = subresourceRange,
	};
	VkDependencyInfo toColorAttachmentDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toColorAttachmentBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toColorAttachmentDependencyInfo);

	// The depth image is shared by both frames in flight and cleared every frame, so discard its contents and order
	// the transition after the previous frame's late fragment tests
	const VkImageSubresourceRange depthSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 1u, 0u, 1u };
	VkImageMemoryBarrier2 toDepthAttachmentBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = m_depthImageView->getImage()->getHandle(),
		.subresourceRange = depthSubresourceRange,
	};
	VkDependencyInfo toDepthAttachmentDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toDepthAttachmentBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toDepthAttachmentDependencyInfo);

	// Clear the image without a render pass or a framebuffer
	VkClearValue clearColor{};
	clearColor.color = { { 0.05f, 0.05f, 0.08f, 1.0f } };

	VkRenderingAttachmentInfo colorAttachment{
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = m_swapchainImageViews[m_swapchainImageIndex],
		.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = clearColor,
	};
	VkClearValue clearDepth{};
	clearDepth.depthStencil = { 1.0f, 0u };

	VkRenderingAttachmentInfo depthAttachment{
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = m_depthImageView->getHandle(),
		.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.clearValue = clearDepth,
	};

	VkRenderingInfo renderingInfo{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { { 0u, 0u }, m_swapchain->getProperties().imageExtent },
		.layerCount = 1u,
		.colorAttachmentCount = 1u,
		.pColorAttachments = &colorAttachment,
		.pDepthAttachment = &depthAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &renderingInfo);

	// GLM builds an OpenGL style projection with Y pointing up while Vulkan's clip space has Y pointing down; Camera
	// already flips the projection's Y axis, so the viewport keeps a positive height
	const VkExtent2D swapchainExtent = m_swapchain->getProperties().imageExtent;
	const VkViewport viewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(swapchainExtent.width),
		.height = static_cast<float>(swapchainExtent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor{ { 0, 0 }, swapchainExtent };

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
	vkCmdSetViewport(commandBuffer, 0u, 1u, &viewport);
	vkCmdSetScissor(commandBuffer, 0u, 1u, &scissor);
	const VkDescriptorSet textureSetHandle = m_textureSet->getHandle();
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0u, 1u, &textureSetHandle, 0u, nullptr);
	// One draw per model; the push constants carry that model's addresses and where it sits in the grid
	for (const MeshModel &model : m_models)
	{
		const PushConstants pushConstants{
			.viewProjection = m_viewProjection,
			.model = glm::rotate(glm::translate(glm::mat4(1.0f), model.translation), m_rotation, glm::vec3(0.0f, 1.0f, 0.0f)),
			.vertices = model.vertexAddress,
			.meshlets = model.meshletAddress,
			.meshletVertices = model.meshletVertexAddress,
			.meshletTriangles = model.meshletTriangleAddress,
			.bounds = model.boundsAddress,
			.camera = m_cameraAddress,
			.meshletCount = model.meshletCount,
		};
		vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0u, sizeof(PushConstants), &pushConstants);

		// One task workgroup per meshlet, read from the buffer so a compute pass could rewrite the count later; the task
		// shader drops the clusters that cannot be seen and only those reach the mesh shader
		vkCmdDrawMeshTasksIndirectEXT(commandBuffer, model.dispatchBuffer->getHandle(), 0u, 1u, sizeof(VkDrawMeshTasksIndirectCommandEXT));
	}

	vkCmdEndRendering(commandBuffer);

	// Hand the image back to the presentation engine
	VkImageMemoryBarrier2 toPresentBarrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
		.dstAccessMask = VK_ACCESS_2_NONE,
		.oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchainImage,
		.subresourceRange = subresourceRange,
	};
	VkDependencyInfo toPresentDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1u,
		.pImageMemoryBarriers = &toPresentBarrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toPresentDependencyInfo);

	frame.commandBuffer->end();
}

void MeshApp::recreateSwapchain()
{
	// The platform window is not resizable, so rebuilding the swapchain is not needed yet
	LOGW("Swapchain is out of date; recreation is not implemented yet");
}

void MeshApp::cleanup()
{
	if (m_device == nullptr)
	{
		return;
	}

	m_device->waitIdle();

	if (m_pipeline != VK_NULL_HANDLE)
	{
		vkDestroyPipeline(m_device->getHandle(), m_pipeline, nullptr);
		m_pipeline = VK_NULL_HANDLE;
	}

	if (m_pipelineLayout != VK_NULL_HANDLE)
	{
		vkDestroyPipelineLayout(m_device->getHandle(), m_pipelineLayout, nullptr);
		m_pipelineLayout = VK_NULL_HANDLE;
	}

	if (m_slangCompiler != nullptr)
	{
		m_slangCompiler->destroyShaderModules(*m_device);
		m_slangCompiler.reset();
	}

	m_textureSet.reset();
	m_texturePool.reset();
	m_textureSetLayout.reset();
	m_textureSampler.reset();
	m_textureViews.clear();

	m_models.clear();

	// The camera cull data is a host visible buffer, so it holds a VMA allocation like every other resource here
	m_cameraBuffer.reset();

	m_depthImageView.reset();

	for (VkImageView imageView : m_swapchainImageViews)
	{
		vkDestroyImageView(m_device->getHandle(), imageView, nullptr);
	}
	m_swapchainImageViews.clear();
	m_renderFinishedSemaphores.clear();

	// A CommandBuffer holds a reference to its CommandPool, so the command buffers have to be released before the pools.
	// Assigning a default constructed FrameData here would release the members in declaration order (pool first) instead
	for (FrameData &frame : m_frames)
	{
		frame.commandBuffer.reset();
		frame.commandPool.reset();
	}

	m_fencePool.reset();
	m_semaphorePool.reset();

	m_swapchain.reset();
}

} // namespace vulkr
