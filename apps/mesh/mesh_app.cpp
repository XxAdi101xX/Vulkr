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

#include <cstring>
#include <iterator>
#include <limits>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "common/logger.h"
#include "common/helpers.h"
#include "platform/platform.h"

#include "rendering/slang_shader_compiler.h"

#include "core/buffer.h"
#include "core/command_buffer.h"
#include "core/command_pool.h"
#include "core/descriptor_pool.h"
#include "core/descriptor_set.h"
#include "core/descriptor_set_layout.h"
#include "core/device.h"
#include "core/fence_pool.h"
#include "core/image.h"
#include "core/image_view.h"
#include "core/instance.h"
#include "core/queue.h"
#include "core/semaphore_pool.h"
#include "core/swapchain.h"

#include "gltf_mesh.h"
#include "meshlet_builder.h"

namespace vulkr
{

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
	createMesh();
	createDescriptors();
	createDepthResources();
	createPipeline();
}

void MeshApp::createMesh()
{
	// The asset is normalized into a unit cube by the loader, so a fixed camera can frame it
	const GltfMeshData gltfMesh = loadGltfMesh("../../assets/gltf_models/SciFiHelmet/SciFiHelmet.gltf");
	const MeshletData meshletData = buildMeshlets(gltfMesh.positions, gltfMesh.indices);

	m_meshletCount = to_u32(meshletData.meshlets.size());

	// The builder quantizes positions to two packed 16 bit signed normalized components per word, so a vertex is 8 bytes
	m_positionBuffer = createBuffer(meshletData.packedPositions.data(), meshletData.packedPositions.size() * sizeof(glm::uvec2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_meshletBuffer = createBuffer(meshletData.meshlets.data(), meshletData.meshlets.size() * sizeof(meshopt_Meshlet), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_meshletVertexBuffer = createBuffer(meshletData.meshletVertices.data(), meshletData.meshletVertices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_meshletTriangleBuffer = createBuffer(meshletData.meshletTriangles.data(), meshletData.meshletTriangles.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);



	const VkExtent2D extent = m_swapchain->getProperties().imageExtent;
	const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
	const glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
	const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	m_pushConstants.viewProjection = projection * view;
	m_pushConstants.model = glm::mat4(1.0f);
}

void MeshApp::createDescriptors()
{
	const std::vector<VkDescriptorSetLayoutBinding> bindings{
		{ .binding = 0u, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1u, .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT },
		{ .binding = 1u, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1u, .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT },
		{ .binding = 2u, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1u, .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT },
		{ .binding = 3u, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1u, .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT },
	};
	m_descriptorSetLayout = std::make_unique<DescriptorSetLayout>(*m_device, bindings);

	std::vector<VkDescriptorPoolSize> poolSizes{ { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4u } };
	m_descriptorPool = std::make_unique<DescriptorPool>(*m_device, poolSizes, 1u, 0u);

	const VkDescriptorSetLayout descriptorSetLayoutHandle = m_descriptorSetLayout->getHandle();
	VkDescriptorSetAllocateInfo allocateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = m_descriptorPool->getHandle(),
		.descriptorSetCount = 1u,
		.pSetLayouts = &descriptorSetLayoutHandle,
	};
	m_descriptorSet = std::make_unique<DescriptorSet>(*m_device, allocateInfo);

	const std::array<VkDescriptorBufferInfo, 4> bufferInfos{
		VkDescriptorBufferInfo{ .buffer = m_positionBuffer->getHandle(), .offset = 0u, .range = VK_WHOLE_SIZE },
		VkDescriptorBufferInfo{ .buffer = m_meshletBuffer->getHandle(), .offset = 0u, .range = VK_WHOLE_SIZE },
		VkDescriptorBufferInfo{ .buffer = m_meshletVertexBuffer->getHandle(), .offset = 0u, .range = VK_WHOLE_SIZE },
		VkDescriptorBufferInfo{ .buffer = m_meshletTriangleBuffer->getHandle(), .offset = 0u, .range = VK_WHOLE_SIZE },
	};

	std::array<VkWriteDescriptorSet, 4> writes{};
	for (size_t binding = 0; binding < writes.size(); ++binding)
	{
		writes[binding] = VkWriteDescriptorSet{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = m_descriptorSet->getHandle(),
			.dstBinding = to_u32(binding),
			.descriptorCount = 1u,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &bufferInfos[binding],
		};
	}

	vkUpdateDescriptorSets(m_device->getHandle(), to_u32(writes.size()), writes.data(), 0u, nullptr);
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
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

	return buffer;
}

void MeshApp::createPipeline()
{
	// Compile the mesh and fragment entry points with Slang at runtime
	m_slangCompiler = std::make_unique<SlangShaderCompiler>();

	const std::vector<std::pair<std::string, VkShaderStageFlagBits>> entryPoints{
		{ "meshMain", VK_SHADER_STAGE_MESH_BIT_EXT },
		{ "fragMain", VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = m_slangCompiler->compile(*m_device, "../../shaders/mesh/helmet.slang", entryPoints);

	// One descriptor set with the meshlet buffers and a push constant block holding the matrices
	const VkDescriptorSetLayout descriptorSetLayoutHandle = m_descriptorSetLayout->getHandle();
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
		.offset = 0u,
		.size = sizeof(PushConstants),
	};
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1u,
		.pSetLayouts = &descriptorSetLayoutHandle,
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
	m_pushConstants.model = glm::rotate(glm::mat4(1.0f), m_rotation, glm::vec3(0.0f, 1.0f, 0.0f));

	m_currentFrame = (m_currentFrame + 1u) % maxFramesInFlight;
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

	// GLM builds an OpenGL style projection with Y pointing up, while Vulkan's clip space has Y pointing down. A negative
	// viewport height flips the Y axis at rasterization time (core Vulkan 1.1 through VK_KHR_maintenance1), which also
	// mirrors the triangle winding, so front faces become back faces
	const VkExtent2D swapchainExtent = m_swapchain->getProperties().imageExtent;
	const VkViewport viewport{
		.x = 0.0f,
		.y = static_cast<float>(swapchainExtent.height),
		.width = static_cast<float>(swapchainExtent.width),
		.height = -static_cast<float>(swapchainExtent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor{ { 0, 0 }, swapchainExtent };

	const VkDescriptorSet descriptorSetHandle = m_descriptorSet->getHandle();

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
	vkCmdSetViewport(commandBuffer, 0u, 1u, &viewport);
	vkCmdSetScissor(commandBuffer, 0u, 1u, &scissor);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0u, 1u, &descriptorSetHandle, 0u, nullptr);
	vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_MESH_BIT_EXT, 0u, sizeof(PushConstants), &m_pushConstants);

	// One mesh shader workgroup per meshlet; the mesh shader reads its cluster from gl_WorkGroupID
	vkCmdDrawMeshTasksEXT(commandBuffer, m_meshletCount, 1u, 1u);

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

	// The descriptor set is owned by its pool, so it is released before the pool and the layouts it uses
	m_descriptorSet.reset();
	m_descriptorPool.reset();
	m_descriptorSetLayout.reset();

	m_meshletTriangleBuffer.reset();
	m_meshletVertexBuffer.reset();
	m_meshletBuffer.reset();
	m_positionBuffer.reset();

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
