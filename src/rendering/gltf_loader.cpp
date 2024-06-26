/**
* Vulkan glTF model and texture loading class based on tinyglTF (https://github.com/syoyo/tinygltf)
*
* Copyright (C) 2018-2022 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
* Note that this file has been modified to integrate into the Vulkr engine, but the core logic remains the same and is covered under the license above.
*/

#define TINYGLTF_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT

#include "gltf_loader.h"
#include "core/device.h"
#include "core/physical_device.h"
#include "core/command_buffer.h"
#include "core/buffer.h"
#include "core/image.h"

namespace vulkr::gltf
{

/**
* Finish command buffer recording and submit it to a queue
*
* @param device: The device that's running vulkan
* @param commandBuffer: Command buffer to flush
* @param queue: Queue to submit the command buffer to
*
* @note The queue that the command buffer is submitted to must be from the same family index as the pool it was allocated from
* @note Uses a fence to ensure command buffer has finished executing
*/
void flushCommandBuffer(vulkr::Device *device, vulkr::CommandBuffer *commandBuffer, VkQueue queue)
{
	VK_CHECK(vkEndCommandBuffer(commandBuffer->getHandle()));

	VkCommandBufferSubmitInfo commandBufferSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	commandBufferSubmitInfo.pNext = nullptr;
	commandBufferSubmitInfo.commandBuffer = commandBuffer->getHandle();
	commandBufferSubmitInfo.deviceMask = 0u;

	VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.pNext = nullptr;
	submitInfo.waitSemaphoreInfoCount = 0u;
	submitInfo.pWaitSemaphoreInfos = nullptr;
	submitInfo.commandBufferInfoCount = 1u;
	submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
	submitInfo.signalSemaphoreInfoCount = 0u;
	submitInfo.pSignalSemaphoreInfos = nullptr;

	VK_CHECK(vkQueueSubmit2KHR(queue, 1u, &submitInfo, VK_NULL_HANDLE));
	vkQueueWaitIdle(queue);
}

// Bounding box

BoundingBox::BoundingBox()
{
};

BoundingBox::BoundingBox(glm::vec3 min, glm::vec3 max) : min(min), max(max)
{
};

BoundingBox BoundingBox::getAABB(glm::mat4 m)
{
	glm::vec3 min = glm::vec3(m[3]);
	glm::vec3 max = min;
	glm::vec3 v0, v1;

	glm::vec3 right = glm::vec3(m[0]);
	v0 = right * this->min.x;
	v1 = right * this->max.x;
	min += glm::min(v0, v1);
	max += glm::max(v0, v1);

	glm::vec3 up = glm::vec3(m[1]);
	v0 = up * this->min.y;
	v1 = up * this->max.y;
	min += glm::min(v0, v1);
	max += glm::max(v0, v1);

	glm::vec3 back = glm::vec3(m[2]);
	v0 = back * this->min.z;
	v1 = back * this->max.z;
	min += glm::min(v0, v1);
	max += glm::max(v0, v1);

	return BoundingBox(min, max);
}

// Texture
void Texture::updateDescriptor()
{
	descriptor.sampler = sampler;
	descriptor.imageView = view;
	descriptor.imageLayout = imageLayout;
}

void Texture::destroy()
{
	vkDestroyImageView(device->getHandle(), view, nullptr);
	vmaDestroyImage(device->getMemoryAllocator(), image, allocation);
	vkDestroySampler(device->getHandle(), sampler, nullptr);
}

void Texture::fromglTfImage(tinygltf::Image &gltfimage, TextureSampler textureSampler, vulkr::Device *device, vulkr::CommandPool *commandPool, VkQueue copyQueue)
{
	std::unique_ptr<CommandBuffer> copyCommandBuffer = std::make_unique<CommandBuffer>(*commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	copyCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
	this->device = device;

	unsigned char *buffer = nullptr;
	VkDeviceSize bufferSize = 0;
	bool deleteBuffer = false;
	if (gltfimage.component == 3)
	{
		// Most devices don't support RGB only on Vulkan so convert if necessary
		// TODO: Check actual format support and transform only if required
		bufferSize = gltfimage.width * gltfimage.height * 4;
		buffer = new unsigned char[bufferSize];
		unsigned char *rgba = buffer;
		unsigned char *rgb = &gltfimage.image[0];
		for (int32_t i = 0; i < gltfimage.width * gltfimage.height; ++i)
		{
			for (int32_t j = 0; j < 3; ++j)
			{
				rgba[j] = rgb[j];
			}
			rgba += 4;
			rgb += 3;
		}
		deleteBuffer = true;
	}
	else
	{
		buffer = &gltfimage.image[0];
		bufferSize = gltfimage.image.size();
	}

	VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

	VkFormatProperties formatProperties;

	width = gltfimage.width;
	height = gltfimage.height;
	mipLevels = static_cast<uint32_t>(floor(log2(std::max(width, height))) + 1.0);

	vkGetPhysicalDeviceFormatProperties(device->getPhysicalDevice().getHandle(), format, &formatProperties);
	assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT);
	assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);


	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	std::unique_ptr<Buffer> stagingBuffer = std::make_unique<Buffer>(*device, bufferInfo, memoryInfo);

	stagingBuffer->update(buffer, bufferSize);

	VkImageCreateInfo imageCreateInfo{};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.format = format;
	imageCreateInfo.mipLevels = mipLevels;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.extent = { width, height, 1 };
	imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	VmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	if (VMA_MEMORY_USAGE_GPU_ONLY & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
	{
		allocationInfo.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
	}
	VK_CHECK(vmaCreateImage(device->getMemoryAllocator(), &imageCreateInfo, &allocationInfo, &image, &allocation, nullptr));

	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;

	{
		VkImageMemoryBarrier2 imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		imageMemoryBarrier.pNext = nullptr;
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.image = image;
		imageMemoryBarrier.subresourceRange = subresourceRange;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = 1u;
		dependencyInfo.pImageMemoryBarriers = &imageMemoryBarrier;
		vkCmdPipelineBarrier2KHR(copyCommandBuffer->getHandle(), &dependencyInfo);
	}

	VkBufferImageCopy bufferCopyRegion = {};
	bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	bufferCopyRegion.imageSubresource.mipLevel = 0;
	bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
	bufferCopyRegion.imageSubresource.layerCount = 1;
	bufferCopyRegion.imageExtent.width = width;
	bufferCopyRegion.imageExtent.height = height;
	bufferCopyRegion.imageExtent.depth = 1;

	vkCmdCopyBufferToImage(copyCommandBuffer->getHandle(), stagingBuffer->getHandle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferCopyRegion);

	{
		VkImageMemoryBarrier2 imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		imageMemoryBarrier.pNext = nullptr;
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.image = image;
		imageMemoryBarrier.subresourceRange = subresourceRange;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = 1u;
		dependencyInfo.pImageMemoryBarriers = &imageMemoryBarrier;
		vkCmdPipelineBarrier2KHR(copyCommandBuffer->getHandle(), &dependencyInfo);
	}

	flushCommandBuffer(device, copyCommandBuffer.get(), copyQueue);
	copyCommandBuffer.reset();

	stagingBuffer.reset();

	// Generate the mip chain (glTF uses jpg and png, so we need to create this manually)
	std::unique_ptr<CommandBuffer> blitCommandBuffer = std::make_unique<CommandBuffer>(*commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	blitCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);
	for (uint32_t i = 1; i < mipLevels; i++)
	{
		VkImageBlit2 imageBlit{ VK_STRUCTURE_TYPE_IMAGE_BLIT_2 };
		imageBlit.pNext = nullptr;

		imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBlit.srcSubresource.layerCount = 1;
		imageBlit.srcSubresource.mipLevel = i - 1;
		imageBlit.srcOffsets[1].x = int32_t(width >> (i - 1));
		imageBlit.srcOffsets[1].y = int32_t(height >> (i - 1));
		imageBlit.srcOffsets[1].z = 1;

		imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBlit.dstSubresource.layerCount = 1;
		imageBlit.dstSubresource.mipLevel = i;
		imageBlit.dstOffsets[1].x = int32_t(width >> i);
		imageBlit.dstOffsets[1].y = int32_t(height >> i);
		imageBlit.dstOffsets[1].z = 1;

		VkImageSubresourceRange mipSubRange = {};
		mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		mipSubRange.baseMipLevel = i;
		mipSubRange.levelCount = 1;
		mipSubRange.layerCount = 1;

		{
			VkImageMemoryBarrier2 imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
			imageMemoryBarrier.pNext = nullptr;
			imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
			imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = mipSubRange; // That we're transitioning this for the specified mips level

			VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dependencyInfo.pNext = nullptr;
			dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
			dependencyInfo.memoryBarrierCount = 0u;
			dependencyInfo.pMemoryBarriers = nullptr;
			dependencyInfo.bufferMemoryBarrierCount = 0u;
			dependencyInfo.pBufferMemoryBarriers = nullptr;
			dependencyInfo.imageMemoryBarrierCount = 1u;
			dependencyInfo.pImageMemoryBarriers = &imageMemoryBarrier;
			vkCmdPipelineBarrier2KHR(blitCommandBuffer->getHandle(), &dependencyInfo);
		}


		VkBlitImageInfo2 blitImageInfo{ VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2 };
		blitImageInfo.pNext = nullptr;
		blitImageInfo.srcImage = image;
		blitImageInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		blitImageInfo.dstImage = image;
		blitImageInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		blitImageInfo.regionCount = 1u;
		blitImageInfo.pRegions = &imageBlit;
		blitImageInfo.filter = VK_FILTER_LINEAR;
		vkCmdBlitImage2(blitCommandBuffer->getHandle(), &blitImageInfo);

		{
			VkImageMemoryBarrier2 imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
			imageMemoryBarrier.pNext = nullptr;
			imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = mipSubRange;

			VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dependencyInfo.pNext = nullptr;
			dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
			dependencyInfo.memoryBarrierCount = 0u;
			dependencyInfo.pMemoryBarriers = nullptr;
			dependencyInfo.bufferMemoryBarrierCount = 0u;
			dependencyInfo.pBufferMemoryBarriers = nullptr;
			dependencyInfo.imageMemoryBarrierCount = 1u;
			dependencyInfo.pImageMemoryBarriers = &imageMemoryBarrier;
			vkCmdPipelineBarrier2KHR(blitCommandBuffer->getHandle(), &dependencyInfo);
		}
	}

	subresourceRange.levelCount = mipLevels;
	imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	{
		VkImageMemoryBarrier2 imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		imageMemoryBarrier.pNext = nullptr;
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.image = image;
		imageMemoryBarrier.subresourceRange = subresourceRange;

		VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependencyInfo.pNext = nullptr;
		dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencyInfo.memoryBarrierCount = 0u;
		dependencyInfo.pMemoryBarriers = nullptr;
		dependencyInfo.bufferMemoryBarrierCount = 0u;
		dependencyInfo.pBufferMemoryBarriers = nullptr;
		dependencyInfo.imageMemoryBarrierCount = 1u;
		dependencyInfo.pImageMemoryBarriers = &imageMemoryBarrier;
		vkCmdPipelineBarrier2KHR(blitCommandBuffer->getHandle(), &dependencyInfo);
	}

	flushCommandBuffer(device, blitCommandBuffer.get(), copyQueue);
	blitCommandBuffer.reset();

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = textureSampler.magFilter;
	samplerInfo.minFilter = textureSampler.minFilter;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.addressModeU = textureSampler.addressModeU;
	samplerInfo.addressModeV = textureSampler.addressModeV;
	samplerInfo.addressModeW = textureSampler.addressModeW;
	samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	samplerInfo.maxAnisotropy = 1.0;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.maxLod = (float)mipLevels;
	samplerInfo.maxAnisotropy = 8.0f;
	samplerInfo.anisotropyEnable = VK_TRUE;
	VK_CHECK(vkCreateSampler(device->getHandle(), &samplerInfo, nullptr, &sampler));

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.layerCount = 1;
	viewInfo.subresourceRange.levelCount = mipLevels;
	VK_CHECK(vkCreateImageView(device->getHandle(), &viewInfo, nullptr, &view));

	descriptor.sampler = sampler;
	descriptor.imageView = view;
	descriptor.imageLayout = imageLayout;

	if (deleteBuffer)
		delete[] buffer;

}

// Primitive
Primitive::Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material &material) : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material)
{
	hasIndices = indexCount > 0;
};

void Primitive::setBoundingBox(glm::vec3 min, glm::vec3 max)
{
	bb.min = min;
	bb.max = max;
	bb.valid = true;
}

// Mesh
Mesh::Mesh(vulkr::Device *device, glm::mat4 matrix)
{
	this->device = device;
	this->uniformBlock.matrix = matrix;

	VkDeviceSize bufferSize{ sizeof(uniformBlock) };

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	uniformBuffer.buffer = std::make_unique<vulkr::Buffer>(*device, bufferInfo, memoryInfo);
	uniformBuffer.buffer->update(&uniformBlock, bufferSize);

	uniformBuffer.descriptor = { uniformBuffer.buffer->getHandle(), 0, sizeof(uniformBlock) };
};

Mesh::~Mesh()
{
	uniformBuffer.buffer.reset();
	for (Primitive *p : primitives)
	{
		delete p;
	}
}

void Mesh::setBoundingBox(glm::vec3 min, glm::vec3 max)
{
	bb.min = min;
	bb.max = max;
	bb.valid = true;
}

// Node
glm::mat4 Node::localMatrix()
{
	return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale) * matrix;
}

glm::mat4 Node::getMatrix()
{
	glm::mat4 m = localMatrix();
	Node *p = parent;
	while (p)
	{
		m = p->localMatrix() * m;
		p = p->parent;
	}
	return m;
}

void Node::update()
{
	if (mesh)
	{
		glm::mat4 m = getMatrix();
		if (skin)
		{
			mesh->uniformBlock.matrix = m;
			// Update join matrices
			glm::mat4 inverseTransform = glm::inverse(m);
			size_t numJoints = std::min((uint32_t)skin->joints.size(), MAX_NUM_JOINTS);
			for (size_t i = 0; i < numJoints; i++)
			{
				Node *jointNode = skin->joints[i];
				glm::mat4 jointMat = jointNode->getMatrix() * skin->inverseBindMatrices[i];
				jointMat = inverseTransform * jointMat;
				mesh->uniformBlock.jointMatrix[i] = jointMat;
			}
			mesh->uniformBlock.jointcount = (float)numJoints;
			mesh->uniformBuffer.buffer->update(&mesh->uniformBlock, sizeof(mesh->uniformBlock));
		}
		else
		{
			mesh->uniformBuffer.buffer->update(&m, sizeof(m));
		}
	}

	for (auto &child : children)
	{
		child->update();
	}
}

Node::~Node()
{
	if (mesh)
	{
		delete mesh;
	}
	for (auto &child : children)
	{
		delete child;
	}
}

// Model

void Model::destroy(vulkr::Device *device)
{
	vertexBuffer.reset();
	indexBuffer.reset();

	for (auto texture : textures)
	{
		texture.destroy();
	}
	textures.resize(0);
	textureSamplers.resize(0);
	for (auto node : nodes)
	{
		delete node;
	}
	materials.resize(0);
	animations.resize(0);
	nodes.resize(0);
	linearNodes.resize(0);
	extensions.resize(0);
	for (auto skin : skins)
	{
		delete skin;
	}
	skins.resize(0);
};

void Model::loadNode(Node *parent, const tinygltf::Node &node, uint32_t nodeIndex, const tinygltf::Model &model, LoaderInfo &loaderInfo, float globalscale)
{
	Node *newNode = new Node{};
	newNode->index = nodeIndex;
	newNode->parent = parent;
	newNode->name = node.name;
	newNode->skinIndex = node.skin;
	newNode->matrix = glm::mat4(1.0f);

	// Generate local node matrix
	glm::vec3 translation = glm::vec3(0.0f);
	if (node.translation.size() == 3)
	{
		translation = glm::make_vec3(node.translation.data());
		newNode->translation = translation;
	}
	glm::mat4 rotation = glm::mat4(1.0f);
	if (node.rotation.size() == 4)
	{
		glm::quat q = glm::make_quat(node.rotation.data());
		newNode->rotation = glm::mat4(q);
	}
	glm::vec3 scale = glm::vec3(1.0f);
	if (node.scale.size() == 3)
	{
		scale = glm::make_vec3(node.scale.data());
		newNode->scale = scale;
	}
	if (node.matrix.size() == 16)
	{
		newNode->matrix = glm::make_mat4x4(node.matrix.data());
	};

	// Node with children
	if (node.children.size() > 0)
	{
		for (size_t i = 0; i < node.children.size(); i++)
		{
			loadNode(newNode, model.nodes[node.children[i]], node.children[i], model, loaderInfo, globalscale);
		}
	}

	// Node contains mesh data
	if (node.mesh > -1)
	{
		const tinygltf::Mesh mesh = model.meshes[node.mesh];
		Mesh *newMesh = new Mesh(device, newNode->matrix);
		for (size_t j = 0; j < mesh.primitives.size(); j++)
		{
			const tinygltf::Primitive &primitive = mesh.primitives[j];
			uint32_t vertexStart = static_cast<uint32_t>(loaderInfo.vertexPos);
			uint32_t indexStart = static_cast<uint32_t>(loaderInfo.indexPos);
			uint32_t indexCount = 0;
			uint32_t vertexCount = 0;
			glm::vec3 posMin{};
			glm::vec3 posMax{};
			bool hasSkin = false;
			bool hasIndices = primitive.indices > -1;
			// Vertices
			{
				const float *bufferPos = nullptr;
				const float *bufferNormals = nullptr;
				const float *bufferTexCoordSet0 = nullptr;
				const float *bufferTexCoordSet1 = nullptr;
				const float *bufferColorSet0 = nullptr;
				const void *bufferJoints = nullptr;
				const float *bufferWeights = nullptr;

				int posByteStride;
				int normByteStride;
				int uv0ByteStride;
				int uv1ByteStride;
				int color0ByteStride;
				int jointByteStride;
				int weightByteStride;

				int jointComponentType;

				// Position attribute is required
				assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

				const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
				const tinygltf::BufferView &posView = model.bufferViews[posAccessor.bufferView];
				bufferPos = reinterpret_cast<const float *>(&(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
				posMin = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
				posMax = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);
				vertexCount = static_cast<uint32_t>(posAccessor.count);
				posByteStride = posAccessor.ByteStride(posView) ? (posAccessor.ByteStride(posView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

				if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
				{
					const tinygltf::Accessor &normAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
					const tinygltf::BufferView &normView = model.bufferViews[normAccessor.bufferView];
					bufferNormals = reinterpret_cast<const float *>(&(model.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
					normByteStride = normAccessor.ByteStride(normView) ? (normAccessor.ByteStride(normView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
				}

				// UVs
				if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
				{
					const tinygltf::Accessor &uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
					const tinygltf::BufferView &uvView = model.bufferViews[uvAccessor.bufferView];
					bufferTexCoordSet0 = reinterpret_cast<const float *>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
					uv0ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
				}
				if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end())
				{
					const tinygltf::Accessor &uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_1")->second];
					const tinygltf::BufferView &uvView = model.bufferViews[uvAccessor.bufferView];
					bufferTexCoordSet1 = reinterpret_cast<const float *>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
					uv1ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
				}

				// Vertex colors
				if (primitive.attributes.find("COLOR_0") != primitive.attributes.end())
				{
					const tinygltf::Accessor &accessor = model.accessors[primitive.attributes.find("COLOR_0")->second];
					const tinygltf::BufferView &view = model.bufferViews[accessor.bufferView];
					bufferColorSet0 = reinterpret_cast<const float *>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					color0ByteStride = accessor.ByteStride(view) ? (accessor.ByteStride(view) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
				}

				// Skinning
				// Joints
				if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end())
				{
					const tinygltf::Accessor &jointAccessor = model.accessors[primitive.attributes.find("JOINTS_0")->second];
					const tinygltf::BufferView &jointView = model.bufferViews[jointAccessor.bufferView];
					bufferJoints = &(model.buffers[jointView.buffer].data[jointAccessor.byteOffset + jointView.byteOffset]);
					jointComponentType = jointAccessor.componentType;
					jointByteStride = jointAccessor.ByteStride(jointView) ? (jointAccessor.ByteStride(jointView) / tinygltf::GetComponentSizeInBytes(jointComponentType)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
				}

				if (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end())
				{
					const tinygltf::Accessor &weightAccessor = model.accessors[primitive.attributes.find("WEIGHTS_0")->second];
					const tinygltf::BufferView &weightView = model.bufferViews[weightAccessor.bufferView];
					bufferWeights = reinterpret_cast<const float *>(&(model.buffers[weightView.buffer].data[weightAccessor.byteOffset + weightView.byteOffset]));
					weightByteStride = weightAccessor.ByteStride(weightView) ? (weightAccessor.ByteStride(weightView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
				}

				hasSkin = (bufferJoints && bufferWeights);

				for (size_t v = 0; v < posAccessor.count; v++)
				{
					Vertex &vert = loaderInfo.vertexBuffer[loaderInfo.vertexPos];
					vert.pos = glm::vec4(glm::make_vec3(&bufferPos[v * posByteStride]), 1.0f);
					vert.normal = glm::normalize(glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * normByteStride]) : glm::vec3(0.0f)));
					vert.uv0 = bufferTexCoordSet0 ? glm::make_vec2(&bufferTexCoordSet0[v * uv0ByteStride]) : glm::vec3(0.0f);
					vert.uv1 = bufferTexCoordSet1 ? glm::make_vec2(&bufferTexCoordSet1[v * uv1ByteStride]) : glm::vec3(0.0f);
					vert.color = bufferColorSet0 ? glm::make_vec4(&bufferColorSet0[v * color0ByteStride]) : glm::vec4(1.0f);

					if (hasSkin)
					{
						switch (jointComponentType)
						{
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
						{
							const uint16_t *buf = static_cast<const uint16_t *>(bufferJoints);
							vert.joint0 = glm::vec4(glm::make_vec4(&buf[v * jointByteStride]));
							break;
						}
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
						{
							const uint8_t *buf = static_cast<const uint8_t *>(bufferJoints);
							vert.joint0 = glm::vec4(glm::make_vec4(&buf[v * jointByteStride]));
							break;
						}
						default:
							// Not supported by spec
							std::cerr << "Joint component type " << jointComponentType << " not supported!" << std::endl;
							break;
						}
					}
					else
					{
						vert.joint0 = glm::vec4(0.0f);
					}
					vert.weight0 = hasSkin ? glm::make_vec4(&bufferWeights[v * weightByteStride]) : glm::vec4(0.0f);
					// Fix for all zero weights
					if (glm::length(vert.weight0) == 0.0f)
					{
						vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
					}
					loaderInfo.vertexPos++;
				}
			}
			// Indices
			if (hasIndices)
			{
				const tinygltf::Accessor &accessor = model.accessors[primitive.indices > -1 ? primitive.indices : 0];
				const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
				const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

				indexCount = static_cast<uint32_t>(accessor.count);
				const void *dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

				switch (accessor.componentType)
				{
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
				{
					const uint32_t *buf = static_cast<const uint32_t *>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++)
					{
						loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
						loaderInfo.indexPos++;
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
				{
					const uint16_t *buf = static_cast<const uint16_t *>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++)
					{
						loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
						loaderInfo.indexPos++;
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
				{
					const uint8_t *buf = static_cast<const uint8_t *>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++)
					{
						loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
						loaderInfo.indexPos++;
					}
					break;
				}
				default:
					std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
					return;
				}
			}
			Primitive *newPrimitive = new Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? materials[primitive.material] : materials.back());
			newPrimitive->setBoundingBox(posMin, posMax);
			newMesh->primitives.push_back(newPrimitive);
		}
		// Mesh BB from BBs of primitives
		for (auto p : newMesh->primitives)
		{
			if (p->bb.valid && !newMesh->bb.valid)
			{
				newMesh->bb = p->bb;
				newMesh->bb.valid = true;
			}
			newMesh->bb.min = glm::min(newMesh->bb.min, p->bb.min);
			newMesh->bb.max = glm::max(newMesh->bb.max, p->bb.max);
		}
		newNode->mesh = newMesh;
	}
	if (parent)
	{
		parent->children.push_back(newNode);
	}
	else
	{
		nodes.push_back(newNode);
	}
	linearNodes.push_back(newNode);
}

void Model::getNodeProps(const tinygltf::Node &node, const tinygltf::Model &model, size_t &vertexCount, size_t &indexCount)
{
	if (node.children.size() > 0)
	{
		for (size_t i = 0; i < node.children.size(); i++)
		{
			getNodeProps(model.nodes[node.children[i]], model, vertexCount, indexCount);
		}
	}
	if (node.mesh > -1)
	{
		const tinygltf::Mesh mesh = model.meshes[node.mesh];
		for (size_t i = 0; i < mesh.primitives.size(); i++)
		{
			auto primitive = mesh.primitives[i];
			vertexCount += model.accessors[primitive.attributes.find("POSITION")->second].count;
			if (primitive.indices > -1)
			{
				indexCount += model.accessors[primitive.indices].count;
			}
		}
	}
}

void Model::loadSkins(tinygltf::Model &gltfModel)
{
	for (tinygltf::Skin &source : gltfModel.skins)
	{
		Skin *newSkin = new Skin{};
		newSkin->name = source.name;

		// Find skeleton root node
		if (source.skeleton > -1)
		{
			newSkin->skeletonRoot = nodeFromIndex(source.skeleton);
		}

		// Find joint nodes
		for (int jointIndex : source.joints)
		{
			Node *node = nodeFromIndex(jointIndex);
			if (node)
			{
				newSkin->joints.push_back(nodeFromIndex(jointIndex));
			}
		}

		// Get inverse bind matrices from buffer
		if (source.inverseBindMatrices > -1)
		{
			const tinygltf::Accessor &accessor = gltfModel.accessors[source.inverseBindMatrices];
			const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
			const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];
			newSkin->inverseBindMatrices.resize(accessor.count);
			memcpy(newSkin->inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(glm::mat4));
		}

		skins.push_back(newSkin);
	}
}

void Model::loadTextures(tinygltf::Model &gltfModel, vulkr::Device *device, vulkr::CommandPool *commandPool, VkQueue transferQueue)
{
	for (tinygltf::Texture &tex : gltfModel.textures)
	{
		tinygltf::Image image = gltfModel.images[tex.source];
		TextureSampler textureSampler;
		if (tex.sampler == -1)
		{
			// No sampler specified, use a default one
			textureSampler.magFilter = VK_FILTER_LINEAR;
			textureSampler.minFilter = VK_FILTER_LINEAR;
			textureSampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			textureSampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			textureSampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		}
		else
		{
			textureSampler = textureSamplers[tex.sampler];
		}
		Texture texture;
		texture.fromglTfImage(image, textureSampler, device, commandPool, transferQueue);
		textures.push_back(texture);
	}
}

VkSamplerAddressMode Model::getVkWrapMode(int32_t wrapMode)
{
	switch (wrapMode)
	{
	case -1:
	case 10497:
		return VK_SAMPLER_ADDRESS_MODE_REPEAT;
	case 33071:
		return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	case 33648:
		return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
	}

	std::cerr << "Unknown wrap mode for getVkWrapMode: " << wrapMode << std::endl;
	return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkFilter Model::getVkFilterMode(int32_t filterMode)
{
	switch (filterMode)
	{
	case -1:
	case 9728:
		return VK_FILTER_NEAREST;
	case 9729:
		return VK_FILTER_LINEAR;
	case 9984:
		return VK_FILTER_NEAREST;
	case 9985:
		return VK_FILTER_NEAREST;
	case 9986:
		return VK_FILTER_LINEAR;
	case 9987:
		return VK_FILTER_LINEAR;
	}

	std::cerr << "Unknown filter mode for getVkFilterMode: " << filterMode << std::endl;
	return VK_FILTER_NEAREST;
}

void Model::loadTextureSamplers(tinygltf::Model &gltfModel)
{
	for (tinygltf::Sampler smpl : gltfModel.samplers)
	{
		TextureSampler sampler{};
		sampler.minFilter = getVkFilterMode(smpl.minFilter);
		sampler.magFilter = getVkFilterMode(smpl.magFilter);
		sampler.addressModeU = getVkWrapMode(smpl.wrapS);
		sampler.addressModeV = getVkWrapMode(smpl.wrapT);
		sampler.addressModeW = sampler.addressModeV;
		textureSamplers.push_back(sampler);
	}
}

void Model::loadMaterials(tinygltf::Model &gltfModel)
{
	for (tinygltf::Material &mat : gltfModel.materials)
	{
		Material material{};
		material.doubleSided = mat.doubleSided;
		if (mat.values.find("baseColorTexture") != mat.values.end())
		{
			material.baseColorTexture = &textures[mat.values["baseColorTexture"].TextureIndex()];
			material.texCoordSets.baseColor = mat.values["baseColorTexture"].TextureTexCoord();
		}
		if (mat.values.find("metallicRoughnessTexture") != mat.values.end())
		{
			material.metallicRoughnessTexture = &textures[mat.values["metallicRoughnessTexture"].TextureIndex()];
			material.texCoordSets.metallicRoughness = mat.values["metallicRoughnessTexture"].TextureTexCoord();
		}
		if (mat.values.find("roughnessFactor") != mat.values.end())
		{
			material.roughnessFactor = static_cast<float>(mat.values["roughnessFactor"].Factor());
		}
		if (mat.values.find("metallicFactor") != mat.values.end())
		{
			material.metallicFactor = static_cast<float>(mat.values["metallicFactor"].Factor());
		}
		if (mat.values.find("baseColorFactor") != mat.values.end())
		{
			material.baseColorFactor = glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
		}
		if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end())
		{
			material.normalTexture = &textures[mat.additionalValues["normalTexture"].TextureIndex()];
			material.texCoordSets.normal = mat.additionalValues["normalTexture"].TextureTexCoord();
		}
		if (mat.additionalValues.find("emissiveTexture") != mat.additionalValues.end())
		{
			material.emissiveTexture = &textures[mat.additionalValues["emissiveTexture"].TextureIndex()];
			material.texCoordSets.emissive = mat.additionalValues["emissiveTexture"].TextureTexCoord();
		}
		if (mat.additionalValues.find("occlusionTexture") != mat.additionalValues.end())
		{
			material.occlusionTexture = &textures[mat.additionalValues["occlusionTexture"].TextureIndex()];
			material.texCoordSets.occlusion = mat.additionalValues["occlusionTexture"].TextureTexCoord();
		}
		if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end())
		{
			tinygltf::Parameter param = mat.additionalValues["alphaMode"];
			if (param.string_value == "BLEND")
			{
				material.alphaMode = Material::ALPHAMODE_BLEND;
			}
			if (param.string_value == "MASK")
			{
				material.alphaCutoff = 0.5f;
				material.alphaMode = Material::ALPHAMODE_MASK;
			}
		}
		if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end())
		{
			material.alphaCutoff = static_cast<float>(mat.additionalValues["alphaCutoff"].Factor());
		}
		if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end())
		{
			material.emissiveFactor = glm::vec4(glm::make_vec3(mat.additionalValues["emissiveFactor"].ColorFactor().data()), 1.0);
		}

		// Extensions
		// @TODO: Find out if there is a nicer way of reading these properties with recent tinygltf headers
		if (mat.extensions.find("KHR_materials_pbrSpecularGlossiness") != mat.extensions.end())
		{
			auto ext = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
			if (ext->second.Has("specularGlossinessTexture"))
			{
				auto index = ext->second.Get("specularGlossinessTexture").Get("index");
				material.extension.specularGlossinessTexture = &textures[index.Get<int>()];
				auto texCoordSet = ext->second.Get("specularGlossinessTexture").Get("texCoord");
				material.texCoordSets.specularGlossiness = texCoordSet.Get<int>();
				material.pbrWorkflows.specularGlossiness = true;
			}
			if (ext->second.Has("diffuseTexture"))
			{
				auto index = ext->second.Get("diffuseTexture").Get("index");
				material.extension.diffuseTexture = &textures[index.Get<int>()];
			}
			if (ext->second.Has("diffuseFactor"))
			{
				auto factor = ext->second.Get("diffuseFactor");
				for (uint32_t i = 0; i < factor.ArrayLen(); i++)
				{
					auto val = factor.Get(i);
					material.extension.diffuseFactor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
				}
			}
			if (ext->second.Has("specularFactor"))
			{
				auto factor = ext->second.Get("specularFactor");
				for (uint32_t i = 0; i < factor.ArrayLen(); i++)
				{
					auto val = factor.Get(i);
					material.extension.specularFactor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
				}
			}
		}

		if (mat.extensions.find("KHR_materials_unlit") != mat.extensions.end())
		{
			material.unlit = true;
		}

		if (mat.extensions.find("KHR_materials_emissive_strength") != mat.extensions.end())
		{
			auto ext = mat.extensions.find("KHR_materials_emissive_strength");
			if (ext->second.Has("emissiveStrength"))
			{
				auto value = ext->second.Get("emissiveStrength");
				material.emissiveStrength = (float)value.Get<double>();
			}
		}

		material.index = static_cast<uint32_t>(materials.size());
		materials.push_back(material);
	}
	// Push a default material at the end of the list for meshes with no material assigned
	materials.push_back(Material());
}

void Model::loadAnimations(tinygltf::Model &gltfModel)
{
	for (tinygltf::Animation &anim : gltfModel.animations)
	{
		Animation animation{};
		animation.name = anim.name;
		if (anim.name.empty())
		{
			animation.name = std::to_string(animations.size());
		}

		// Samplers
		for (auto &samp : anim.samplers)
		{
			AnimationSampler sampler{};

			if (samp.interpolation == "LINEAR")
			{
				sampler.interpolation = AnimationSampler::InterpolationType::LINEAR;
			}
			if (samp.interpolation == "STEP")
			{
				sampler.interpolation = AnimationSampler::InterpolationType::STEP;
			}
			if (samp.interpolation == "CUBICSPLINE")
			{
				sampler.interpolation = AnimationSampler::InterpolationType::CUBICSPLINE;
			}

			// Read sampler input time values
			{
				const tinygltf::Accessor &accessor = gltfModel.accessors[samp.input];
				const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
				const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

				assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

				const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
				const float *buf = static_cast<const float *>(dataPtr);
				for (size_t index = 0; index < accessor.count; index++)
				{
					sampler.inputs.push_back(buf[index]);
				}

				for (auto input : sampler.inputs)
				{
					if (input < animation.start)
					{
						animation.start = input;
					};
					if (input > animation.end)
					{
						animation.end = input;
					}
				}
			}

			// Read sampler output T/R/S values 
			{
				const tinygltf::Accessor &accessor = gltfModel.accessors[samp.output];
				const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
				const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

				assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

				const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

				switch (accessor.type)
				{
				case TINYGLTF_TYPE_VEC3:
				{
					const glm::vec3 *buf = static_cast<const glm::vec3 *>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++)
					{
						sampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
					}
					break;
				}
				case TINYGLTF_TYPE_VEC4:
				{
					const glm::vec4 *buf = static_cast<const glm::vec4 *>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++)
					{
						sampler.outputsVec4.push_back(buf[index]);
					}
					break;
				}
				default:
				{
					std::cout << "unknown type" << std::endl;
					break;
				}
				}
			}

			animation.samplers.push_back(sampler);
		}

		// Channels
		for (auto &source : anim.channels)
		{
			AnimationChannel channel{};

			if (source.target_path == "rotation")
			{
				channel.path = AnimationChannel::PathType::ROTATION;
			}
			if (source.target_path == "translation")
			{
				channel.path = AnimationChannel::PathType::TRANSLATION;
			}
			if (source.target_path == "scale")
			{
				channel.path = AnimationChannel::PathType::SCALE;
			}
			if (source.target_path == "weights")
			{
				std::cout << "weights not yet supported, skipping channel" << std::endl;
				continue;
			}
			channel.samplerIndex = source.sampler;
			channel.node = nodeFromIndex(source.target_node);
			if (!channel.node)
			{
				continue;
			}

			animation.channels.push_back(channel);
		}

		animations.push_back(animation);
	}
}

void Model::loadFromFile(std::string filename, vulkr::Device *device, vulkr::CommandPool *commandPool, VkQueue transferQueue, float scale)
{
	tinygltf::Model gltfModel;
	tinygltf::TinyGLTF gltfContext;

	std::string error;
	std::string warning;

	this->device = device;

	bool binary = false;
	size_t extpos = filename.rfind('.', filename.length());
	if (extpos != std::string::npos)
	{
		binary = (filename.substr(extpos + 1, filename.length() - extpos) == "glb");
	}

	bool fileLoaded = binary ? gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.c_str()) : gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.c_str());

	LoaderInfo loaderInfo{};
	size_t vertexCount = 0;
	size_t indexCount = 0;

	if (fileLoaded)
	{
		loadTextureSamplers(gltfModel);
		loadTextures(gltfModel, device, commandPool, transferQueue);
		loadMaterials(gltfModel);

		const tinygltf::Scene &scene = gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0];

		// Get vertex and index buffer sizes up-front
		for (size_t i = 0; i < scene.nodes.size(); i++)
		{
			getNodeProps(gltfModel.nodes[scene.nodes[i]], gltfModel, vertexCount, indexCount);
		}
		loaderInfo.vertexBuffer = new Vertex[vertexCount];
		loaderInfo.indexBuffer = new uint32_t[indexCount];

		// TODO: scene handling with no default scene
		for (size_t i = 0; i < scene.nodes.size(); i++)
		{
			const tinygltf::Node node = gltfModel.nodes[scene.nodes[i]];
			loadNode(nullptr, node, scene.nodes[i], gltfModel, loaderInfo, scale);
		}
		if (gltfModel.animations.size() > 0)
		{
			loadAnimations(gltfModel);
		}
		loadSkins(gltfModel);

		for (auto node : linearNodes)
		{
			// Assign skins
			if (node->skinIndex > -1)
			{
				node->skin = skins[node->skinIndex];
			}
			// Initial pose
			if (node->mesh)
			{
				node->update();
			}
		}
	}
	else
	{
		LOGE("Could not load gltf file: {}", error);
		std::abort();;
	}

	extensions = gltfModel.extensionsUsed;

	size_t vertexBufferSize = vertexCount * sizeof(Vertex);
	size_t indexBufferSize = indexCount * sizeof(uint32_t);

	assert(vertexBufferSize > 0);

	VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferInfo.size = vertexBufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo memoryInfo{};
	memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	// Create staging buffers
	std::unique_ptr<vulkr::Buffer> vertexStagingBuffer;
	std::unique_ptr<vulkr::Buffer> indexStagingBuffer;

	{
		VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bufferInfo.size = vertexBufferSize;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo memoryInfo{};
		memoryInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

		vertexStagingBuffer = std::make_unique<vulkr::Buffer>(*device, bufferInfo, memoryInfo);
		vertexStagingBuffer->update(loaderInfo.vertexBuffer, vertexBufferSize);


		// Index data
		if (indexBufferSize > 0)
		{
			bufferInfo.size = indexBufferSize;
			// all other bufferInfo variables are the same as the ones for the vertex buffer creation
			indexStagingBuffer = std::make_unique<vulkr::Buffer>(*device, bufferInfo, memoryInfo);
			indexStagingBuffer->update(loaderInfo.indexBuffer, indexBufferSize);
		}
	}

	// Create device local buffers
	{
		VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bufferInfo.size = vertexBufferSize;
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo memoryInfo{};
		memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		vertexBuffer = std::make_unique<vulkr::Buffer>(*device, bufferInfo, memoryInfo);

		// Index data
		if (indexBufferSize > 0)
		{
			bufferInfo.size = indexBufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			// all other bufferInfo variables are the same as the ones for the vertex buffer creation
			indexBuffer = std::make_unique<vulkr::Buffer>(*device, bufferInfo, memoryInfo);
		}
	}

	// Copy from staging buffers
	std::unique_ptr<CommandBuffer> copyCommandBuffer = std::make_unique<CommandBuffer>(*commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	copyCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr);

	VkBufferCopy copyRegion = {};

	copyRegion.size = vertexBufferSize;
	vkCmdCopyBuffer(copyCommandBuffer->getHandle(), vertexStagingBuffer->getHandle(), vertexBuffer->getHandle(), 1, &copyRegion);

	if (indexBufferSize > 0)
	{
		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(copyCommandBuffer->getHandle(), indexStagingBuffer->getHandle(), indexBuffer->getHandle(), 1, &copyRegion);
	}

	flushCommandBuffer(device, copyCommandBuffer.get(), transferQueue);
	copyCommandBuffer.reset();

	vertexStagingBuffer.reset();
	if (indexBufferSize > 0)
	{
		indexStagingBuffer.reset();
	}

	delete[] loaderInfo.vertexBuffer;
	delete[] loaderInfo.indexBuffer;

	getSceneDimensions();
}

void Model::drawNode(Node *node, VkCommandBuffer commandBuffer)
{
	if (node->mesh)
	{
		for (Primitive *primitive : node->mesh->primitives)
		{
			vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
		}
	}
	for (auto &child : node->children)
	{
		drawNode(child, commandBuffer);
	}
}

void Model::draw(VkCommandBuffer commandBuffer)
{
	const VkDeviceSize offsets[1] = { 0 };
	VkBuffer vertexBuffers[] = { vertexBuffer->getHandle() };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getHandle(), 0, VK_INDEX_TYPE_UINT32);
	for (auto &node : nodes)
	{
		drawNode(node, commandBuffer);
	}
}

void Model::calculateBoundingBox(Node *node, Node *parent)
{
	if (node->mesh)
	{
		if (node->mesh->bb.valid)
		{
			node->aabb = node->mesh->bb.getAABB(node->getMatrix());
			if (node->children.size() == 0)
			{
				node->bvh.min = node->aabb.min;
				node->bvh.max = node->aabb.max;
				node->bvh.valid = true;
			}
		}
	}

	if (parent != nullptr)
	{
		parent->bvh.min = glm::min(parent->bvh.min, node->bvh.min);
		parent->bvh.max = glm::min(parent->bvh.max, node->bvh.max);
	}

	for (auto &child : node->children)
	{
		calculateBoundingBox(child, node);
	}
}

void Model::getSceneDimensions()
{
	// Calculate binary volume hierarchy for all nodes in the scene
	for (auto node : linearNodes)
	{
		calculateBoundingBox(node, nullptr);
	}

	dimensions.min = glm::vec3(FLT_MAX);
	dimensions.max = glm::vec3(-FLT_MAX);

	for (auto node : linearNodes)
	{
		if (node->bvh.valid)
		{
			dimensions.min = glm::min(dimensions.min, node->bvh.min);
			dimensions.max = glm::max(dimensions.max, node->bvh.max);
		}
	}

	// Calculate scene aabb
	aabb = glm::scale(glm::mat4(1.0f), glm::vec3(dimensions.max[0] - dimensions.min[0], dimensions.max[1] - dimensions.min[1], dimensions.max[2] - dimensions.min[2]));
	aabb[3][0] = dimensions.min[0];
	aabb[3][1] = dimensions.min[1];
	aabb[3][2] = dimensions.min[2];
}

void Model::updateAnimation(uint32_t index, float time)
{
	if (animations.empty())
	{
		std::cout << ".glTF does not contain animation." << std::endl;
		return;
	}
	if (index > static_cast<uint32_t>(animations.size()) - 1)
	{
		std::cout << "No animation with index " << index << std::endl;
		return;
	}
	Animation &animation = animations[index];

	bool updated = false;
	for (auto &channel : animation.channels)
	{
		AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
		if (sampler.inputs.size() > sampler.outputsVec4.size())
		{
			continue;
		}

		for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
		{
			if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1]))
			{
				float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
				if (u <= 1.0f)
				{
					switch (channel.path)
					{
					case AnimationChannel::PathType::TRANSLATION:
					{
						glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
						channel.node->translation = glm::vec3(trans);
						break;
					}
					case AnimationChannel::PathType::SCALE:
					{
						glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
						channel.node->scale = glm::vec3(trans);
						break;
					}
					case AnimationChannel::PathType::ROTATION:
					{
						glm::quat q1;
						q1.x = sampler.outputsVec4[i].x;
						q1.y = sampler.outputsVec4[i].y;
						q1.z = sampler.outputsVec4[i].z;
						q1.w = sampler.outputsVec4[i].w;
						glm::quat q2;
						q2.x = sampler.outputsVec4[i + 1].x;
						q2.y = sampler.outputsVec4[i + 1].y;
						q2.z = sampler.outputsVec4[i + 1].z;
						q2.w = sampler.outputsVec4[i + 1].w;
						channel.node->rotation = glm::normalize(glm::slerp(q1, q2, u));
						break;
					}
					}
					updated = true;
				}
			}
		}
	}
	if (updated)
	{
		for (auto &node : nodes)
		{
			node->update();
		}
	}
}

Node *Model::findNode(Node *parent, uint32_t index)
{
	Node *nodeFound = nullptr;
	if (parent->index == index)
	{
		return parent;
	}
	for (auto &child : parent->children)
	{
		nodeFound = findNode(child, index);
		if (nodeFound)
		{
			break;
		}
	}
	return nodeFound;
}

Node *Model::nodeFromIndex(uint32_t index)
{
	Node *nodeFound = nullptr;
	for (auto &node : nodes)
	{
		nodeFound = findNode(node, index);
		if (nodeFound)
		{
			break;
		}
	}
	return nodeFound;
}

}