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
#include "command_buffer.h"

namespace vulkr
{

class Device;

class Image
{
public:
	Image(
		Device &device,
		VkFormat format,
		VkExtent3D extent,
		VkImageUsageFlags imageUsage,
		VmaMemoryUsage memoryUsage,
		uint32_t mipLevels = 1,
		uint32_t arrayLayers = 1,
		VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT,
		VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL,
		VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		VkImageCreateFlags flags = 0,
		uint32_t queueFamilyIndexCount = 0,
		const uint32_t *pQueueFamilyIndices = nullptr
	);

	Image(
		Device &device,
		VkImage handle,
		VkExtent3D extent,
		VkFormat format,
		VkImageUsageFlags imageUsageFlags,
		VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT
	);
	~Image();
 
	Image(Image &&) = delete;
	Image(const Image &) = delete;
	Image &operator=(const Image &) = delete;
	Image &operator=(Image &&) = delete;

	void *map();

	void unmap();

	/* Transition the image from the old layout to the new layout; the commandBuffer must have been started and ready to record commands */
	void transitionImageLayout(CommandBuffer &commandBuffer, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange);

	/* Getters */
	VkImage getHandle() const;

	VmaAllocation getAllocation() const;

	Device &getDevice() const;

	VkImageType getType() const;

	const VkExtent3D &getExtent() const;

	VkFormat getFormat() const;

	VkSampleCountFlagBits getSampleCount() const;

	VkImageUsageFlags getUsageFlags() const;

	VkImageTiling getTiling() const;

	VkImageSubresource getSubresource() const;

	uint32_t getArrayLayerCount() const;

	/* TODO: this value might be incorrect as we've not updated layouts when they're updated by the renderpass initalLayout, finalLayout and the per-subpass transitions as described in the AttachmentRefs*/
	VkImageLayout getLayout() const;

private:
	Device &device;

	VkImage handle{ VK_NULL_HANDLE };

	VmaAllocation allocation{ VK_NULL_HANDLE };

	// Whether the buffer has been mapped with vmaMapMemory
	bool mapped{ false };
	void *mappedData{ nullptr };

	VkImageType type{};

	VkExtent3D extent{};

	VkFormat format{};

	VkImageUsageFlags usageFlags{};

	VkSampleCountFlagBits sampleCount{};

	VkImageTiling tiling{};

	VkImageSubresource subresource{};

	uint32_t arrayLayerCount{ 0u };

	VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
};

} // namespace vulkr