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

namespace vulkr
{

/* Forward declaration */
class Device;

class Image
{
	// TODO: consider how to do memeory management!
public:
	Image(
		Device& device,
		const VkExtent3D& extent,
		VkFormat format,
		VkImageUsageFlags imageUsage,
		VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT,
		uint32_t mipLevels = 1,
		uint32_t arrayLayers = 1,
		VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL,
		VkImageCreateFlags flags = 0
	);

	/* Disable unnecessary operators to prevent error prone usages */
	Image(Image &&) = delete; // TODO: do we need this?
	Image(const Image &) = delete;
	Image& operator=(const Image &) = delete;
	Image& operator=(Image &&) = delete;

	VkImage getHandle() const;

	Device &getDevice() const;

	VkImageType getType() const;

	const VkExtent3D& getExtent() const;

	VkFormat getFormat() const;

	VkSampleCountFlagBits getSampleCount() const;

	VkImageUsageFlags getUsage() const;

	VkImageTiling getTiling() const;

	VkImageSubresource getSubresource() const;

	uint32_t getArrayLayerCount() const;

private:
	Device& device;

	VkImage handle{ VK_NULL_HANDLE };

	//VmaAllocation memory{ VK_NULL_HANDLE };

	VkImageType type{};

	VkExtent3D extent{};

	VkFormat format{};

	VkImageUsageFlags usage{};

	VkSampleCountFlagBits sampleCount{};

	VkImageTiling tiling{};

	VkImageSubresource subresource{};

	uint32_t arrayLayerCount{ 0u };
};

} // namespace vulkr