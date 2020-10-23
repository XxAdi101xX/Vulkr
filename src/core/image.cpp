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

#include "image.h"

#include "common/logger.h"

namespace vulkr
{

namespace
{
VkImageType getImageType(VkExtent3D extent)
{
	VkImageType result{};

	uint32_t dimensionCount{ 0 };

	if (extent.width >= 1)
	{
		dimensionCount++;
	}

	if (extent.height >= 1)
	{
		dimensionCount++;
	}

	if (extent.depth > 1)
	{
		dimensionCount++;
	}

	switch (dimensionCount)
	{
	case 1:
		result = VK_IMAGE_TYPE_1D;
		break;
	case 2:
		result = VK_IMAGE_TYPE_2D;
		break;
	case 3:
		result = VK_IMAGE_TYPE_3D;
		break;
	default:
		LOGEANDABORT("Invalid dimension count calculated");
	}

	return result;
}
} // namespace

Image::Image(
	Device &device,
	VkExtent3D extent,
	VkFormat format,
	VkImageUsageFlags imageUsage,
	VkSampleCountFlagBits sampleCount,
	const uint32_t mipLevels,
	const uint32_t arrayLayers,
	VkImageTiling tiling,
	VkImageCreateFlags flags) :
	device{ device },
	type{ getImageType(extent) },
	extent{ extent },
	format{ format },
	sampleCount{ sampleCount },
	usageFlags{ imageUsage },
	arrayLayerCount{ arrayLayers },
	tiling{ tiling }
{
	if (mipLevels < 1)
	{
		LOGEANDABORT("Device::Device: mipLevels of {} is invalid", mipLevels);
	}

	if (arrayLayers < 1)
	{
		LOGEANDABORT("Device::Device: arrayLayers of {} is invalid", arrayLayers);
	}

	subresource.mipLevel = mipLevels;
	subresource.arrayLayer = arrayLayers;

	VkImageCreateInfo imageCreateInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
	imageCreateInfo.flags = flags;
	imageCreateInfo.imageType = type;
	imageCreateInfo.format = format;
	imageCreateInfo.extent = extent;
	imageCreateInfo.mipLevels = mipLevels;
	imageCreateInfo.arrayLayers = arrayLayers;
	imageCreateInfo.samples = sampleCount;
	imageCreateInfo.tiling = tiling;
	imageCreateInfo.usage = imageUsage;

	//if (imageUsage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
	//{
	//	memory_info.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
	//}

	//auto result = vmaCreateImage(device.get_memory_allocator(),
	//	&image_info, &memory_info,
	//	&handle, &memory,
	//	nullptr);

	//if (result != VK_SUCCESS)
	//{
	//	throw VulkanException{ result, "Cannot create Image" };
	//}
}

Image::Image(Device &device, VkImage handle, VkExtent3D extent, VkFormat format, VkImageUsageFlags imageUsageFlags, VkSampleCountFlagBits sampleCount) :
	device{ device },
	handle{ handle },
	type{ getImageType(extent) },
	extent{ extent },
	format{ format },
	sampleCount{ sampleCount },
	usageFlags{ imageUsageFlags }
{
	subresource.mipLevel = 1;
	subresource.arrayLayer = 1;
}

Device &Image::getDevice() const
{
	return device;
}

VkImage Image::getHandle() const
{
	return handle;
}

VkImageType Image::getType() const
{
	return type;
}

const VkExtent3D& Image::getExtent() const
{
	return extent;
}

VkFormat Image::getFormat() const
{
	return format;
}

VkSampleCountFlagBits Image::getSampleCount() const
{
	return sampleCount;
}

VkImageUsageFlags Image::getUsageFlags() const
{
	return usageFlags;
}

VkImageTiling Image::getTiling() const
{
	return tiling;
}

VkImageSubresource Image::getSubresource() const
{
	return subresource;
}

uint32_t Image::getArrayLayerCount() const
{
	return arrayLayerCount;
}

} // namespace vulkr