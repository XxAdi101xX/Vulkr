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
#include "device.h"

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
	VkFormat format,
	VkExtent3D extent,
	VkImageUsageFlags imageUsage,
	VmaMemoryUsage memoryUsage,
	const uint32_t mipLevels,
	const uint32_t arrayLayers,
	VkSampleCountFlagBits sampleCount,
	VkImageTiling tiling,
	VkSharingMode sharingMode,
	VkImageLayout initialLayout,
	VkImageCreateFlags flags) :
	device{ device },
	type{ getImageType(extent) },
	format{ format },
	extent{ extent },
	usageFlags{ imageUsage },
	arrayLayerCount{ arrayLayers },
	sampleCount{ sampleCount },
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

	VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
	imageInfo.flags = flags;
	imageInfo.imageType = type;
	imageInfo.format = format;
	imageInfo.extent = extent;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = arrayLayers;
	imageInfo.samples = sampleCount;
	imageInfo.tiling = tiling;
	imageInfo.usage = imageUsage;
	imageInfo.initialLayout = initialLayout;

	VmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = memoryUsage;
	if (imageUsage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
	{
		allocationInfo.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
	}

	VK_CHECK(vmaCreateImage(device.getMemoryAllocator(), &imageInfo, &allocationInfo, &handle, &allocation, nullptr));
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

Image::~Image()
{
	if (handle != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
	{
		if (mapped)
		{
			LOGW("Mapped data has not be explicity unmapped, will be unmapped in destructor..");
			unmap();
		}
		vmaDestroyImage(device.getMemoryAllocator(), handle, allocation);
	}
}

void *Image::map()
{
	if (!mapped)
	{
		if (tiling != VK_IMAGE_TILING_LINEAR)
		{
			LOGW("Mapping image memory that is not linear");
		}
		VK_CHECK(vmaMapMemory(device.getMemoryAllocator(), allocation, reinterpret_cast<void**>(&mappedData)));
		mapped = true;
	}
	return mappedData;
}

void Image::unmap()
{
	if (!mapped)
	{
		LOGEANDABORT("Trying to unmap memory on a buffer that's not mapped");
	}
	vmaUnmapMemory(device.getMemoryAllocator(), allocation);
	mappedData = nullptr;
	mapped = false;

}

Device &Image::getDevice() const
{
	return device;
}

VkImage Image::getHandle() const
{
	return handle;
}

VmaAllocation Image::getAllocation() const
{
	return allocation;
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