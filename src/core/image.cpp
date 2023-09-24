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
	VkImageCreateFlags flags,
	uint32_t queueFamilyIndexCount,
	const uint32_t *pQueueFamilyIndices
) :
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
	imageInfo.sharingMode = sharingMode;
	imageInfo.queueFamilyIndexCount = queueFamilyIndexCount;
	imageInfo.pQueueFamilyIndices = pQueueFamilyIndices;
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
		// Equivilant to vkDestroyImage(device, image, allocationCallbacks) and vmaFreeMemory(allocator, allocation)
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

VkImageMemoryBarrier2 Image::transitionImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange)
{
	LOGEANDABORT("This method should not be used as it's probably outdated, leaving it here for reference");
	VkImageMemoryBarrier2 imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	imageMemoryBarrier.pNext = nullptr;
	imageMemoryBarrier.oldLayout = oldLayout;
	imageMemoryBarrier.newLayout = newLayout;
	imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageMemoryBarrier.image = handle;
	imageMemoryBarrier.subresourceRange = subresourceRange;


	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
	{
		imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
		imageMemoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	}
	else
	{
		LOGEANDABORT("Unsupported layout transition! Reference the pipelineStageForLayout and accessFlagsForImageLayout methods for reference");
	}

	// Source layouts (old)
	// Source access mask controls actions that have to be finished on the old layout
	// before it will be transitioned to the new layout
	switch (oldLayout)
	{
	case VK_IMAGE_LAYOUT_UNDEFINED:
	case VK_IMAGE_LAYOUT_GENERAL:
		// Image layout is undefined or general
		// Only valid as initial layout
		// No flags required, listed only for completeness
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
		break;

	case VK_IMAGE_LAYOUT_PREINITIALIZED:
		// Image is preinitialized
		// Only valid as initial layout for linear images, preserves memory contents
		// Make sure host writes have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
		// Image is a color attachment
		// Make sure any writes to the color buffer have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
		// Image is a depth/stencil attachment
		// Make sure any writes to the depth/stencil buffer have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
		// Image is a transfer source
		// Make sure any reads from the image have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
		break;

	case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
		// Image is a transfer destination
		// Make sure any writes to the image have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
		// Image is read by a shader
		// Make sure any shader reads from the image have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
		break;
	default:
		LOGEANDABORT("Unhandled oldImageLayout encountered");
		break;
	}

	// Target layouts (new)
	// Destination access mask controls the dependency for the new image layout
	switch (newLayout)
	{
	case VK_IMAGE_LAYOUT_UNDEFINED:
		LOGEANDABORT("VK_IMAGE_LAYOUT_UNDEFINED is not valid as a newLayout");
		break;
	case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: // Do nothing?
	case VK_IMAGE_LAYOUT_GENERAL:
		// Image will be general so there isn't a specific destination mask
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		break;
	case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
		// Image will be used as a transfer destination
		// Make sure any writes to the image have been finished
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
		// Image will be used as a transfer source
		// Make sure any reads from the image have been finished
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
		break;

	case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
		// Image will be used as a color attachment
		// Make sure any writes to the color buffer have been finished
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
		// Image layout will be used as a depth/stencil attachment
		// Make sure any writes to depth/stencil buffer have been finished
		imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
		// Image will be read in a shader (sampler, input attachment)
		// Make sure any writes to the image have been finished
		if (imageMemoryBarrier.srcAccessMask == VK_ACCESS_2_NONE)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
		}
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
		break;
	default:
		LOGEANDABORT("Unhandled newImageLayout encountered");
		break;
	}

	return imageMemoryBarrier;
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