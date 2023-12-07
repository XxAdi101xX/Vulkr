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

#include "swapchain.h"
#include "physical_device.h"
#include "device.h"
#include "queue.h"
#include "image.h"

#include "platform/window.h"

#include "common/logger.h"
#include "common/strings.h"

namespace vulkr
{

Swapchain::Swapchain(
	Device &device,
	VkSurfaceKHR surface,
	const VkSurfaceTransformFlagBitsKHR transform,
	const VkPresentModeKHR presentMode,
	const std::set<VkImageUsageFlagBits> &imageUsageFlags,
	uint32_t graphicsQueueFamilyIndex,
	uint32_t presentQueueFamilyIndex) :
	device{ device }, surface{ surface }
{
	uint32_t surfaceFormatCount{ 0u };
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(this->device.getPhysicalDevice().getHandle(), surface, &surfaceFormatCount, nullptr));
	availableSurfaceFormats.resize(surfaceFormatCount);
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(this->device.getPhysicalDevice().getHandle(), surface, &surfaceFormatCount, availableSurfaceFormats.data()));

	LOGI("The following surface formats are available:");
	for (auto &surfaceFormat : availableSurfaceFormats)
	{
		LOGI("  \t{}", to_string(surfaceFormat));
	}

	uint32_t presentModeCount{ 0u };
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(this->device.getPhysicalDevice().getHandle(), surface, &presentModeCount, nullptr));
	availablePresentModes.resize(presentModeCount);
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(this->device.getPhysicalDevice().getHandle(), surface, &presentModeCount, availablePresentModes.data()));

	LOGI("Surface supports the following present modes:");
	for (auto &presentMode : availablePresentModes)
	{
		LOGI("  \t{}", to_string(presentMode));
	}

	VkSurfaceCapabilitiesKHR surfaceCapabilities{};
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(this->device.getPhysicalDevice().getHandle(), surface, &surfaceCapabilities);

	// Chose best properties based on surface capabilities
	properties.imageCount = chooseImageCount(surfaceCapabilities.minImageCount, surfaceCapabilities.maxImageCount);
	properties.surfaceFormat = chooseSurfaceFormat(availableSurfaceFormats);
	properties.imageExtent = chooseImageExtent(surfaceCapabilities.currentExtent, surfaceCapabilities.minImageExtent, surfaceCapabilities.maxImageExtent);
	properties.imageArraylayers = chooseImageArrayLayers(1u, surfaceCapabilities.maxImageArrayLayers);
	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(this->device.getPhysicalDevice().getHandle(), properties.surfaceFormat.format, &formatProperties);
	properties.imageUsage = chooseImageUsage(imageUsageFlags, surfaceCapabilities.supportedUsageFlags, formatProperties.optimalTilingFeatures);
	properties.preTransform = choosePreTransform(transform, surfaceCapabilities.supportedTransforms, surfaceCapabilities.currentTransform);
	properties.compositeAlpha = chooseCompositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, surfaceCapabilities.supportedCompositeAlpha);
	properties.presentMode = choosePresentMode(presentMode);
	properties.clipped = VK_TRUE;

	create(graphicsQueueFamilyIndex, presentQueueFamilyIndex);
}

Swapchain::~Swapchain()
{
	if (handle != VK_NULL_HANDLE)
	{
		vkDestroySwapchainKHR(device.getHandle(), handle, nullptr);
	}
}

VkSwapchainKHR Swapchain::getHandle() const
{
	return handle;
}

const SwapchainProperties &Swapchain::getProperties() const
{
	return properties;
}

const std::vector<std::unique_ptr<Image>> &Swapchain::getImages() const
{
	return images;
}

void Swapchain::create(uint32_t graphicsQueueFamilyIndex, uint32_t presentQueueFamilyIndex)
{
	VkSwapchainCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
	createInfo.surface = surface;
	createInfo.minImageCount = properties.imageCount;
	createInfo.imageFormat = properties.surfaceFormat.format;
	createInfo.imageColorSpace = properties.surfaceFormat.colorSpace;
	createInfo.imageExtent = properties.imageExtent;
	createInfo.imageArrayLayers = properties.imageArraylayers;
	createInfo.imageUsage = properties.imageUsage;

	// Check whether our graphics queue can support present and populate info accordingly
	uint32_t queueFamilyIndices[] = { graphicsQueueFamilyIndex, presentQueueFamilyIndex };

	if (graphicsQueueFamilyIndex != presentQueueFamilyIndex)
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2u;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	createInfo.preTransform = properties.preTransform;
	createInfo.compositeAlpha = properties.compositeAlpha;
	createInfo.presentMode = properties.presentMode;
	createInfo.clipped = properties.clipped;
	createInfo.oldSwapchain = properties.oldSwapchain;

	VK_CHECK(vkCreateSwapchainKHR(device.getHandle(), &createInfo, nullptr, &handle));

	// Get the swapchain images after it's been created
	std::vector<VkImage> imageHandles;
	uint32_t imageCount{ 0u };
	VK_CHECK(vkGetSwapchainImagesKHR(device.getHandle(), handle, &imageCount, nullptr));
	imageHandles.resize(imageCount);
	VK_CHECK(vkGetSwapchainImagesKHR(device.getHandle(), handle, &imageCount, imageHandles.data()));

	images.reserve(imageCount);
	for (uint32_t i = 0; i < imageCount; ++i)
	{
		VkExtent3D extent{ properties.imageExtent.width, properties.imageExtent.height, 1 };
		images.emplace_back(std::make_unique<Image>(this->device, imageHandles[i], extent, properties.surfaceFormat.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT));
	}
}

uint32_t Swapchain::chooseImageCount(uint32_t minImageCount, uint32_t maxImageCount) const
{
	// We default to 1 more than the minimum as per the suggestions in the vulkan-tutorial
	uint32_t imageCount = minImageCount + 1;
	if (maxImageCount > 0 && imageCount > maxImageCount)
	{
		LOGI("An image count of {} was chosen", maxImageCount);
		return maxImageCount;
	}

	LOGI("An image count of {} was chosen", imageCount);
	return imageCount;
}

VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) const
{
	if (availableFormats.empty())
	{
		LOGEANDABORT("No surface formats were found.");
	}

	for (const auto &prioritySurfaceFormat : surfaceFormatPriorityList)
	{
		for (const auto &availableFormat : availableFormats)
		{
			if (availableFormat.format == prioritySurfaceFormat.format && availableFormat.colorSpace == prioritySurfaceFormat.colorSpace)
			{
				return availableFormat;
			}
		}
	}

	LOGW("A surface format from the priority list was not found hence the first available format was chosen. This might be problematic if the format does not support storage images");
	return availableFormats[0];
}

VkExtent2D Swapchain::chooseImageExtent(VkExtent2D currentExtent, VkExtent2D minImageExtent, VkExtent2D maxImageExtent) const
{
	if (currentExtent.width != UINT32_MAX)
	{
		return currentExtent;
	}

	VkExtent2D extent = Window::getWindowExtent();
	extent.width = std::max(minImageExtent.width, std::min(maxImageExtent.width, extent.width));
	extent.height = std::max(minImageExtent.height, std::min(maxImageExtent.height, extent.height));

	return extent;
}

uint32_t Swapchain::chooseImageArrayLayers(
	uint32_t requestedImageArrayLayers,
	uint32_t maxImageArrayLayers) const
{
	requestedImageArrayLayers = std::min(requestedImageArrayLayers, maxImageArrayLayers);
	requestedImageArrayLayers = std::max(requestedImageArrayLayers, 1u);

	return requestedImageArrayLayers;
}

bool Swapchain::validateFormatFeature(VkImageUsageFlagBits imageUsage, VkFormatFeatureFlags supportedFormatFeatures) const
{
	switch (imageUsage)
	{
	case VK_IMAGE_USAGE_STORAGE_BIT:
		return VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT & supportedFormatFeatures;
	default:
		return true;
	}
}

VkImageUsageFlags Swapchain::chooseImageUsage(
	const std::set<VkImageUsageFlagBits> &requestedImageUsageFlags,
	VkImageUsageFlags supportedImageUsage,
	VkFormatFeatureFlags supportedFormatFeatures) const
{
	std::set<VkImageUsageFlagBits> imageUsageFlagSet;
	for (auto flag : requestedImageUsageFlags)
	{
		if ((flag & supportedImageUsage) && validateFormatFeature(flag, supportedFormatFeatures))
		{
			imageUsageFlagSet.insert(flag);
		}
		else
		{
			LOGW("(Swapchain) Image usage ({}) requested but not supported.", to_string(flag));
		}
	}

	if (imageUsageFlagSet.empty())
	{
		LOGW("None of the requested image usage flags were available, will try to default to an available flag.");
		// Pick the first format from list of defaults, if supported
		for (VkImageUsageFlagBits imageUsageFlag : imageUsagePriorityFlags)
		{
			if ((imageUsageFlag & supportedImageUsage) && validateFormatFeature(imageUsageFlag, supportedFormatFeatures))
			{
				imageUsageFlagSet.insert(imageUsageFlag);
				break;
			}
		}
	}

	if (imageUsageFlagSet.empty())
	{
		LOGEANDABORT("No compatible image usage found.");
	}

	std::string imageUsageList;
	for (VkImageUsageFlagBits imageUsage : imageUsageFlagSet)
	{
		imageUsageList += to_string(imageUsage) + " | ";
	}
	LOGI("Swapchain image usage flags: {}", imageUsageList);

	VkImageUsageFlags imageUsage{};
	for (auto flag : imageUsageFlagSet)
	{
		imageUsage |= flag;
	}
	return imageUsage;
}

VkSurfaceTransformFlagBitsKHR Swapchain::choosePreTransform(
	VkSurfaceTransformFlagBitsKHR requestedTransform,
	VkSurfaceTransformFlagsKHR supportedTransform,
	VkSurfaceTransformFlagBitsKHR currentTransform) const
{
	if (requestedTransform & supportedTransform)
	{
		return requestedTransform;
	}

	LOGW("(Swapchain) Surface transform '{}' not supported. Selecting '{}'.", to_string(requestedTransform), to_string(currentTransform));

	return currentTransform;
}

VkCompositeAlphaFlagBitsKHR Swapchain::chooseCompositeAlpha(VkCompositeAlphaFlagBitsKHR requestedCompositeAlpha, VkCompositeAlphaFlagsKHR supportedCompositeAlpha) const
{
	if (requestedCompositeAlpha & supportedCompositeAlpha)
	{
		return requestedCompositeAlpha;
	}

	for (VkCompositeAlphaFlagBitsKHR compositeAlpha : compositeAlphaPriorityFlags)
	{
		if (compositeAlpha & supportedCompositeAlpha)
		{
			LOGW("(Swapchain) Composite alpha flag '{}' not supported. Defaulting to composite alpha '{}.", to_string(requestedCompositeAlpha), to_string(compositeAlpha));
			return compositeAlpha;
		}
	}

	LOGEANDABORT("A compatible composite alpha was not found.");
}

VkPresentModeKHR Swapchain::choosePresentMode(VkPresentModeKHR requestedPresentMode) const
{
	auto presentModeIt = std::find(availablePresentModes.begin(), availablePresentModes.end(), requestedPresentMode);

	if (presentModeIt == availablePresentModes.end())
	{
		// If nothing found, default to FIFO since that's guarenteed to be supported
		VkPresentModeKHR chosenPresentMode = VK_PRESENT_MODE_FIFO_KHR;

		for (auto &present_mode : presentModePriorityList)
		{
			if (std::find(availablePresentModes.begin(), availablePresentModes.end(), present_mode) != availablePresentModes.end())
			{
				chosenPresentMode = present_mode;
			}
		}

		LOGW("(Swapchain) Present mode '{}' not supported. Selecting '{}'.", to_string(requestedPresentMode), to_string(chosenPresentMode));
		return chosenPresentMode;
	}

	LOGI("(Swapchain) Present mode selected: {}", to_string(requestedPresentMode));
	return *presentModeIt;
}

} // namespace vulkr