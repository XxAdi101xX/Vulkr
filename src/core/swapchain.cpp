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

#include "common/logger.h"
#include "platform/window.h"
#include "swapchain.h"

namespace vulkr
{

uint32_t Swapchain::chooseImageCount(uint32_t minImageCount, uint32_t maxImageCount)
{
    // We default to 1 more than the minimum as per the suggestions in the vulkan-tutorial
    uint32_t imageCount = minImageCount + 1;
    if (maxImageCount > 0 && imageCount > maxImageCount) {
        return maxImageCount;
    }
    return imageCount;
}

VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& availableFormats,
    const std::vector<VkSurfaceFormatKHR> surfaceFormatPriorityList)
{
    if (availableFormats.empty())
    {
        throw std::runtime_error("No surface formats were found.");
    }

    for (const auto& prioritySurfaceFormat : surfaceFormatPriorityList)
    {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == prioritySurfaceFormat.format && availableFormat.colorSpace == prioritySurfaceFormat.colorSpace)
            {
                return availableFormat;
            }
        }
    }

    LOGW("A surface format from the priority list was not found hence the first available format was chosen");
    return availableFormats[0];
}

VkExtent2D Swapchain::chooseImageExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }

    VkExtent2D extent = Window::getWindowExtent();
    extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, extent.width));
    extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, extent.height));

    return extent;
}

uint32_t Swapchain::chooseImageArrayLayers(
    uint32_t requestedImageArrayLayers,
    uint32_t maxImageArrayLayers)
{
    requestedImageArrayLayers = std::min(requestedImageArrayLayers, maxImageArrayLayers);
    requestedImageArrayLayers = std::max(requestedImageArrayLayers, 1u);

    return requestedImageArrayLayers;
}

bool Swapchain::validateFormatFeature(VkImageUsageFlagBits imageUsage, VkFormatFeatureFlags supportedFormatFeatures)
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
    const std::set<VkImageUsageFlagBits>& requestedImageUsageFlags,
    VkImageUsageFlags supportedImageUsage,
    VkFormatFeatureFlags supportedFormatFeatures)
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
			LOGW("(Swapchain) Image usage ({}) requested but not supported.", flag);
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
		throw std::runtime_error("No compatible image usage found.");
	}

    /* Optional: print out flags */

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
    VkSurfaceTransformFlagBitsKHR currentTransform)
{
    if (requestedTransform & supportedTransform)
    {
        return requestedTransform;
    }

    LOGW("Requested transform not available, defaulting to the current existing transform");

    return currentTransform;
}

VkCompositeAlphaFlagBitsKHR Swapchain::chooseCompositeAlpha(VkCompositeAlphaFlagBitsKHR requestedCompositeAlpha, VkCompositeAlphaFlagsKHR supportedCompositeAlpha)
{
    if (requestedCompositeAlpha & supportedCompositeAlpha)
    {
        return requestedCompositeAlpha;
    }

    for (VkCompositeAlphaFlagBitsKHR compositeAlpha : compositeAlphaPriorityFlags)
    {
        if (compositeAlpha & supportedCompositeAlpha)
        {
            LOGW("(Swapchain) Composite alpha flag '{}' not supported. Defaulting to composite alpha '{}.", requestedCompositeAlpha, compositeAlpha);
            return compositeAlpha;
        }
    }

    throw std::runtime_error("A compatible composite alpha was not found.");
}

VkPresentModeKHR Swapchain::choosePresentMode(VkPresentModeKHR requestedPresentMode, const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    auto presentModeIt = std::find(availablePresentModes.begin(), availablePresentModes.end(), requestedPresentMode);

    if (presentModeIt == availablePresentModes.end())
    {
        // If nothing found, default to FIFO since that's guarenteed to be supported
        VkPresentModeKHR chosenPresentMode = VK_PRESENT_MODE_FIFO_KHR;

        for (auto& present_mode : presentModePriorityList)
        {
            if (std::find(availablePresentModes.begin(), availablePresentModes.end(), present_mode) != availablePresentModes.end())
            {
                chosenPresentMode = present_mode;
            }
        }

        LOGW("(Swapchain) Present mode '{}' not supported. Selecting '{}'.", requestedPresentMode, chosenPresentMode);
        return chosenPresentMode;
    }
    else
    {
        LOGI("(Swapchain) Present mode selected: {}", requestedPresentMode);
        return *presentModeIt;
    }
}

} // namespace vulkr