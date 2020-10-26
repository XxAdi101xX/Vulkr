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

#include <set>

#include "common/vulkan_common.h"

namespace vulkr
{

class Device;
class Image;

struct SwapchainProperties
{
	uint32_t imageCount{ 3u };
	VkSurfaceFormatKHR surfaceFormat{};
	VkExtent2D imageExtent{};
	uint32_t imageArraylayers{ 1u };
	VkImageUsageFlags imageUsage;
	VkSurfaceTransformFlagBitsKHR preTransform;
	VkCompositeAlphaFlagBitsKHR compositeAlpha;
	VkPresentModeKHR presentMode;
	VkBool32 clipped;
	VkSwapchainKHR oldSwapchain{ VK_NULL_HANDLE };
};

class Swapchain
{
public:
	Swapchain::Swapchain(
		Device &device,
		VkSurfaceKHR surface,
		const VkSurfaceTransformFlagBitsKHR transform,
		const VkPresentModeKHR presentMode,
		const std::set<VkImageUsageFlagBits> &imageUsageFlags);
	~Swapchain();

	/* Disable unnecessary operators to prevent error prone usages */
	Swapchain(const Swapchain &) = delete;
	Swapchain(Swapchain &&) = delete;
	Swapchain& operator=(const Swapchain &) = delete;
	Swapchain& operator=(Swapchain &&) = delete;

	VkSwapchainKHR getHandle() const;
	const SwapchainProperties &getProperties() const;
	const std::vector<std::unique_ptr<Image>> &getImages() const;
private:
	/* The swapchain handle */
	VkSwapchainKHR handle{ VK_NULL_HANDLE };

	/* The surface handle */
	VkSurfaceKHR surface{ VK_NULL_HANDLE };

	/* The logical device associated with the swapchain */
	Device &device;

	/* The associating properties of the swapchain */
	SwapchainProperties properties{};
	
	/* The images associated with the swapchain */
	std::vector<std::unique_ptr<Image>> images;
	// TODO: should I add the image_views here and also destruct them

	/* All available surface formats available to use */
	std::vector<VkSurfaceFormatKHR> availableSurfaceFormats{};

	/* All available present modes available to use */
	std::vector<VkPresentModeKHR> availablePresentModes{};

	/* A list of present modes in decreasing priority */
	const std::vector<VkPresentModeKHR> presentModePriorityList = {
		VK_PRESENT_MODE_FIFO_KHR, /* Ideal option */
		VK_PRESENT_MODE_MAILBOX_KHR
	};

	/* A list of surface formats in descreasing priority */
	const std::vector<VkSurfaceFormatKHR> surfaceFormatPriorityList = {
		{VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}, /* Ideal option */
		{VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
		{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
		{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
	};

	/* A list of default image usage flags to fall back on */
	const std::vector<VkImageUsageFlagBits> imageUsagePriorityFlags = {
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		VK_IMAGE_USAGE_STORAGE_BIT,
		VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT
	};

	/* A list of default composite alpha modes */
	const std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaPriorityFlags = {
		VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
	};

	void create();

	/* Swapchain properties selection helper functions */
	uint32_t chooseImageCount(uint32_t minImageCount, uint32_t maxImageCount) const;
	VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) const;
	VkExtent2D chooseImageExtent(VkExtent2D currentExtent, VkExtent2D minImageExtent, VkExtent2D maxImageExtent) const;
	uint32_t chooseImageArrayLayers(uint32_t requestedImageArrayLayers, uint32_t maxImageArrayLayers) const;
	bool validateFormatFeature(VkImageUsageFlagBits imageUsage, VkFormatFeatureFlags supportedFormatFeatures) const; /* used in chooseImageUsage */
	VkImageUsageFlags chooseImageUsage(const std::set<VkImageUsageFlagBits> &requestedImageUsageFlags, VkImageUsageFlags supportedImageUsage, VkFormatFeatureFlags supportedFormatFeatures) const;
	VkSurfaceTransformFlagBitsKHR choosePreTransform(VkSurfaceTransformFlagBitsKHR requestedTransform, VkSurfaceTransformFlagsKHR supportedTransform, VkSurfaceTransformFlagBitsKHR currentTransform) const;
	VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(VkCompositeAlphaFlagBitsKHR requestedCompositeAlpha, VkCompositeAlphaFlagsKHR supportedCompositeAlpha) const;
	VkPresentModeKHR choosePresentMode(VkPresentModeKHR requestedPresentMode) const;
	// clipped
	// old swapchain
};

} // namespace vulkr