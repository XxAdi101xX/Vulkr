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

#include <vector>

#include "common/vulkan_common.h"

namespace vulkr
{

class Instance;

class PhysicalDevice
{
public:
	PhysicalDevice(Instance& instance, VkPhysicalDevice physicalDevice);

	/* Disable unnecessary operators to prevent error prone usages */
	PhysicalDevice(const PhysicalDevice &) = delete;
	PhysicalDevice(PhysicalDevice &&) = delete;
	PhysicalDevice& operator=(const PhysicalDevice &) = delete;
	PhysicalDevice& operator=(PhysicalDevice &&) = delete;

	/* Get the physical device handle */
	VkPhysicalDevice getHandle() const;

	/* Get all the physical device features supported */
	const VkPhysicalDeviceFeatures &getFeatures() const;

	/* Get the physical device features that were requested by the application */
	const VkPhysicalDeviceFeatures &getRequestedFeatures() const;

	/* Get the properties for the physical device */
	const VkPhysicalDeviceProperties getProperties() const;

	/* Get the memory properties for the physical device */
	const VkPhysicalDeviceMemoryProperties getMemoryProperties() const;

	/* Get an array of all the queue family properties for each queue family available */
	const std::vector<VkQueueFamilyProperties> &getQueueFamilyProperties() const;

	/* Check whether a queue family supports presentation */
	VkBool32 isPresentSupported(VkSurfaceKHR surface, uint32_t queue_family_index) const;
private:
	/* The physical device handle */
	VkPhysicalDevice handle { VK_NULL_HANDLE };

	/* The associated Vulkan instance */
	Instance &instance;

	/* The features that the GPU supports */
	VkPhysicalDeviceFeatures features{};

	/* The requested features to be enabled within the logical device */
	VkPhysicalDeviceFeatures requestedFeatures{};

	/* The GPU properties */
	VkPhysicalDeviceProperties properties;

	/* The GPU memory properties */
	VkPhysicalDeviceMemoryProperties memoryProperties;

	/* The GPU queue family properties */
	std::vector<VkQueueFamilyProperties> queueFamilyProperties;
}; // class PhysicalDevice

} // namespace vulkr
