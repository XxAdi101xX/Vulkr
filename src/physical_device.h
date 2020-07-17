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

#include "instance.h"

namespace vulkr {
	class PhysicalDevice
	{
	public:
		PhysicalDevice(Instance& instance, VkPhysicalDevice physicalDevice);

		/* Disable unnecessary operators to prevent error prone usages */
		PhysicalDevice(const PhysicalDevice&) = delete;
		PhysicalDevice(PhysicalDevice&&) = delete;
		PhysicalDevice& operator=(const PhysicalDevice&) = delete;
		PhysicalDevice& operator=(PhysicalDevice&&) = delete;

		VkPhysicalDevice getHandle() const;

		const VkPhysicalDeviceFeatures& getFeatures() const;

		const VkPhysicalDeviceProperties getProperties() const;

		const VkPhysicalDeviceMemoryProperties getMemoryProperties() const;

		const std::vector<VkQueueFamilyProperties>& getQueueFamilyProperties() const;
	private:
		// TODO: check for device extension support
		/* The physical device handle */
		VkPhysicalDevice physicalDevice { VK_NULL_HANDLE };

		/* The associated Vulkan instance */
		Instance& instance;

		/* The features that the GPU supports */
		VkPhysicalDeviceFeatures features{};

		/* The GPU properties */
		VkPhysicalDeviceProperties properties;

		/* The GPU memory properties */
		VkPhysicalDeviceMemoryProperties memoryProperties;

		/* The GPU queue family properties */
		std::vector<VkQueueFamilyProperties> queueFamilyProperties;
	}; // class PhysicalDevice
} // namespace vulkr
