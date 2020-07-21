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

#include "physical_device.h"

namespace vulkr
{

PhysicalDevice::PhysicalDevice(Instance& instance, VkPhysicalDevice gpu) :
	instance{ instance },
	gpu{ gpu }
{
	vkGetPhysicalDeviceFeatures(gpu, &features);
	vkGetPhysicalDeviceProperties(gpu, &properties);
	vkGetPhysicalDeviceMemoryProperties(gpu, &memoryProperties);

	uint32_t queueFamilyPropertiesCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyPropertiesCount, nullptr);
	queueFamilyProperties = std::vector<VkQueueFamilyProperties>(queueFamilyPropertiesCount);
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyPropertiesCount, queueFamilyProperties.data());
}

VkPhysicalDevice PhysicalDevice::getHandle() const
{
	return gpu;
}

const VkPhysicalDeviceFeatures& PhysicalDevice::getFeatures() const
{
	return features;
}

const VkPhysicalDeviceProperties PhysicalDevice::getProperties() const
{
	return properties;
}

const VkPhysicalDeviceMemoryProperties PhysicalDevice::getMemoryProperties() const
{
	return memoryProperties;
}

const std::vector<VkQueueFamilyProperties>& PhysicalDevice::getQueueFamilyProperties() const
{
	return queueFamilyProperties;
}

} // namespace vulkr