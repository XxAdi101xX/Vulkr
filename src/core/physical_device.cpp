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
#include "common/logger.h"

namespace vulkr
{

PhysicalDevice::PhysicalDevice(Instance &instance, VkPhysicalDevice gpu) :
	instance{ instance },
	handle{ gpu }
{
	// Requesting physical device features, properties and memory properties
	vkGetPhysicalDeviceFeatures(gpu, &features);
	vkGetPhysicalDeviceProperties(gpu, &properties);
	vkGetPhysicalDeviceMemoryProperties(gpu, &memoryProperties);
	// Requesting acceleration structure properties, ray tracing properties
	rayTracingPipelineProperties.pNext = &accelerationStructureProperties;
	VkPhysicalDeviceProperties2 properties2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	properties2.pNext = &rayTracingPipelineProperties;
	vkGetPhysicalDeviceProperties2(gpu, &properties2);

	// Requesting host query reset features, buffer device address features, raytracing features and acceleration structure features
	hostQueryResetFeatures.pNext = &descriptorIndexingFeatures;
	accelerationStructureFeatures.pNext = &hostQueryResetFeatures;
	rayTracingPipelineFeatures.pNext = &accelerationStructureFeatures;
	bufferDeviceAddressFeatures.pNext = &rayTracingPipelineFeatures;
	VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	features2.pNext = &bufferDeviceAddressFeatures;
	vkGetPhysicalDeviceFeatures2(gpu, &features2);

	uint32_t queueFamilyPropertiesCount{ 0u };
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyPropertiesCount, nullptr);
	queueFamilyProperties = std::vector<VkQueueFamilyProperties>(queueFamilyPropertiesCount);
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyPropertiesCount, queueFamilyProperties.data());

	LOGI("Selected GPU: {}", this->getProperties().deviceName);
}

VkPhysicalDevice PhysicalDevice::getHandle() const
{
	return handle;
}

Instance &PhysicalDevice::getInstance() const
{
	return instance;
}

const VkPhysicalDeviceFeatures &PhysicalDevice::getFeatures() const
{
	return features;
}

const VkPhysicalDeviceFeatures &PhysicalDevice::getRequestedFeatures() const
{
	return requestedFeatures;
}

const VkPhysicalDeviceProperties &PhysicalDevice::getProperties() const
{
	return properties;
}

const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &PhysicalDevice::getRayTracingPipelineProperties() const
{
	return rayTracingPipelineProperties;
}

const VkPhysicalDeviceRayTracingPipelineFeaturesKHR &PhysicalDevice::getRayTracingPipelineFeatures() const
{
	return rayTracingPipelineFeatures;
}

const VkPhysicalDeviceAccelerationStructurePropertiesKHR &PhysicalDevice::getAccelerationStructureProperties() const
{
	return accelerationStructureProperties;
}

const VkPhysicalDeviceAccelerationStructureFeaturesKHR &PhysicalDevice::getAccelerationStructureFeatures() const
{
	return accelerationStructureFeatures;
}

const VkPhysicalDeviceHostQueryResetFeatures &PhysicalDevice::getHostQueryResetFeatures() const
{
	return hostQueryResetFeatures;
}

const VkPhysicalDeviceBufferDeviceAddressFeatures &PhysicalDevice::getBufferDeviceAddressFeatures() const
{
	return bufferDeviceAddressFeatures;
}

const VkPhysicalDeviceDescriptorIndexingFeatures &PhysicalDevice::getDescriptorIndexingFeatures() const
{
	return descriptorIndexingFeatures;
}

const VkPhysicalDeviceMemoryProperties &PhysicalDevice::getMemoryProperties() const
{
	return memoryProperties;
}

const std::vector<VkQueueFamilyProperties> &PhysicalDevice::getQueueFamilyProperties() const
{
	return queueFamilyProperties;
}

VkBool32 PhysicalDevice::isPresentSupported(VkSurfaceKHR surface, uint32_t queue_family_index) const
{
	VkBool32 presentSupported{ VK_FALSE };

	if (surface != VK_NULL_HANDLE)
	{
		VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(handle, queue_family_index, surface, &presentSupported));
	}

	return presentSupported;
}

void PhysicalDevice::setRequestedFeatures(VkPhysicalDeviceFeatures &requestedFeatures)
{
	this->requestedFeatures = requestedFeatures;
}

} // namespace vulkr