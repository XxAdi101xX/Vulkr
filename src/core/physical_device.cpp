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

#include <algorithm>
#include <cstring>

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
	// Check whether the mesh shader extension is available before querying the structures that belong to it
	uint32_t deviceExtensionCount{ 0u };
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &deviceExtensionCount, nullptr);
	std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &deviceExtensionCount, deviceExtensions.data());
	meshShaderSupported = std::any_of(deviceExtensions.begin(), deviceExtensions.end(),
		[](const VkExtensionProperties &extension) { return strcmp(extension.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0; });

	// Requesting acceleration structure properties, ray tracing properties
	rayTracingPipelineProperties.pNext = &accelerationStructureProperties;
	if (meshShaderSupported)
	{
		accelerationStructureProperties.pNext = &meshShaderProperties;
	}
	VkPhysicalDeviceProperties2 properties2{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &rayTracingPipelineProperties,
	};
	vkGetPhysicalDeviceProperties2(gpu, &properties2);

	// Requesting the Vulkan 1.2 and 1.3 features, plus the extension features (acceleration structure, ray tracing, mesh shaders)
	meshShaderFeatures.pNext = &rayTracingPipelineFeatures;
	rayTracingPipelineFeatures.pNext = &accelerationStructureFeatures;
	accelerationStructureFeatures.pNext = &vulkan12Features;
	vulkan12Features.pNext = &vulkan13Features;
	VkPhysicalDeviceFeatures2 features2{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = meshShaderSupported ? static_cast<void *>(&meshShaderFeatures) : static_cast<void *>(&rayTracingPipelineFeatures),
	};
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

const VkPhysicalDeviceVulkan12Features &PhysicalDevice::getVulkan12Features() const
{
	return vulkan12Features;
}

const VkPhysicalDeviceVulkan13Features &PhysicalDevice::getVulkan13Features() const
{
	return vulkan13Features;
}

const VkPhysicalDeviceMeshShaderFeaturesEXT &PhysicalDevice::getMeshShaderFeatures() const
{
	return meshShaderFeatures;
}

const VkPhysicalDeviceMeshShaderPropertiesEXT &PhysicalDevice::getMeshShaderProperties() const
{
	return meshShaderProperties;
}

bool PhysicalDevice::isMeshShaderSupported() const
{
	return meshShaderSupported;
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

} // namespace vulkr