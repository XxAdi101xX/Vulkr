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
	PhysicalDevice(Instance &instance, VkPhysicalDevice physicalDevice);
	~PhysicalDevice() = default;

	PhysicalDevice(const PhysicalDevice &) = delete;
	PhysicalDevice(PhysicalDevice &&) = delete;
	PhysicalDevice &operator=(const PhysicalDevice &) = delete;
	PhysicalDevice &operator=(PhysicalDevice &&) = delete;

	/* Get the physical device handle */
	VkPhysicalDevice getHandle() const;

	/* Get instance */
	Instance &getInstance() const;

	/* Get all the physical device features supported */
	const VkPhysicalDeviceFeatures &getFeatures() const;

	/* Get the physical device features that were requested by the application */
	const VkPhysicalDeviceFeatures &getRequestedFeatures() const;

	/* Get the properties for the physical device */
	const VkPhysicalDeviceProperties &getProperties() const;

	/* Get the ray tracing pipeline properties for the physical device */
	const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &getRayTracingPipelineProperties() const;

	/* Get the ray tracing pipeline features for the physical device */
	const VkPhysicalDeviceRayTracingPipelineFeaturesKHR &getRayTracingPipelineFeatures() const;

	/* Get the acceleration structure properties for the physical device */
	const VkPhysicalDeviceAccelerationStructurePropertiesKHR &getAccelerationStructureProperties() const;

	/* Get the acceleration structure features for the physical device */
	const VkPhysicalDeviceAccelerationStructureFeaturesKHR &getAccelerationStructureFeatures() const;

	/* Get the host query reset features for the physical device */
	const VkPhysicalDeviceHostQueryResetFeatures &getHostQueryResetFeatures() const;

	/* Get the ray tracing pipeline properties for the physical device */
	const VkPhysicalDeviceBufferDeviceAddressFeatures &getBufferDeviceAddressFeatures() const;

	/* Get the descriptor indexing features for the physical device */
	const VkPhysicalDeviceDescriptorIndexingFeatures &getDescriptorIndexingFeatures() const;

	/* Get the memory properties for the physical device */
	const VkPhysicalDeviceMemoryProperties &getMemoryProperties() const;

	/* Get an array of all the queue family properties for each queue family available */
	const std::vector<VkQueueFamilyProperties> &getQueueFamilyProperties() const;

	/* Check whether a queue family supports presentation */
	VkBool32 isPresentSupported(VkSurfaceKHR surface, uint32_t queue_family_index) const;

	/* Set the requested features */
	void setRequestedFeatures(VkPhysicalDeviceFeatures &requestedFeatures);
private:
	/* The physical device handle */
	VkPhysicalDevice handle{ VK_NULL_HANDLE };

	/* The associated Vulkan instance */
	Instance &instance;

	/* The features that the GPU supports */
	VkPhysicalDeviceFeatures features{};

	/* The requested features to be enabled within the logical device */
	VkPhysicalDeviceFeatures requestedFeatures{};

	/* The GPU properties */
	VkPhysicalDeviceProperties properties;

	/* The GPU ray tracing pipeline properties */
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rayTracingPipelineProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

	/* The GPU ray tracing pipeline features */
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };

	/* The GPU acceleration structure properties */
	VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };

	/* The GPU acceleration structure features */
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };

	/* The GPU host query rest features */
	VkPhysicalDeviceHostQueryResetFeatures hostQueryResetFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES };

	/* The buffer device address features that the GPU supports */
	VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES };

	/* The descriptor indexing features that the GPU supports */
	VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };

	/* The GPU memory properties */
	VkPhysicalDeviceMemoryProperties memoryProperties;

	/* The GPU queue family properties */
	std::vector<VkQueueFamilyProperties> queueFamilyProperties;
}; // class PhysicalDevice

} // namespace vulkr
