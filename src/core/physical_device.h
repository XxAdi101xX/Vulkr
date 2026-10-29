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

	/* Get the Vulkan 1.2 features for the physical device (descriptor indexing, host query reset, buffer device address) */
	const VkPhysicalDeviceVulkan12Features &getVulkan12Features() const;

	/* Get the Vulkan 1.3 features for the physical device (synchronization2, dynamic rendering, maintenance4) */
	const VkPhysicalDeviceVulkan13Features &getVulkan13Features() const;

	/* Get the mesh shader features for the physical device */
	const VkPhysicalDeviceMeshShaderFeaturesEXT &getMeshShaderFeatures() const;

	/* Get the mesh shader properties for the physical device */
	const VkPhysicalDeviceMeshShaderPropertiesEXT &getMeshShaderProperties() const;

	/* Check whether VK_EXT_mesh_shader is supported by the physical device */
	bool isMeshShaderSupported() const;

	/* Get the memory properties for the physical device */
	const VkPhysicalDeviceMemoryProperties &getMemoryProperties() const;

	/* Get an array of all the queue family properties for each queue family available */
	const std::vector<VkQueueFamilyProperties> &getQueueFamilyProperties() const;

	/* Check whether a queue family supports presentation */
	VkBool32 isPresentSupported(VkSurfaceKHR surface, uint32_t queue_family_index) const;
private:
	/* The physical device handle */
	VkPhysicalDevice handle{ VK_NULL_HANDLE };

	/* The associated Vulkan instance */
	Instance &instance;

	/* The features that the GPU supports */
	VkPhysicalDeviceFeatures features{};

	/* The GPU properties */
	VkPhysicalDeviceProperties properties;

	/* The GPU ray tracing pipeline properties */
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rayTracingPipelineProperties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

	/* The GPU ray tracing pipeline features */
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };

	/* The GPU acceleration structure properties */
	VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };

	/* The GPU acceleration structure features */
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };

	/* The Vulkan 1.2 features that the GPU supports; these were promoted from extensions and are queried as one struct */
	VkPhysicalDeviceVulkan12Features vulkan12Features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };

	/* The Vulkan 1.3 features that the GPU supports; these were promoted from extensions and are queried as one struct */
	VkPhysicalDeviceVulkan13Features vulkan13Features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };

	/* Whether VK_EXT_mesh_shader is available on the GPU */
	bool meshShaderSupported{ false };

	/* The mesh shader features that the GPU supports */
	VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };

	/* The mesh shader limits of the GPU */
	VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProperties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT };

	/* The GPU memory properties */
	VkPhysicalDeviceMemoryProperties memoryProperties;

	/* The GPU queue family properties */
	std::vector<VkQueueFamilyProperties> queueFamilyProperties;
}; // class PhysicalDevice

} // namespace vulkr
