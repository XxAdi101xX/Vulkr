/* Copyright (c) 2022 Adithya Venkatarao
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

#include "instance.h"
#include "device.h"
#include "physical_device.h"
#include "queue.h"

#include "common/helpers.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace vulkr
{

Device::Device(std::unique_ptr<PhysicalDevice> &&selectedPhysicalDevice, VkSurfaceKHR surface, const std::vector<const char *> requestedExtensions) :
	physicalDevice{ std::move(selectedPhysicalDevice) }
{
	// Get all device extension properties
	uint32_t deviceExtensionCount;
	vkEnumerateDeviceExtensionProperties(this->physicalDevice->getHandle(), nullptr, &deviceExtensionCount, nullptr);
	deviceExtensions.resize(deviceExtensionCount);
	vkEnumerateDeviceExtensionProperties(this->physicalDevice->getHandle(), nullptr, &deviceExtensionCount, deviceExtensions.data());

	// Check if all desired extensions are available
	for (auto &requestedExtension : requestedExtensions)
	{
		if (!isExtensionSupported(requestedExtension))
		{
			LOGEANDABORT("Extension {} is not available!", requestedExtension);
		}

		enabledExtensions.emplace_back(requestedExtension);
	}

	// Enable dedicated allocations if it's available to us
	bool canGetMemoryRequirements = isExtensionSupported("VK_KHR_get_memory_requirements2");
	bool supportsDedicatedAllocation = isExtensionSupported("VK_KHR_dedicated_allocation");
	if (canGetMemoryRequirements && supportsDedicatedAllocation)
	{
		enabledExtensions.push_back("VK_KHR_get_memory_requirements2");
		enabledExtensions.push_back("VK_KHR_dedicated_allocation");

		LOGI("Dedicated allocation enabled");
	}

	// Prepare the device queues
	uint32_t queueFamilyPropertiesCount{ to_u32(this->physicalDevice->getQueueFamilyProperties().size()) };
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	queueCreateInfos.reserve(queueFamilyPropertiesCount);
	std::vector<std::vector<float>> queuePriorities(queueFamilyPropertiesCount);

	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queueFamilyPropertiesCount; ++queueFamilyIndex)
	{
		const VkQueueFamilyProperties &queueFamilyProperties = this->physicalDevice->getQueueFamilyProperties()[queueFamilyIndex];

		// Populate queueCreateInfos
		queuePriorities[queueFamilyIndex].resize(queueFamilyProperties.queueCount, 1.0f);

		queueCreateInfos.emplace_back(VkDeviceQueueCreateInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = queueFamilyIndex,
			.queueCount = queueFamilyProperties.queueCount,
			.pQueuePriorities = queuePriorities[queueFamilyIndex].data(),
		});
	}

	// The engine targets Vulkan 1.3, so every feature that was promoted into 1.2/1.3 core is guaranteed to be supported and is
	// requested unconditionally; copying the queried support flags would silently leave a required feature disabled
	VkPhysicalDeviceVulkan13Features vulkan13Features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.synchronization2 = VK_TRUE,
		// Dynamic rendering lets us render without render passes
		.dynamicRendering = VK_TRUE,
		// Maintenance4 is required for usage of the LocalSizeId variable in compute shaders
		.maintenance4 = VK_TRUE,
	};

	VkPhysicalDeviceVulkan12Features vulkan12Features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &vulkan13Features,
		.descriptorIndexing = VK_TRUE,
		.shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
		.descriptorBindingVariableDescriptorCount = VK_TRUE,
		.runtimeDescriptorArray = VK_TRUE,
		.hostQueryReset = VK_TRUE,
		.bufferDeviceAddress = VK_TRUE,
	};

	// Acceleration structure, ray tracing and mesh shader features come from extensions, so those structures are only chained in
	// when the application enabled the matching extension (the Device constructor aborts earlier if a requested extension is missing)
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		.accelerationStructure = VK_TRUE,
	};

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
		.rayTracingPipeline = VK_TRUE,
	};

	VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
		.taskShader = VK_TRUE,
		.meshShader = VK_TRUE,
	};

	const bool accelerationStructureExtensionEnabled{ isExtensionEnabled(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) };
	// The ray tracing pipeline extension depends on the acceleration structure extension
	const bool rayTracingPipelineExtensionEnabled{ accelerationStructureExtensionEnabled && isExtensionEnabled(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) };
	const bool meshShaderExtensionEnabled{ isExtensionEnabled(VK_EXT_MESH_SHADER_EXTENSION_NAME) };

	// Build the pNext chain from the extension features back to the core ones
	void *featureChain = &vulkan12Features;
	if (accelerationStructureExtensionEnabled)
	{
		accelerationStructureFeatures.pNext = featureChain;
		featureChain = &accelerationStructureFeatures;
	}
	if (rayTracingPipelineExtensionEnabled)
	{
		rayTracingPipelineFeatures.pNext = featureChain;
		featureChain = &rayTracingPipelineFeatures;
	}
	if (meshShaderExtensionEnabled)
	{
		meshShaderFeatures.pNext = featureChain;
		featureChain = &meshShaderFeatures;
	}

	// The Vulkan 1.0 features that the engine requires; all of them are core features that desktop GPUs support
	VkPhysicalDeviceFeatures engineVulkan10Features{
		.geometryShader = VK_TRUE,
		.samplerAnisotropy = VK_TRUE,
		.fragmentStoresAndAtomics = VK_TRUE,
		.shaderInt64 = VK_TRUE,
	};

	VkPhysicalDeviceFeatures2 features2{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = featureChain,
		.features = engineVulkan10Features,
	};

	VkDeviceCreateInfo createInfo{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.queueCreateInfoCount = to_u32(queueCreateInfos.size()),
		.pQueueCreateInfos = queueCreateInfos.data(),
		// pEnabledFeatures is left null because the features are enabled through the pNext chain above
		.enabledExtensionCount = to_u32(enabledExtensions.size()),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};

	// Create the device
	VK_CHECK(vkCreateDevice(this->physicalDevice->getHandle(), &createInfo, nullptr, &handle));

	// Create queues
	queues.resize(queueFamilyPropertiesCount);
	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queueFamilyPropertiesCount; ++queueFamilyIndex)
	{
		const VkQueueFamilyProperties &queueFamilyProperties = this->physicalDevice->getQueueFamilyProperties()[queueFamilyIndex];
		bool presentSupported = this->physicalDevice->isPresentSupported(surface, queueFamilyIndex);

		for (uint32_t queueIndex = 0u; queueIndex < queueFamilyProperties.queueCount; ++queueIndex)
		{
			queues[queueFamilyIndex].emplace_back(*this, queueFamilyIndex, queueIndex, queueFamilyProperties, presentSupported);
		}
	}

	// Load device related Vulkan entrypoints directly from the driver to prevent the dispatch overhead incurred from supporting multiple VkDevice objects (see Volk docs)
	volkLoadDevice(handle);

	// Setup VMA
	VmaVulkanFunctions vmaVulkanFunctions{
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
		.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties,
		.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties,
		.vkAllocateMemory = vkAllocateMemory,
		.vkFreeMemory = vkFreeMemory,
		.vkMapMemory = vkMapMemory,
		.vkUnmapMemory = vkUnmapMemory,
		.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges,
		.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges,
		.vkBindBufferMemory = vkBindBufferMemory,
		.vkBindImageMemory = vkBindImageMemory,
		.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements,
		.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements,
		.vkCreateBuffer = vkCreateBuffer,
		.vkDestroyBuffer = vkDestroyBuffer,
		.vkCreateImage = vkCreateImage,
		.vkDestroyImage = vkDestroyImage,
		.vkCmdCopyBuffer = vkCmdCopyBuffer,
	};

	// VK_KHR_buffer_device_address has been core as of vulkan 1.2
	VmaAllocatorCreateInfo allocatorInfo{
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = this->physicalDevice->getHandle(),
		.device = handle,
		.pVulkanFunctions = &vmaVulkanFunctions,
		.instance = this->physicalDevice->getInstance().getHandle(),
	};

	if (canGetMemoryRequirements && supportsDedicatedAllocation)
	{
		allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
		vmaVulkanFunctions.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2KHR;
		vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2KHR;
	}

	vmaVulkanFunctions.vkBindBufferMemory2KHR = vkBindBufferMemory2KHR;
	vmaVulkanFunctions.vkBindImageMemory2KHR = vkBindImageMemory2KHR;
	vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2KHR;
	vmaVulkanFunctions.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
	vmaVulkanFunctions.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;

	VK_CHECK(vmaCreateAllocator(&allocatorInfo, &memoryAllocator));
}


Device::~Device()
{
	if (memoryAllocator != VK_NULL_HANDLE)
	{
#ifdef VULKR_DEBUG
		VmaTotalStatistics stats;
		vmaCalculateStatistics(memoryAllocator, &stats);

		VkDeviceSize unusedBytes = stats.total.statistics.blockBytes - stats.total.statistics.allocationBytes;
		if (unusedBytes > 0)
		{
			LOGW("Total vma memory leaked: {} bytes. Note that this might not be due to Vulkr's usage; VMA should throw an error in debug mode if there is allocated memory that is unfreed", unusedBytes);
		}
#endif

		vmaDestroyAllocator(memoryAllocator);
	}

	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyDevice(handle, nullptr);
	}
}

void Device::waitIdle() const
{
	vkDeviceWaitIdle(handle);
}

VkDevice Device::getHandle() const
{
	return handle;
}

const PhysicalDevice &Device::getPhysicalDevice() const
{
	return *physicalDevice;
}

const int32_t Device::getQueueFamilyIndexByFlags(VkQueueFlags desiredQueueFlags, bool requiresPresentation) const
{
	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queues.size(); ++queueFamilyIndex)
	{
		const Queue &firstQueueInFamily = queues[queueFamilyIndex][0];

		if (firstQueueInFamily.getProperties().queueCount > 0 &&
			firstQueueInFamily.supportsQueueFlags(desiredQueueFlags) &&
			(requiresPresentation ? firstQueueInFamily.canSupportPresentation() : true))
		{
			return queueFamilyIndex;
		}
	}

	return -1;
}

Queue *Device::getQueue(uint32_t queueFamilyIndex, uint32_t queueIndex)
{
	return &queues[queueFamilyIndex][queueIndex];
}

bool Device::isExtensionSupported(const char *extension) const
{
	for (auto &existingExtension : deviceExtensions)
	{
		if (strcmp(existingExtension.extensionName, extension) == 0)
		{
			return true;
		}
	}

	return false;
}

bool Device::isExtensionEnabled(const char *extension) const
{
	return std::find_if(enabledExtensions.begin(), enabledExtensions.end(), [extension](const char *enabledExtension) { return strcmp(extension, enabledExtension) == 0; }) != enabledExtensions.end();
}

VmaAllocator Device::getMemoryAllocator() const
{
	return memoryAllocator;
}

const VkAllocationCallbacks *Device::getAllocationCallbacks() const
{
	return memoryAllocator->GetAllocationCallbacks();
}

uint32_t Device::getMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags propertyFlags) const
{
	for (uint32_t i = 0; i < physicalDevice->getMemoryProperties().memoryTypeCount; ++i)
	{
		if ((memoryTypeBits & (1 << i)) && (physicalDevice->getMemoryProperties().memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags)
		{
			return i;
		}
	}

	LOGEANDABORT("Failed to find suitable memory type!");
}

} // namespace vulkr