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

		VkDeviceQueueCreateInfo queueCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = queueFamilyProperties.queueCount;
		queueCreateInfo.pQueuePriorities = queuePriorities[queueFamilyIndex].data();

		queueCreateInfos.emplace_back(queueCreateInfo);
	}

	// Enable synchronization2
	VkPhysicalDeviceSynchronization2Features synchronization2Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES };
	synchronization2Features.synchronization2 = VK_TRUE;
	// Enable maintenance4 features; required for usage of the LocalSizeId variable in compute shaders
	VkPhysicalDeviceMaintenance4Features maintenance4Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES };
	maintenance4Features.maintenance4 = VK_TRUE;
	maintenance4Features.pNext = &synchronization2Features;
	// Enabling the descriptor indexing features, host query reset features, acceleration structure features, ray tracing features and buffer device address features
	VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };
	descriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = getPhysicalDevice().getDescriptorIndexingFeatures().shaderSampledImageArrayNonUniformIndexing;
	descriptorIndexingFeatures.runtimeDescriptorArray = getPhysicalDevice().getDescriptorIndexingFeatures().runtimeDescriptorArray;
	descriptorIndexingFeatures.pNext = &maintenance4Features;
	VkPhysicalDeviceHostQueryResetFeatures hostQueryResetFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES };
	hostQueryResetFeatures.hostQueryReset = getPhysicalDevice().getHostQueryResetFeatures().hostQueryReset;
	hostQueryResetFeatures.pNext = &descriptorIndexingFeatures;
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	accelerationStructureFeatures.accelerationStructure = getPhysicalDevice().getAccelerationStructureFeatures().accelerationStructure;
	// accelerationStructureFeatures.accelerationStructureHostCommands = getPhysicalDevice().getAccelerationStructureFeatures().accelerationStructureHostCommands;
	// accelerationStructureFeatures.accelerationStructureIndirectBuild = getPhysicalDevice().getAccelerationStructureFeatures().accelerationStructureIndirectBuild;
	accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = getPhysicalDevice().getAccelerationStructureFeatures().descriptorBindingAccelerationStructureUpdateAfterBind;
	accelerationStructureFeatures.pNext = &hostQueryResetFeatures;
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	rayTracingPipelineFeatures.rayTracingPipeline = getPhysicalDevice().getRayTracingPipelineFeatures().rayTracingPipeline;
	rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = getPhysicalDevice().getRayTracingPipelineFeatures().rayTracingPipelineTraceRaysIndirect;
	rayTracingPipelineFeatures.rayTraversalPrimitiveCulling = getPhysicalDevice().getRayTracingPipelineFeatures().rayTraversalPrimitiveCulling;
	rayTracingPipelineFeatures.pNext = &accelerationStructureFeatures;
	VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES };
	bufferDeviceAddressFeatures.bufferDeviceAddress = getPhysicalDevice().getBufferDeviceAddressFeatures().bufferDeviceAddress;
	bufferDeviceAddressFeatures.pNext = &rayTracingPipelineFeatures;

	VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	features2.features = this->physicalDevice->getRequestedFeatures(); // These features are specified in app.h
	features2.pNext = &bufferDeviceAddressFeatures;

	VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	createInfo.queueCreateInfoCount = to_u32(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.enabledExtensionCount = to_u32(enabledExtensions.size());
	createInfo.ppEnabledExtensionNames = enabledExtensions.data();
	// createInfo.pEnabledFeatures = &this->physicalDevice->getRequestedFeatures(); // We are enabling the features through feature2
	createInfo.pNext = &features2;

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
	VmaVulkanFunctions vmaVulkanFunctions{};
	vmaVulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vmaVulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
	vmaVulkanFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
	vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
	vmaVulkanFunctions.vkAllocateMemory = vkAllocateMemory;
	vmaVulkanFunctions.vkFreeMemory = vkFreeMemory;
	vmaVulkanFunctions.vkMapMemory = vkMapMemory;
	vmaVulkanFunctions.vkUnmapMemory = vkUnmapMemory;
	vmaVulkanFunctions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
	vmaVulkanFunctions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
	vmaVulkanFunctions.vkBindBufferMemory = vkBindBufferMemory;
	vmaVulkanFunctions.vkBindImageMemory = vkBindImageMemory;
	vmaVulkanFunctions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
	vmaVulkanFunctions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
	vmaVulkanFunctions.vkCreateBuffer = vkCreateBuffer;
	vmaVulkanFunctions.vkDestroyBuffer = vkDestroyBuffer;
	vmaVulkanFunctions.vkCreateImage = vkCreateImage;
	vmaVulkanFunctions.vkDestroyImage = vkDestroyImage;
	vmaVulkanFunctions.vkCmdCopyBuffer = vkCmdCopyBuffer;

	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = this->physicalDevice->getHandle();
	allocatorInfo.device = handle;
	allocatorInfo.instance = this->physicalDevice->getInstance().getHandle();

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

	// VK_KHR_buffer_device_address has been core as of vulkan 1.2
	allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

	allocatorInfo.pVulkanFunctions = &vmaVulkanFunctions;

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