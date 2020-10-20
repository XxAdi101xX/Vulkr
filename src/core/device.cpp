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

#include "device.h"
#include "physical_device.h"
#include "queue.h"

#include "common/helpers.h"

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

		VkDeviceQueueCreateInfo queueCreateInfo { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = queueFamilyProperties.queueCount;
		queueCreateInfo.pQueuePriorities = queuePriorities[queueFamilyIndex].data();

		queueCreateInfos.emplace_back(queueCreateInfo);
	}

	// Create the device
	const VkPhysicalDeviceFeatures &requestedFeatures = this->physicalDevice->getRequestedFeatures();

	VkDeviceCreateInfo createInfo { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	createInfo.queueCreateInfoCount = to_u32(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.enabledExtensionCount = to_u32(enabledExtensions.size());
	createInfo.ppEnabledExtensionNames = enabledExtensions.data();
	createInfo.pEnabledFeatures = &requestedFeatures;

	VK_CHECK(vkCreateDevice(this->physicalDevice->getHandle(), &createInfo, nullptr, &handle));

	// Create queues
	queues.resize(queueFamilyPropertiesCount);
	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queueFamilyPropertiesCount; ++queueFamilyIndex)
	{
		const VkQueueFamilyProperties& queueFamilyProperties = this->physicalDevice->getQueueFamilyProperties()[queueFamilyIndex];
		bool presentSupported = this->physicalDevice->isPresentSupported(surface, queueFamilyIndex);

		for (uint32_t queueIndex = 0u; queueIndex < queueFamilyProperties.queueCount; ++queueIndex)
		{
			queues[queueFamilyIndex].emplace_back(*this, queueFamilyIndex, queueIndex, queueFamilyProperties, presentSupported);
		}
	}

	// Load device-related Vulkan entrypoints directly from the driver to prevent the dispatch overhead incurred from supporting multiple VkDevice objects (see Volk docs)
    volkLoadDevice(handle);
}


Device::~Device()
{
	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyDevice(handle, nullptr);
	}
}

void Device::waitIdle()
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

const Queue &Device::getOptimalGraphicsQueue()
{
	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queues.size(); ++queueFamilyIndex)
	{
		Queue &firstQueueInFamily = queues[queueFamilyIndex][0];

		if (firstQueueInFamily.getProperties().queueCount > 0 && firstQueueInFamily.canSupportPresentation())
		{
			return firstQueueInFamily;
		}
	}

	return getQueueByFlags(VK_QUEUE_GRAPHICS_BIT);
}

const Queue &Device::getQueueByFlags(VkQueueFlags desiredQueueFlags)
{
	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queues.size(); ++queueFamilyIndex)
	{
		Queue& firstQueueInFamily = queues[queueFamilyIndex][0];

		if (firstQueueInFamily.getProperties().queueCount > 0 && firstQueueInFamily.supportsQueueFlags(desiredQueueFlags))
		{
			return firstQueueInFamily;
		}
	}

	LOGEANDABORT("Could not find a queue with the desired queueflags");
}

const Queue &Device::getQueueByPresentation()
{
	for (uint32_t queueFamilyIndex = 0u; queueFamilyIndex < queues.size(); ++queueFamilyIndex)
	{
		Queue& firstQueueInFamily = queues[queueFamilyIndex][0];

		if (firstQueueInFamily.getProperties().queueCount > 0 && firstQueueInFamily.canSupportPresentation())
		{
			return firstQueueInFamily;
		}
	}

	LOGEANDABORT("Could not find a queue with presentation support");
}

bool Device::isExtensionSupported(const char *extension) const
{
	for (auto& existingExtension : deviceExtensions)
	{
		if (strcmp(existingExtension.extensionName, extension) == 0)
		{
			return true;
		}
	}

	return false;
}

} // namespace vulkr