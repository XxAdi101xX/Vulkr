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

namespace vulkr
{

Device::Device(PhysicalDevice& gpu, std::vector<const char *> requestedExtensions) :
    gpu{ gpu }
{
	// Get all device extension properties
	uint32_t deviceExtensionCount;
	vkEnumerateDeviceExtensionProperties(gpu.getHandle(), nullptr, &deviceExtensionCount, nullptr);
	deviceExtensions.resize(deviceExtensionCount);
	vkEnumerateDeviceExtensionProperties(gpu.getHandle(), nullptr, &deviceExtensionCount, deviceExtensions.data());

	// Check if all desired extensions are available
	for (auto &requestedExtension : requestedExtensions)
	{
		if (!isExtensionSupported(requestedExtension))
		{
			LOGE("Extension {} is not available!", requestedExtension);
		}
	}

	// Prepare the device queues
	uint32_t queueFamilyPropertiesCount{ static_cast<uint32_t>(gpu.getQueueFamilyProperties().size()) };
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(queueFamilyPropertiesCount);
	std::vector<std::vector<float>> queuePriorities(queueFamilyPropertiesCount);

	for (uint32_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilyPropertiesCount; ++queueFamilyIndex)
	{
		const VkQueueFamilyProperties& queueFamilyProperties = gpu.getQueueFamilyProperties()[queueFamilyIndex];

		queuePriorities[queueFamilyIndex].resize(queueFamilyProperties.queueCount, 1.0f);

		VkDeviceQueueCreateInfo queueCreateInfo { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = queueFamilyProperties.queueCount;
		queueCreateInfo.pQueuePriorities = queuePriorities[queueFamilyIndex].data();

		queueCreateInfos.emplace_back(queueCreateInfo);
	}

	//VkDeviceCreateInfo create_info { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	//create_info.pQueueCreateInfos = queueCreateInfos.data();
	//create_info.queueCreateInfoCount = queueCreateInfos.size();
	//create_info.enabledExtensionCount = to_u32(enabled_extensions.size());
	//create_info.ppEnabledExtensionNames = enabled_extensions.data();

	//const auto requested_gpu_features = gpu.get_requested_features();
	//create_info.pEnabledFeatures = &requested_gpu_features;

	//VK_CHECK(vkCreateDevice(gpu.getHandle(), &create_info, nullptr, &handle));

    volkLoadDevice(handle);
}

VkDevice Device::getHandle() const
{
	return handle;
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