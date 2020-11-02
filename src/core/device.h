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

#include "common/vulkan_common.h"

namespace vulkr
{

class PhysicalDevice;
class Queue;

class Device
{
public:
	Device(std::unique_ptr<PhysicalDevice> &&physicalDevice, VkSurfaceKHR surface, const std::vector<const char*> requestedExtensions = {});
	~Device();

	/* Disable unnecessary operators to prevent error prone usages */
	Device(const Device &) = delete;
	Device(Device &&) = delete;
	Device& operator=(const Device &) = delete;
	Device& operator=(Device &&) = delete;

	/* Wait for a device to be fully idle, is the equivilant of calling vkQueueWaitIdle on all the queues owned by the device */
	void Device::waitIdle() const;

	/* Get the logical device handle */
	VkDevice getHandle() const;

	/* Get the physical device handle */
	const PhysicalDevice &getPhysicalDevice() const;

	/* Get a graphics queue with present support if available, else just grab the first available graphics queue */
	const Queue &getOptimalGraphicsQueue();

	/* Get a queue with the desired queue flags */
	const Queue &getQueueByFlags(VkQueueFlags desiredQueueFlags);

	/* Get the first available queue that supports presentation. This is only called when the graphics queue does not support presentation */
	const Queue &getQueueByPresentation();

	/* Get the memory allocator */
	VmaAllocator getMemoryAllocator() const;

	/* Get the memory type for the specified memoryPropertyFlags */
	uint32_t getMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags propertieFlags);
private:
	/* The logical device handle */
	VkDevice handle{ VK_NULL_HANDLE };
	
	/* The gpu used */
	std::unique_ptr<PhysicalDevice> physicalDevice;

	/* All the extensions available on the device */
	std::vector<VkExtensionProperties> deviceExtensions;

	/* The extensions that we specifically want to use */
	std::vector<const char*> enabledExtensions;

	/* All the queues available on our gpu */
	std::vector<std::vector<Queue>> queues;

	/* Check if a specified extension is supported */
	bool isExtensionSupported(const char *extension) const;

	/* Check if an extension is enabled */
	bool isExtensionEnabled(const char* extension) const;

	/* The memory allocator */
	VmaAllocator memoryAllocator{ VK_NULL_HANDLE };

	/* TODO
	- Dedicated transfer queue is very under utilized so it can be used to defragment memeory, streaming resources or textures
	- Add the command pool and the fence pool?
	- Add a resource cache if necessary
	*/
};

} // namespace vulkr
