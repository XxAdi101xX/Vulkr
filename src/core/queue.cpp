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

#include "queue.h"
#include "device.h"

namespace vulkr
{

Queue::Queue(Device &device, uint32_t familyIndex, VkQueueFamilyProperties properties, bool canPresent, uint32_t index) :
	device{ device },
	familyIndex{ familyIndex },
	index{ index },
	canPresent{ canPresent },
	properties{ properties }
{
	vkGetDeviceQueue(device.getHandle(), familyIndex, index, &handle);
}

Queue::Queue(Queue &&other) :
	handle{ other.handle },
	device{ other.device },
	familyIndex{ other.familyIndex },
	index{ other.index },
	canPresent{ other.canPresent },
	properties{ other.properties }
{
	other.handle = VK_NULL_HANDLE;
	other.familyIndex = {};
	other.properties = {};
	other.canPresent = VK_FALSE;
	other.index = 0;

	LOGI("Queue's move constructor has been called!");
}

Device& Queue::getDevice() const
{
	return device;
}

VkQueue Queue::getHandle() const
{
	return handle;
}

uint32_t Queue::getFamilyIndex() const
{
	return familyIndex;
}

uint32_t Queue::getIndex() const
{
	return index;
}

VkQueueFamilyProperties Queue::getProperties() const
{
	return properties;
}

bool Queue::canSupportPresentation() const
{
	return canPresent;
}

bool Queue::supportsQueueFlags(VkQueueFlags desiredQueueFlags) const
{
	return (properties.queueFlags & desiredQueueFlags) == desiredQueueFlags;
}

} // namespace vulkr