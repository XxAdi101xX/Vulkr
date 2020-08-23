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

#include "common/vk_common.h"

namespace vulkr
{

/* Forward declaration */
class Device;

class Queue
{
public:
	Queue(Device& device, uint32_t familyIndex, VkQueueFamilyProperties properties, bool canPresent, uint32_t index);
	Queue(Queue&&);

	/* Disable unnecessary operators to prevent error prone usages */
	Queue(const Queue&) = delete;
	Queue& operator=(const Queue&) = delete;
	Queue& operator=(Queue&&) = delete;

	/* Get the handle to the queue */
	VkQueue getHandle() const;

	/* Get the associating logical device for the queue */
	Device& getDevice() const;

	/* Get the family queue index */
	uint32_t getFamilyIndex() const;

	/* Get the queue index */
	uint32_t getIndex() const;

	VkQueueFamilyProperties getProperties() const;

	/* Returns whether the queue supports presentation */
	bool canSupportPresentation() const;

	/* Checks whether the queue supports all specified queue flags */
	bool supportsQueueFlags(VkQueueFlags desiredQueueFlags) const;

	/* TODO
	- add submit command
	- add present command
	- add a wait idle command?
	*/

private:
	/* The queue handle */
	VkQueue handle{ VK_NULL_HANDLE };

	/* The logical device that the queue is associated with */
	Device &device;

	/* The queue family index */
	uint32_t familyIndex{ 0u };

	/* The index of the queue within its queue family */
	uint32_t index{ 0u };

	/* Whether the queue supports presentation */
	bool canPresent{ false };

	/* The properties of the queue family that the queue is part of */
	VkQueueFamilyProperties properties{};
};

} // namespace vulkr
