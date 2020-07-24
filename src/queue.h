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

#include "vk_common.h"

namespace vulkr
{

class Device;

class Queue
{
public:
	Queue(Device& device, uint32_t familyIndex, VkQueueFamilyProperties properties, bool canPresent, uint32_t index);

	Queue(const Queue&) = delete; // TODO should this be default
	Queue(Queue &&) = delete; // TODO should this be overriden
	Queue& operator=(const Queue&) = delete;
	Queue& operator=(Queue&&) = delete;

	const Device& getDevice() const;

	VkQueue getHandle() const;

	uint32_t getFamilyIndex() const;

	uint32_t getIndex() const;

	VkQueueFamilyProperties getProperties() const;

	bool supportPresent() const;

	//VkResult submit(const std::vector<VkSubmitInfo> & submit_infos, VkFence fence) const;

	//VkResult submit(const CommandBuffer & command_buffer, VkFence fence) const;

	//VkResult present(const VkPresentInfoKHR & present_infos) const;

	VkResult waitIdle() const;

private:
	Device& device;

	VkQueue handle{ VK_NULL_HANDLE };

	uint32_t familyIndex{ 0 };

	uint32_t index{ 0 };

	bool canPresent{ false };

	VkQueueFamilyProperties properties{};
};

} // namespace vulkr
