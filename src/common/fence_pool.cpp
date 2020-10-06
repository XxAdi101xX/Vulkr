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

#include "fence_pool.h"

namespace vulkr
{

FencePool::FencePool(Device &device) : device{ device }
{}

FencePool::~FencePool()
{
	wait();
	reset();

	for (VkFence fence : fences)
	{
		vkDestroyFence(device.getHandle(), fence, nullptr);
	}

	fences.clear();
}

VkFence FencePool::requestFence()
{
	if (activeFenceCount < fences.size())
	{
		return fences.at(activeFenceCount++);
	}

	VkFence fence{ VK_NULL_HANDLE };
	VkFenceCreateInfo fenceCreateInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

	VK_CHECK(vkCreateFence(device.getHandle(), &fenceCreateInfo, nullptr, &fence));

	fences.push_back(fence);
	++activeFenceCount;

	return fences.back();
}

void FencePool::wait(uint64_t timeout) const
{
	if (activeFenceCount == 0 || fences.empty())
	{
		return;
	}

	VK_CHECK(vkWaitForFences(device.getHandle(), activeFenceCount, fences.data(), true, timeout));
}

void FencePool::reset()
{
	if (activeFenceCount == 0 || fences.empty())
	{
		return;
	}

	VK_CHECK(vkResetFences(device.getHandle(), activeFenceCount, fences.data()));

	activeFenceCount = 0;
}

} // namespace vulkr