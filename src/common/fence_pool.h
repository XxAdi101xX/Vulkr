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

#include "vulkan_common.h"
#include "core/device.h"

namespace vulkr
{

class FencePool
{
public:
	FencePool(Device& device);
	~FencePool();

	FencePool(const FencePool &) = delete;
	FencePool(FencePool &&) = delete;
	FencePool& operator=(const FencePool &) = delete;
	FencePool& operator=(FencePool &&) = delete;

	VkFence requestFence();

	void wait(VkFence *fence, uint64_t timeout = std::numeric_limits<uint64_t>::max()) const;

	void reset(VkFence *fence);

	void waitAll(uint64_t timeout = std::numeric_limits<uint64_t>::max()) const;

	void resetAll();
private:
	Device &device;

	std::vector<VkFence> fences;

	uint32_t activeFenceCount{ 0 };
};

} // namespace vulkr