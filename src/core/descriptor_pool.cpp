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

#include "descriptor_pool.h"
#include "device.h"
#include "descriptor_set_layout.h"
#include "common/helpers.h"

namespace vulkr
{

DescriptorPool::DescriptorPool(Device &device, DescriptorSetLayout &descriptorSetLayout, std::vector<VkDescriptorPoolSize> &poolSizes, uint32_t maxSets) :
	device{ device },
	descriptorSetLayout{ descriptorSetLayout }
{
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;
    poolInfo.flags = 0; // We will not be checking if individual descriptor sets can be freed individually

    VK_CHECK(vkCreateDescriptorPool(device.getHandle(), &poolInfo, nullptr, &handle));
}

DescriptorPool::~DescriptorPool()
{
	vkDestroyDescriptorPool(device.getHandle(), handle, nullptr);
}

VkDescriptorPool DescriptorPool::getHandle() const
{
    return handle;
}

VkDescriptorSet DescriptorPool::allocate()
{
	VkDescriptorSetAllocateInfo descriptorSetInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	descriptorSetInfo.descriptorPool = handle;
	descriptorSetInfo.descriptorSetCount = 1u;
	descriptorSetInfo.pSetLayouts = &(descriptorSetLayout.getHandle());

	VkDescriptorSet descriptorSetHandle{ VK_NULL_HANDLE };

	// Allocate a new descriptor set from the current pool
	VK_CHECK(vkAllocateDescriptorSets(device.getHandle(), &descriptorSetInfo, &descriptorSetHandle));

	return descriptorSetHandle;
}

} // namespace vulkr