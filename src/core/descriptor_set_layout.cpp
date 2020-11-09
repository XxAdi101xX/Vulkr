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

#include "descriptor_set_layout.h"
#include "device.h"

#include "common/helpers.h"

namespace vulkr
{

DescriptorSetLayout::DescriptorSetLayout(Device &device, const std::vector<VkDescriptorSetLayoutBinding> bindings) :
	device{ device },
	bindings{ std::move(bindings)}
{
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	descriptorSetLayoutInfo.bindingCount = to_u32(this->bindings.size());
	descriptorSetLayoutInfo.pBindings = this->bindings.data();

	VK_CHECK(vkCreateDescriptorSetLayout(device.getHandle(), &descriptorSetLayoutInfo, nullptr, &handle));
}

DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout &&other) :
	device{ other.device },
	handle{ other.handle },
	bindings{ std::move(other.bindings) }
{
	other.handle = VK_NULL_HANDLE;
}

DescriptorSetLayout::~DescriptorSetLayout()
{
	vkDestroyDescriptorSetLayout(device.getHandle(), handle, nullptr);
}

const VkDescriptorSetLayout &DescriptorSetLayout::getHandle() const
{
	return handle;
}

const std::vector<VkDescriptorSetLayoutBinding> &DescriptorSetLayout::getBindings() const
{
	return bindings;
}

} // namespace vulkr