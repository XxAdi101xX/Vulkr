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

#include "buffer.h"
#include "device.h"

namespace vulkr
{

Buffer::Buffer(Device &device, VkBufferCreateInfo bufferInfo, VmaAllocationCreateInfo memoryInfo) :
	device{ device },
	size{ bufferInfo.size }
{
	// Creates the buffer, allocates the appropriate memory to it and binds the buffer with the memory
	VK_CHECK(vmaCreateBuffer(device.getMemoryAllocator(), &bufferInfo, &memoryInfo, &handle, &allocation, &allocationInfo));

	persistent = (memoryInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;
	if (persistent)
	{
		mappedData = static_cast<uint8_t*>(allocationInfo.pMappedData);
	}
}

Buffer::Buffer(Buffer &&other) :
	device{ other.device },
	handle{ other.handle },
	allocation{ other.allocation },
	allocationInfo{ other.allocationInfo },
	size{ other.size },
	mappedData{ other.mappedData },
	mapped{ other.mapped }
{
	other.handle = VK_NULL_HANDLE;
	other.allocation = VK_NULL_HANDLE;
	other.allocationInfo = {};
	other.mappedData = nullptr;
	other.mapped = false;
}

Buffer::~Buffer()
{
	if (mapped)
	{
		LOGW("Mapped data has not be explicity unmapped, will be unmapped in destructor..");
		unmap();
	}

	if (handle != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
	{
		vmaDestroyBuffer(device.getMemoryAllocator(), handle, allocation);
	}
}

const Device &Buffer::getDevice() const
{
	return device;
}

VkBuffer Buffer::getHandle() const
{
	return handle;
}

VmaAllocation Buffer::getAllocation() const
{
	return allocation;
}

VkDeviceMemory Buffer::getMemory() const
{
	return allocationInfo.deviceMemory;
}

VkDeviceSize Buffer::getSize() const
{
	return size;
}

void *Buffer::map()
{
	if (!mapped)
	{
		VK_CHECK(vmaMapMemory(device.getMemoryAllocator(), allocation, reinterpret_cast<void **>(&mappedData)));
		mapped = true;
	}
	return mappedData;
}

void Buffer::unmap()
{
	if (!mapped)
	{
		LOGEANDABORT("Trying to unmap memory on a buffer that's not mapped");
	}
	vmaUnmapMemory(device.getMemoryAllocator(), allocation);
	mappedData = nullptr;
	mapped = false;

}

void Buffer::flush() const
{
	vmaFlushAllocation(device.getMemoryAllocator(), allocation, 0, size);
}

void Buffer::update(const std::vector<uint8_t>& data, size_t offset)
{
	update(data.data(), data.size(), offset);
}

void Buffer::update(void* data, size_t size, size_t offset)
{
	update(reinterpret_cast<const uint8_t*>(data), size, offset);
}

void Buffer::update(const uint8_t *data, const size_t size, const size_t offset)
{
	// TODO fix this
	//if (persistent)
	//{
	//	std::copy(data, data + size, mappedData + offset);
	//	flush();
	//}
	//else
	//{
	//	map();
	//	std::copy(data, data + size, mappedData + offset);
	//	flush();
	//	unmap();
	//}
}

} // namespace vulkr