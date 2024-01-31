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

class Device;

class Buffer
{
public:
	Buffer(Device &device, VkBufferCreateInfo bufferInfo, VmaAllocationCreateInfo memoryInfo);
	~Buffer();

	Buffer(Buffer &&other);

	Buffer(const Buffer &) = delete;
	Buffer &operator=(const Buffer &) = delete;
	Buffer &operator=(Buffer &&) = delete;

	const Device &getDevice() const;

	VkBuffer getHandle() const;

	VmaAllocation getAllocation() const;

	VkDeviceMemory getMemory() const;

	/* Flushes memory if it is HOST_VISIBLE and not HOST_COHERENT */
	void flush() const;

	/**
	 * Maps vulkan memory if it isn't already mapped to an host visible address
	 * @return Pointer to host visible memory
	 */
	void *map();

	/* Unmaps vulkan memory from the host visible address */
	void unmap();

	/* Gets the size of the buffer */
	VkDeviceSize getSize() const;

	/**
		* Copies byte data into the buffer
		* @param data The data to copy from
		* @param dataSize The amount of bytes to copy
		* @param offset The offset to start the copying into the mapped data
		*/
	void update(const uint8_t *data, size_t dataSize, size_t offset = 0);
	// TODO: add update method that takes a pointer to a generic array of type T

	/**
		* Converts any non byte data into bytes and then updates the buffer
		* @param data The data to copy from
		* @param dataSize The amount of bytes to copy
		* @param offset The offset to start the copying into the mapped data
		*/
	void update(void *data, size_t dataSize, size_t offset = 0);

	/**
		* Copies a vector of bytes into the buffer
		* @param data The data vector to upload
		* @param offset The offset to start the copying into the mapped data
		*/
	void update(const std::vector<uint8_t> &data, size_t offset = 0);

	/**
		* Copies an object as byte data into the buffer
		* @param object The object to convert into byte data
		* @param offset The offset to start the copying into the mapped data
		*/
	template <class T>
	void convert_and_update(const T &object, size_t offset = 0)
	{
		update(reinterpret_cast<const uint8_t *>(&object), sizeof(T), offset);
	}

private:
	Device &device;

	VkBuffer handle{ VK_NULL_HANDLE };

	VmaAllocation allocation{ VK_NULL_HANDLE };

	VmaAllocationInfo allocationInfo{};

	VkDeviceSize size{ 0 };

	// Whether the buffer is persistently mapped or not
	bool persistent{ false };

	// Whether the buffer has been mapped with vmaMapMemory
	bool mapped{ false };
	uint8_t *mappedData{ nullptr };
};

} // namespace vulkr