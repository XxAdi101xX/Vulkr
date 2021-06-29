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

#include <iostream>
#include <cstdlib>

#include <vk_mem_alloc.h>
#include <volk.h>

#include "logger.h"

#define VULKR_DEBUG /* Enable the validation layers */

/* @brief Assert whether an VkResult has returned an error */
#define VK_CHECK(r)                                                  \
	do                                                               \
	{                                                                \
		VkResult result = r;                                         \
		if (result != VK_SUCCESS)                                    \
		{                                                            \
			LOGEANDABORT("Vulkan Error: {}", printVkResult(result)); \
		}                                                            \
	} while (0);

/* Printing out the VkResult enum name */
const char *printVkResult(const VkResult result);

/* Determine whether a format ONLY has the depth component available */
bool isDepthOnlyFormat(VkFormat format);

/* Determine whether a format has both the depth AND stencil components available */
bool isDepthStencilFormat(VkFormat format);

/* Determine a suitable supported depth format */
VkFormat getSupportedDepthFormat(
	VkPhysicalDevice  physicalDeviceHandle,
	bool depthOnly = false,
	const std::vector<VkFormat> &formatPriorityList = {
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D24_UNORM_S8_UINT
	}
);