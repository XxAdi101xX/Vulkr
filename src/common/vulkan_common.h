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

#include "logger.h"

#ifdef _DEBUG // Visual Studio defines this for debug builds and NDEBUG for non-debug builds
	#define VULKR_DEBUG /* Enable the validation layers and all debugging messages */
#endif

/* Route VMA's internal assertions through the logger. A plain assert() writes straight to stderr and carries no tag, so
   a fatal VMA check is easy to miss in the middle of the log; this way it is tagged as an error and reports the file and
   line inside vk_mem_alloc.h */
#define VMA_VULKAN_VERSION 1004000
#ifdef VULKR_DEBUG
	#define VMA_ASSERT(expr) VULKR_ASSERT_IMPL(expr)
	#define VMA_ASSERT_LEAK(expr) VULKR_ASSERT_IMPL(expr)
#endif
#include <volk.h>
#include <vk_mem_alloc.h>

/* The Vulkan API version the engine targets: the instance requests it and the shader toolchain derives its SPIR-V
   revision from it (Vulkan 1.4 still caps at SPIR-V 1.6) */
inline constexpr uint32_t vulkanApiVersion{ VK_API_VERSION_1_4 };

 //#define RENDERDOC_DEBUG /* Disable extensions unsupported by RenderDoc (like raytracing) in order to use it to debug this application */

 /* Assert whether an VkResult has returned an error */
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

/* Check the VkResult and print out if it's in a error state similar function to VK_CHECK, used for imgui's CheckVkResultFn callback function */
void checkVkResult(VkResult result);

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

/* Get the SPIR-V profile name that shader compilers use for the given Vulkan API version. The revisions follow the API
   version: 1.0 -> spirv_1_0, 1.1 -> spirv_1_3, 1.2 -> spirv_1_5 and 1.3 or newer -> spirv_1_6, because Vulkan 1.3
   raised the limit to SPIR-V 1.6 and Vulkan 1.4 did not raise it further */
const char *getSpirvProfileName(uint32_t apiVersion);