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

#include "vulkan_common.h"

// TODO: change this from overloading the ostream and make it work with the LOG* functions
std::ostream& operator<<(std::ostream& os, const VkResult result)
{
#define PRINT_VK_RESULT_ENUM_NAME(r) \
	case r:                          \
		os << #r;                    \
		break;

	switch (result) 
	{
		PRINT_VK_RESULT_ENUM_NAME(VK_SUCCESS);
		PRINT_VK_RESULT_ENUM_NAME(VK_NOT_READY);
		PRINT_VK_RESULT_ENUM_NAME(VK_TIMEOUT);
		PRINT_VK_RESULT_ENUM_NAME(VK_EVENT_SET);
		PRINT_VK_RESULT_ENUM_NAME(VK_EVENT_RESET);
		PRINT_VK_RESULT_ENUM_NAME(VK_INCOMPLETE);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_OUT_OF_HOST_MEMORY);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_OUT_OF_DEVICE_MEMORY);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INITIALIZATION_FAILED);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_DEVICE_LOST);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_MEMORY_MAP_FAILED);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_LAYER_NOT_PRESENT);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_EXTENSION_NOT_PRESENT);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_FEATURE_NOT_PRESENT);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INCOMPATIBLE_DRIVER);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_TOO_MANY_OBJECTS);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_FORMAT_NOT_SUPPORTED);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_FRAGMENTED_POOL);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_UNKNOWN);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_OUT_OF_POOL_MEMORY);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INVALID_EXTERNAL_HANDLE);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_FRAGMENTATION);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_SURFACE_LOST_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_SUBOPTIMAL_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_OUT_OF_DATE_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_VALIDATION_FAILED_EXT);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INVALID_SHADER_NV);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INCOMPATIBLE_VERSION_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_NOT_PERMITTED_EXT);
		PRINT_VK_RESULT_ENUM_NAME(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
		PRINT_VK_RESULT_ENUM_NAME(VK_THREAD_IDLE_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_THREAD_DONE_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_OPERATION_DEFERRED_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_OPERATION_NOT_DEFERRED_KHR);
		PRINT_VK_RESULT_ENUM_NAME(VK_PIPELINE_COMPILE_REQUIRED_EXT);
		PRINT_VK_RESULT_ENUM_NAME(VK_RESULT_MAX_ENUM);
	default:
		os << "UNKNOWN_ERROR with code " << result;
	}

#undef PRINT_VK_RESULT_ENUM_NAME

	return os;
}

// Note that we don't use 16 bit float for depth, only 24 or 32
bool isDepthOnlyFormat(VkFormat format)
{
	return format == VK_FORMAT_D32_SFLOAT;
}

bool isDepthStencilFormat(VkFormat format)
{
	return format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

VkFormat getSupportedDepthFormat(VkPhysicalDevice physicalDeviceHandle, bool depthOnly, const std::vector<VkFormat> &formatPriorityList)
{
	for (auto& format : formatPriorityList)
	{
		if (depthOnly && !isDepthOnlyFormat(format))
		{
			continue;
		}

		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(physicalDeviceHandle, format, &properties);

		// Format must support depth stencil attachment for optimal tiling
		if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			return format;
		}
	}

	throw std::runtime_error("Failed to find a supported format");
}