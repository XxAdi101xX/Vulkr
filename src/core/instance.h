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

#include <memory>
#include <string>
#include <vector>

#include "common/vk_common.h"
#include "physical_device.h"

#include <GLFW/glfw3.h>
namespace vulkr 
{

class Instance 
{
public:
	Instance();
	~Instance();
	
	/* Disable unnecessary operators to prevent error prone usages */
	Instance(const Instance &) = delete;
	Instance(Instance &&) = delete;
	Instance& operator=(const Instance &) = delete;
	Instance& operator=(Instance &&) = delete;

	/* Validation layer callback */
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData
	);

	/* Get the instance handle */
	VkInstance getHandle() const;

private:
	/* The Vulkan instance */
	VkInstance instance{ VK_NULL_HANDLE };

	/* The GPU used for the Vulkan application */
	std::unique_ptr<PhysicalDevice> gpu;

	/* The required validation layers */
	const std::vector<const char *> requiredValidationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

#ifdef VULKR_DEBUG
	/* Debug utils messenger callback for VK_EXT_Debug_Utils */
	VkDebugUtilsMessengerEXT debugUtilsMessenger{ VK_NULL_HANDLE };
#endif // VULKR_DEBUG

	/* Check if requiredValidationLayers are all available */
	bool checkValidationLayerSupport() const;

	/* Get all required instances */
	std::vector<const char*> getRequiredInstanceExtensions() const;

	/* Select a physical device for our application; will populate the gpu field */
	void selectGPU();
};

} // namespace vulkr