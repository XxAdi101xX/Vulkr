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

#include "common/logger.h"
#include "instance.h"

namespace vulkr
{

Instance::Instance()
{
#ifdef VULKR_DEBUG
    if (!checkValidationLayerSupport())
    {
        throw std::runtime_error("Validation layers are not available when requested!");
    }
#endif // VULKR_DEBUG

    VK_CHECK(volkInitialize());

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan App";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Vulkr";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    std::vector<const char*> extensions = getRequiredInstanceExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo;
#ifdef VULKR_DEBUG
    createInfo.enabledLayerCount = static_cast<uint32_t>(requiredValidationLayers.size());
    createInfo.ppEnabledLayerNames = requiredValidationLayers.data();

    debugUtilsCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugUtilsCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugUtilsCreateInfo.pfnUserCallback = debugUtilsMessengerCallback;

    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugUtilsCreateInfo;
#else
    createInfo.enabledLayerCount = 0;
    createInfo.pNext = nullptr;
#endif // VULKR_DEBUG

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

    volkLoadInstance(instance);

#ifdef VULKR_DEBUG
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsCreateInfo, nullptr, &debugUtilsMessenger));
    LOGI("Validation layer enabled");
#endif // VULKR_DEBUG

    selectGPU();
}

Instance::~Instance() 
{
#ifdef VULKR_DEBUG
vkDestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
#endif // VULKR_DEBUG

    vkDestroyInstance(instance, nullptr);
}

VkInstance Instance::getHandle() const {
    return instance;
}

bool Instance::checkValidationLayerSupport() const
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : requiredValidationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) 
        {
            if (strcmp(layerName, layerProperties.layerName) == 0) 
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char*> Instance::getRequiredInstanceExtensions() const
{
    uint32_t glfwExtensionCount{ 0 };
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef VULKR_DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif // VULKR_DEBUG

    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Instance::debugUtilsMessengerCallback (
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) 
{
    std::cerr << pCallbackData->messageIdNumber << " - " << pCallbackData->pMessageIdName << ": " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

void Instance::selectGPU()
{
    // Querying valid physical devices on the machine
    uint32_t physicalDeviceCount{ 0 };
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr));

    if (physicalDeviceCount < 1)
    {
        throw std::runtime_error("Instance::selectGPU: Couldn't find a physical device that supports Vulkan.");
    }

    std::vector<VkPhysicalDevice> physicalDevices;
    physicalDevices.resize(physicalDeviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()));


    for (const VkPhysicalDevice &physicalDevice : physicalDevices)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            gpu = std::make_unique<PhysicalDevice>(*this, physicalDevice);
            LOGI("Selected GPU: {}", gpu->getProperties().deviceName);
            return;
        }
    }

    // If a discrete GPU isn't found, we default to the first available one
    gpu = std::make_unique<PhysicalDevice>(*this, physicalDevices[0]);
    LOGW("A discrete GPU wasn't found hence we will default to using {}", gpu->getProperties().deviceName);
}

} // namespace vulkr