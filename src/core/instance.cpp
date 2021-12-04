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

#include "instance.h"
#include "physical_device.h"

#include <array>

#include "common/logger.h"
#include "common/helpers.h"

namespace vulkr
{

Instance::Instance(std::string applicationName)
{
    VK_CHECK(volkInitialize());

#ifdef VULKR_DEBUG
    if (!checkValidationLayerSupport())
    {
        LOGEANDABORT("Validation layers are not available when requested!");
    }
#endif // VULKR_DEBUG

    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO  };
    appInfo.pApplicationName = applicationName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Vulkr";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceCreateInfo.pApplicationInfo = &appInfo;

    std::vector<const char *> extensions = getRequiredInstanceExtensions();
    instanceCreateInfo.enabledExtensionCount = to_u32(extensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

#ifdef VULKR_DEBUG
    instanceCreateInfo.enabledLayerCount = to_u32(requiredValidationLayers.size());
    instanceCreateInfo.ppEnabledLayerNames = requiredValidationLayers.data();

    std::array<VkValidationFeatureEnableEXT, 3> validationFeatureToEnable {
        // VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT and VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT can not be enabled at the same time
#if 1
        VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
#else
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
#endif
        VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };
    VkValidationFeaturesEXT validationFeatures{ VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
    validationFeatures.enabledValidationFeatureCount = to_u32(validationFeatureToEnable.size());
    validationFeatures.pEnabledValidationFeatures = validationFeatureToEnable.data();

    VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    debugUtilsCreateInfo.messageSeverity = /*VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | */VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugUtilsCreateInfo.pfnUserCallback = debugUtilsMessengerCallback;
    debugUtilsCreateInfo.pNext = &validationFeatures;

    instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugUtilsCreateInfo;
#else
    instanceCreateInfo.enabledLayerCount = 0;
    instanceCreateInfo.pNext = nullptr;
#endif // VULKR_DEBUG

    VK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

    volkLoadInstance(instance);

#ifdef VULKR_DEBUG
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsCreateInfo, nullptr, &debugUtilsMessenger));
    LOGD("Validation layer enabled");
#endif // VULKR_DEBUG
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

    for (const char *layerName : requiredValidationLayers)
    {
        bool layerFound = false;

        for (const auto &layerProperties : availableLayers)
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

std::vector<const char *> Instance::getRequiredInstanceExtensions() const
{
    uint32_t glfwExtensionCount{ 0u };
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef VULKR_DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif // VULKR_DEBUG

    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Instance::debugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    // messageIdName could be null but the messageIdNumber and Message can not
    const char *messageIdName = pCallbackData->pMessageIdName;
    if (pCallbackData->pMessageIdName == nullptr)
    {
        messageIdName =  "";
    }

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        LOGI("{} - {}: {}", pCallbackData->messageIdNumber, messageIdName, pCallbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        LOGW("{} - {}: {}", pCallbackData->messageIdNumber, messageIdName, pCallbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        LOGE("{} - {}: {}", pCallbackData->messageIdNumber,messageIdName, pCallbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
    {
        // TODO: Log messages that are triggered after the cleanup process begin to fail since it seems like the spdlog context cleanup process has already begun
        std::cerr << pCallbackData->messageIdNumber << " - " << messageIdName << ": " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

std::unique_ptr<PhysicalDevice> Instance::getSuitablePhysicalDevice()
{
    // Querying valid physical devices on the machine
    uint32_t physicalDeviceCount{ 0u };
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr));

    if (physicalDeviceCount < 1)
    {
        LOGEANDABORT("Instance::selectGPU: Couldn't find a physical device that supports Vulkan.");
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
            return std::make_unique<PhysicalDevice>(*this, physicalDevice);
        }
    }

    // If a discrete GPU isn't found, we default to the first available one
    LOGW("A discrete GPU wasn't found hence the first available one was chosen");
    return std::make_unique<PhysicalDevice>(*this, physicalDevices[0]);
}

} // namespace vulkr