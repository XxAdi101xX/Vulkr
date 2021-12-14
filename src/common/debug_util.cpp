/* Copyright (c) 2021 Adithya Venkatarao
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

#include "debug_util.h"

void setDebugUtilsObjectName(VkDevice device, const uint64_t object, const std::string &name, VkObjectType t)
{
#ifdef VULKR_DEBUG
	VkDebugUtilsObjectNameInfoEXT info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, t, object, name.c_str() };
	vkSetDebugUtilsObjectNameEXT(device, &info);
#endif
}

void setDebugUtilsObjectName(VkDevice device, VkBuffer object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_BUFFER); }
void setDebugUtilsObjectName(VkDevice device, VkBufferView object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_BUFFER_VIEW); }
void setDebugUtilsObjectName(VkDevice device, VkCommandBuffer object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_BUFFER); }
void setDebugUtilsObjectName(VkDevice device, VkCommandPool object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_POOL); }
void setDebugUtilsObjectName(VkDevice device, VkDescriptorPool object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_POOL); }
void setDebugUtilsObjectName(VkDevice device, VkDescriptorSet object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET); }
void setDebugUtilsObjectName(VkDevice device, VkDescriptorSetLayout object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT); }
void setDebugUtilsObjectName(VkDevice device, VkDeviceMemory object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DEVICE_MEMORY); }
void setDebugUtilsObjectName(VkDevice device, VkFramebuffer object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_FRAMEBUFFER); }
void setDebugUtilsObjectName(VkDevice device, VkImage object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_IMAGE); }
void setDebugUtilsObjectName(VkDevice device, VkImageView object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_IMAGE_VIEW); }
void setDebugUtilsObjectName(VkDevice device, VkPipeline object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE); }
void setDebugUtilsObjectName(VkDevice device, VkPipelineLayout object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE_LAYOUT); }
void setDebugUtilsObjectName(VkDevice device, VkQueryPool object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_QUERY_POOL); }
void setDebugUtilsObjectName(VkDevice device, VkQueue object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_QUEUE); }
void setDebugUtilsObjectName(VkDevice device, VkRenderPass object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_RENDER_PASS); }
void setDebugUtilsObjectName(VkDevice device, VkSampler object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SAMPLER); }
void setDebugUtilsObjectName(VkDevice device, VkSemaphore object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SEMAPHORE); }
void setDebugUtilsObjectName(VkDevice device, VkShaderModule object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SHADER_MODULE); }
void setDebugUtilsObjectName(VkDevice device, VkSwapchainKHR object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SWAPCHAIN_KHR); }
void setDebugUtilsObjectName(VkDevice device, VkAccelerationStructureKHR object, const std::string &name) { setDebugUtilsObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR); }

void debugUtilBeginLabel(VkCommandBuffer commandBuffer, const std::string &label)
{
#ifdef VULKR_DEBUG
    VkDebugUtilsLabelEXT debugUtilsLabel{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f} };
    vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &debugUtilsLabel);
#endif
}

void debugUtilEndLabel(VkCommandBuffer commandBuffer)
{
#ifdef VULKR_DEBUG
    vkCmdEndDebugUtilsLabelEXT(commandBuffer);
#endif
}

void debugUtilInsertLabel(VkCommandBuffer commandBuffer, const std::string &label)
{
#ifdef VULKR_DEBUG
    VkDebugUtilsLabelEXT debugUtilsLabel{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f} };
    vkCmdInsertDebugUtilsLabelEXT(commandBuffer, &debugUtilsLabel);
#endif
}