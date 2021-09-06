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

#pragma once

#include "vulkan_common.h"

/* Set the object name for a specified object type; used for easier debugging */
void setDebugUtilsObjectName(VkDevice device, const uint64_t object, const std::string &name, VkObjectType t);

/* Overloading the setDebugUtilsObjectName to pass in the VkObjectType based on the object type passed in */
void setDebugUtilsObjectName(VkDevice device, VkBuffer object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkBufferView object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkCommandBuffer object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkCommandPool object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkDescriptorPool object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkDescriptorSet object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkDescriptorSetLayout object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkDeviceMemory object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkFramebuffer object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkImage object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkImageView object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkPipeline object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkPipelineLayout object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkQueryPool object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkQueue object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkRenderPass object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkSampler object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkSemaphore object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkShaderModule object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkSwapchainKHR object, const std::string &name);
void setDebugUtilsObjectName(VkDevice device, VkAccelerationStructureKHR object, const std::string &name);

/* Set the debug utils begin label; must be associated with an end label */
void debugUtilBeginLabel(VkCommandBuffer commandBuffer, const std::string &label);

/* Set a debug until end label to match a begin label */
void debugUtilEndLabel(VkCommandBuffer commandBuffer);

/* Insert a debug utils label */
void debugUtiliInsertLabel(VkCommandBuffer commandBuffer, const std::string &label);