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
class ShaderModule;
//class DescriptorSetLayout;

class PipelineLayout
{
public:
	PipelineLayout(Device &device, const std::vector<ShaderModule> &shaderModules);
	~PipelineLayout();


	/* Disable unnecessary operators to prevent error prone usages */
	PipelineLayout(const PipelineLayout &) = delete;
	PipelineLayout(PipelineLayout &&) = delete;
	PipelineLayout& operator=(const PipelineLayout &) = delete;
	PipelineLayout& operator=(PipelineLayout &&) = delete;


	VkPipelineLayout getHandle() const;

	const std::vector<VkDescriptorSetLayout*> &getSetLayouts() const;
	const std::vector<VkPushConstantRange*> &getPushConstantRanges() const;
	// TODO

	const std::vector<ShaderModule> &getShaderModules() const;

	//const std::unordered_map<uint32_t, std::vector<ShaderResource>>& get_bindings() const;

	//const std::vector<ShaderResource>& get_set_bindings(uint32_t set_index) const;

	//bool has_set_layout(uint32_t set_index) const;

	//DescriptorSetLayout& get_set_layout(uint32_t set_index);

	//std::vector<ShaderResource> get_vertex_input_attributes() const;

	//std::vector<ShaderResource> get_fragment_output_attachments() const;

	//std::vector<ShaderResource> get_fragment_input_attachments() const;

	//VkShaderStageFlags get_push_constant_range_stage(uint32_t offset, uint32_t size) const;

private:
	VkPipelineLayout handle{ VK_NULL_HANDLE };
	Device& device;

	const std::vector<ShaderModule> &shaderModules;

	std::vector<VkDescriptorSetLayout*> setLayouts;
	std::vector<VkPushConstantRange*> pushConstantRanges;

	//std::map<std::string, shaderresource> resources;

	//std::unordered_map<uint32_t, std::vector<shaderresource>> set_bindings;

	//std::unordered_map<uint32_t, descriptorsetlayout*> set_layouts;
};

} // namespace vulkr