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

#include <array>

#include "common/vulkan_common.h"

namespace vulkr
{

class RenderPass;
class PipelineLayout;

/**
 * @brief Describes the vertex data that will be passed to the vertex shader with two pieces of information.
 * @param bindingDescriptions: specifies the spacing between data and whether the data is per-vertex or per-instance
 * @param attributeDescriptions: type of the attributes passed to the vertex shader, which binding to load them from and at which offset
 */
struct VertexInputState
{
	std::vector<VkVertexInputBindingDescription> bindingDescriptions;
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
};

/* Describes the kind of geometry will be drawn from the vertices and whether primitive restart should be enabled. */
struct InputAssemblyState
{
	VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
	VkBool32 primitiveRestartEnable{ VK_FALSE };
};

/* Describes the region of the framebuffer that the output will be rendered to. Having more than one viewport or scissors requires a feature flag to be enabled */
struct ViewportState
{
	std::vector<VkViewport> viewports;
	std::vector<VkRect2D> scissors;
};

/** 
 * @brief Takes the geometry that is shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader.
 * It also contains information depth testing, face culling and the scissor test.
 */
struct RasterizationState
{
	VkBool32 depthClampEnable{ VK_FALSE };
	VkBool32 rasterizerDiscardEnable{ VK_FALSE };
	VkPolygonMode polygonMode{ VK_POLYGON_MODE_FILL };
	VkCullModeFlags cullMode{ VK_CULL_MODE_BACK_BIT };
	VkFrontFace frontFace{ VK_FRONT_FACE_COUNTER_CLOCKWISE };
	float lineWidth{ 1.0 };
	VkBool32 depthBiasEnable{ VK_FALSE };
	float depthBiasConstantFactor{ 1.0f }; /* required if depthBiasEnable is true */
	float depthBiasClamp{ 1.0f }; /* required if depthBiasEnable is true */
	float depthBiasSlopeFactor{ 1.0f }; /* required if depthBiasEnable is true */
};

/* Describes all multisampling information; is usually used to perform anti-aliasing */
struct MultisampleState
{
	VkSampleCountFlagBits rasterizationSamples{ VK_SAMPLE_COUNT_1_BIT };
	VkBool32 sampleShadingEnable{ VK_FALSE };
	float min_sample_shading{ 0.0f };
	VkSampleMask *sampleMask{ nullptr };
	VkBool32 alphaToCoverageEnable{ VK_FALSE };
	VkBool32 alphaToOneEnable{ VK_FALSE };
};

/* Used in the DepthStencilState */
struct StencilOpState
{
	VkStencilOp failOp{ VK_STENCIL_OP_REPLACE };
	VkStencilOp passOp{ VK_STENCIL_OP_REPLACE };
	VkStencilOp depthFailOp{ VK_STENCIL_OP_REPLACE };
	VkCompareOp compareOp{ VK_COMPARE_OP_NEVER };
};

/* Configures all necessary state for a depth and/or stencil buffer */
struct DepthStencilState
{
	VkBool32 depthTestEnable{ VK_TRUE };
	VkBool32 depthWriteEnable{ VK_TRUE };
	// Note: Using Reversed depth-buffer for increased precision, so Greater depth values are kept
	VkCompareOp depthCompareOp{ VK_COMPARE_OP_GREATER };
	VkBool32 depthBoundsTestEnable{ VK_FALSE };
	VkBool32 stencilTestEnable{ VK_FALSE };
	StencilOpState front{}; // optional if stencilTestEnabled is false
	StencilOpState back{}; // optional if stencilTestEnabled is false
	float minDepthBounds{ 0.0f };
	float maxDepthBounds{ 1.0f } ;
};

/* 
 * @brief After a fragment shader has returned a color, it needs to be combined with the color that is already in the framebuffer.
 * ColorBlendAttachmentState contain the configuration per attached frame buffer
 */
struct ColorBlendAttachmentState
{
	VkBool32 blendEnable{ VK_FALSE };
	VkBlendFactor srcColorBlendFactor{ VK_BLEND_FACTOR_ONE };
	VkBlendFactor dstColorBlendFactor{ VK_BLEND_FACTOR_ZERO };
	VkBlendOp colorBlendOp{ VK_BLEND_OP_ADD };
	VkBlendFactor srcAlphaBlendFactor{ VK_BLEND_FACTOR_ONE };
	VkBlendFactor dstAlphaBlendFactor{ VK_BLEND_FACTOR_ZERO };
	VkBlendOp alphaBlendOp{ VK_BLEND_OP_ADD };
	VkColorComponentFlags colorWriteMask{ VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
};

/* Has the same purpose as explained above for ColorBlendAttachmentState but this struct contains the global color belnding setting */
struct ColorBlendState
{
	VkBool32 logicOpEnable{ VK_FALSE };
	VkLogicOp logicOp{ VK_LOGIC_OP_CLEAR };
	std::vector<ColorBlendAttachmentState> attachments;
	float blendConstants[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

// TODO: create specialization constant class

class PipelineState
{
public:
	PipelineState (
		std::unique_ptr<PipelineLayout> &&pipelineLayout,
		RenderPass& renderPass,
		VertexInputState vertexInputState,
		InputAssemblyState inputAssemblyState,
		ViewportState viewportState,
		RasterizationState rasterizationState,
		MultisampleState multisampleState,
		DepthStencilState depthStencilState,
		ColorBlendState colorBlendState
	);
	~PipelineState() = default;

	/* Disable unnecessary operators to prevent error prone usages */
	PipelineState(const PipelineState &) = delete;
	PipelineState(PipelineState &&) = delete;
	PipelineState& operator=(const PipelineState &) = delete;
	PipelineState& operator=(PipelineState &&) = delete;

	const PipelineLayout &getPipelineLayout() const;

	const RenderPass &getRenderPass() const;

	const VertexInputState &getVertexInputState() const;

	const InputAssemblyState &getInputAssemblyState() const;

	const RasterizationState &getRasterizationState() const;

	const ViewportState &getViewportState() const;

	const MultisampleState &getMultisampleState() const;

	const DepthStencilState &getDepthStencilState() const;

	const ColorBlendState &getColorBlendState() const;

	const std::array<VkDynamicState, 9> &getDyanmicStates() const;

	uint32_t getSubpassIndex() const;


private:
	std::unique_ptr<PipelineLayout> pipelineLayout{ nullptr };

	RenderPass &renderPass;

	VertexInputState vertexInputState{};

	InputAssemblyState inputAssemblyState{};

	ViewportState viewportState{};

	RasterizationState rasterizationState{};

	MultisampleState multisampleState{};

	DepthStencilState depthStencilState{};

	ColorBlendState colorBlendState{};

	std::array<VkDynamicState, 9> dynamicStates{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_LINE_WIDTH,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
		VK_DYNAMIC_STATE_BLEND_CONSTANTS,
		VK_DYNAMIC_STATE_DEPTH_BOUNDS,
		VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
		VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
		VK_DYNAMIC_STATE_STENCIL_REFERENCE
	};

	uint32_t subpassIndex{ 0u }; // TODO do we need this, currently unused
};

} // namespace vulkr