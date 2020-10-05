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

#include "common/vk_common.h"
#include "device.h"
#include "render_pass.h"
#include "framebuffer.h"

namespace vulkr
{

class CommandPool;

class CommandBuffer
{
public:
	enum class State
	{
		Invalid,
		Initial,
		Recording,
		Executable,
	};

	CommandBuffer(CommandPool &commandPool, VkCommandBufferLevel level, RenderPass &renderPass, Framebuffer &framebuffer);
	~CommandBuffer();

	/* Disable unnecessary operators to prevent error prone usages */
	CommandBuffer(const CommandBuffer &) = delete;
	CommandBuffer(CommandBuffer &&) = delete;
	CommandBuffer& operator=(const CommandBuffer &) = delete;
	CommandBuffer& operator=(CommandBuffer &&) = delete;

	Device &getDevice();

	const VkCommandBuffer &getHandle() const;

	const VkCommandBufferLevel getLevel() const;

	bool isRecording() const;

	void begin(VkCommandBufferUsageFlags flags, CommandBuffer* primary_cmd_buf = nullptr);

	void end();

	void beginRenderPass(const std::vector<VkClearValue>& clearValues, const std::vector<std::unique_ptr<Subpass>>& subpasses, const VkExtent2D extent, VkSubpassContents subpassContents = VK_SUBPASS_CONTENTS_INLINE);

	void endRenderPass();
private:
	VkCommandBuffer handle{ VK_NULL_HANDLE };

	State state{ State::Initial };

	CommandPool &commandPool;

	const VkCommandBufferLevel level;

	RenderPass &renderPass;

	Framebuffer &framebuffer;

	uint32_t maxPushConstantsSize;

	//std::vector<uint8_t> stored_push_constants;

	//VkExtent2D last_framebuffer_extent{};

	//VkExtent2D last_render_area_extent{};
};

} // namespace vulkr