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

#include "core/instance.h"
#include "core/device.h"
#include "core/swapchain.h"
#include "core/image_view.h"
#include "core/render_pass.h"
#include "rendering/subpass.h"
#include "rendering/shader_module.h"
#include "rendering/pipeline_state.h"
#include "core/pipeline_layout.h"
#include "core/pipeline.h"

#include "platform/application.h"

namespace vulkr
{

class MainApp : public Application
{
public:
    MainApp(Platform &platform, std::string name);

    ~MainApp();

    virtual void prepare() override;

    virtual void update();
protected:
    /* The Vulkan instance */
    std::unique_ptr<Instance> instance{ nullptr };
    
    VkSurfaceKHR surface{ VK_NULL_HANDLE };

    /* The Vulkan device */
    std::unique_ptr<Device> device{ nullptr };

    std::unique_ptr<Swapchain> swapchain{ nullptr };

    std::vector<std::unique_ptr<ImageView>> swapChainImageViews;

    std::vector<VkAttachmentReference> inputAttachments;
    std::vector<VkAttachmentReference> colorAttachments;
    std::vector<VkAttachmentReference> resolveAttachments;
    std::vector<VkAttachmentReference> depthStencilAttachments;
    std::vector<uint32_t> preserveAttachments;

    std::vector<Subpass> subpasses;
    std::unique_ptr<RenderPass> renderPass{ nullptr };

    std::vector<ShaderModule> shaderModules;
    std::unique_ptr<GraphicsPipeline> pipeline{ nullptr };

    const std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
};

} // namespace vulkr