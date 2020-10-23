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

//#include <iostream>

#include "app.h"
#include "platform/platform.h"
#include "core/image.h"

namespace vulkr
{

MainApp::MainApp(Platform& platform, std::string name) : Application{ platform, name } {}

 MainApp::~MainApp()
 {
     if (device)
     {
         device->waitIdle();
     }

     for (uint32_t i = 0; i < swapChainImageViews.size(); ++i) {
         swapChainImageViews[i].reset();
     }

     swapchain.reset();

     device.reset();

     // Must destroy surface before instance
	 if (surface != VK_NULL_HANDLE)
	 {
		 vkDestroySurfaceKHR(instance->getHandle(), surface, nullptr);
	 }

	 instance.reset();
 }

void MainApp::prepare()
{
    Application::prepare();


    instance = std::make_unique<Instance>(getName());

    platform.createSurface(instance->getHandle());
    surface = platform.getSurface();

    device = std::make_unique<Device>(std::move(instance->getSuitablePhysicalDevice()), surface, deviceExtensions);

    const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
    swapchain = std::make_unique<Swapchain>(*device, surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags);

    swapChainImageViews.reserve(swapchain->getImages().size());
    for (uint32_t i = 0; i < swapchain->getImages().size(); ++i) {
        swapChainImageViews.emplace_back(std::make_unique<ImageView>(*(swapchain->getImages()[i]), VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, swapchain->getProperties().surfaceFormat.format));
    }

}

void MainApp::update()
{

}

} // namespace vulkr

int main()
{
    vulkr::Platform platform;
    std::unique_ptr<vulkr::MainApp> app = std::make_unique<vulkr::MainApp>(platform, "Vulkan App");

    platform.initialize(std::move(app));
    platform.prepareApplication();

    platform.runMainProcessingLoop();
    platform.terminate();

    return EXIT_SUCCESS;
}

