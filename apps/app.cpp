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

namespace vulkr
{

MainApp::MainApp(Platform& platform, std::string name) : Application{ platform, name } {}

 MainApp::~MainApp()
 {
     if (device)
     {
         device->waitIdle();
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

    surface = platform.getSurface();

    instance = std::make_unique<Instance>(getName());

    platform.createSurface(instance->getHandle());

    device = std::make_unique<Device>(std::move(instance->getSuitablePhysicalDevice()), surface, deviceExtensions);

    const std::set<VkImageUsageFlagBits> imageUsageFlags{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
    //swapchain = std::make_unique<Swapchain>(*device, surface, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, imageUsageFlags);

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

