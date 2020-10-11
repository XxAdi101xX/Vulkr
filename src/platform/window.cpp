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

#include "window.h"

namespace vulkr
{

Window::Window()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

	if (window == nullptr) {
		LOGEANDABORT("glfwCreateWindow has failed to create a window");
	}
}

Window::~Window()
{
	if (surface != VK_NULL_HANDLE) {
		vkDestroySurfaceKHR(instance->getHandle(), surface, nullptr);
	}

	glfwDestroyWindow(window);

	glfwTerminate();
}

void Window::createSurface(Instance *instance)
{
	if (surface != nullptr) {
		LOGEANDABORT("createSurface was called more than once")
	}
	this->instance = instance;

	VK_CHECK(glfwCreateWindowSurface(instance->getHandle(), window, nullptr, &surface));
}

const GLFWwindow *Window::getWindowHandle() const
{
	return window;
}

const VkSurfaceKHR Window::getSurfaceHandle() const
{
	return surface;
}

VkExtent2D Window::getWindowExtent()
{
	VkExtent2D extent = { WIDTH, HEIGHT };
	return extent;
}

} // namespace vulkr