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

#include <GLFW/glfw3.h>
namespace vulkr
{

class Platform;

class Window
{
public:
	Window(Platform &platform);
	~Window();

	/* Disable unnecessary operators to prevent error prone usages */
	Window(const Window &) = delete;
	Window(Window &&) = delete;
	Window& operator=(const Window &) = delete;
	Window& operator=(Window &&) = delete;

	/* Create the window surface */
	void createSurface(VkInstance instance);

	/* Getters */
	const GLFWwindow *getHandle() const;
	const VkSurfaceKHR getSurfaceHandle() const;
	const Platform &getPlatform() const;
	static VkExtent2D getWindowExtent();

	/* Checks whether the window should close */
	bool shouldClose() const;

	/* Poll for any input events */
	void processEvents() const;

	/* Update the title of the window */
	void updateTitle(std::string title) const;
private:
	GLFWwindow *handle;
	VkSurfaceKHR surface;
	VkInstance instance;
	Platform &platform;

	static const int32_t WIDTH{ 1280 };
	static const int32_t HEIGHT{ 720 };
};

} // namespace vulkr
