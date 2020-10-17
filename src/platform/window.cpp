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
#include "platform.h"
#include "input_event.h"
#include "common/helpers.h"

namespace vulkr
{

namespace
{

void errorCallback(int error, const char *description)
{
	LOGEANDABORT("GLFW error code {} thrown: {}", error, description);
}

void windowCloseCallback(GLFWwindow *window)
{
	glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void windowSizeCallback(GLFWwindow *window, int width, int height)
{
	if (Window *windowClassHandle = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window)))
	{
		const Platform &platform = windowClassHandle->getPlatform();
		platform.handleWindowResize(to_u32(width), to_u32(height));
	}
}

void windowFocusCallback(GLFWwindow *window, int focused)
{
	if (Window *windowClassHandle = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window)))
	{
		const Platform &platform = windowClassHandle->getPlatform();
		platform.handleFocusChange(focused ? true : false);
	}
}

void keyCallback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
	KeyInput keyInput = KeyInput::Unknown;
	KeyAction keyAction = KeyAction::Unknown;

	auto keyInputIt = keyInputMap.find(key);
	if (keyInputIt != keyInputMap.end())
	{
		keyInput = keyInputIt->second;
	}

	auto keyActionIt = keyActiontMap.find(action);
	if (keyActionIt != keyActiontMap.end())
	{
		keyAction = keyActionIt->second;
	}

	if (Window *windowClassHandle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)))
	{
		const Platform &platform = windowClassHandle->getPlatform();
		platform.handleInputEvents(KeyInputEvent{ keyInput, keyAction });
	}
}

void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos)
{
	if (Window *windowClassHandle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)))
	{
		const Platform &platform = windowClassHandle->getPlatform();
		platform.handleInputEvents(MouseInputEvent{
			MouseInput::None,
			MouseAction::Move,
			xPos,
			yPos
		});
	}
}

void mouseButtonBallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
	MouseInput mouseInput = MouseInput::None;
	MouseAction mouseAction = MouseAction::Unknown;

	auto mouseInputIt = mouseInputMap.find(button);
	if (mouseInputIt != mouseInputMap.end())
	{
		mouseInput = mouseInputIt->second;
	}

	auto mouseActionIt = mouseActionMap.find(action);
	if (mouseActionIt != mouseActionMap.end())
	{
		mouseAction = mouseActionIt->second;
	}

	if (Window *windowClassHandle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)))
	{
		const Platform& platform = windowClassHandle->getPlatform();
		double xPos, yPos;
		glfwGetCursorPos(window, &xPos, &yPos);

		platform.handleInputEvents(MouseInputEvent{
			mouseInput,
			mouseAction,
			xPos,
			yPos
		});
	}
}

} // namespace vulkr


Window::Window(Platform& platform) : platform{ platform }
{
	if (!glfwInit())
	{
		LOGEANDABORT("GLFW failed to initialize");
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	handle = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	if (handle == VK_NULL_HANDLE) {
		LOGEANDABORT("glfwCreateWindow has failed to create a window");
	}

	// Expose the window class to glfw
	glfwSetWindowUserPointer(handle, this);

	/* Direct window interation callbacks */
	glfwSetErrorCallback(errorCallback);
	glfwSetWindowCloseCallback(handle, windowCloseCallback);
	glfwSetWindowSizeCallback(handle, windowSizeCallback);
	glfwSetWindowFocusCallback(handle, windowFocusCallback);

	/* Window input callbacks */
	glfwSetKeyCallback(handle, keyCallback);
	glfwSetCursorPosCallback(handle, cursorPositionCallback);
	glfwSetMouseButtonCallback(handle, mouseButtonBallback);

	/* Prevent inputs from being lost if actions happen between pollEvent calls */
	glfwSetInputMode(handle, GLFW_STICKY_KEYS, 1);
	glfwSetInputMode(handle, GLFW_STICKY_MOUSE_BUTTONS, 1);
}

Window::~Window()
{
	if (surface != VK_NULL_HANDLE) {
		vkDestroySurfaceKHR(instance, surface, nullptr);
	}

	glfwDestroyWindow(handle);
	glfwTerminate();
}

void Window::createSurface(VkInstance instance)
{
	if (surface != nullptr) {
		LOGEANDABORT("createSurface was called more than once")
	}
	this->instance = instance;

	VK_CHECK(glfwCreateWindowSurface(instance, handle, nullptr, &surface));
}

const GLFWwindow *Window::getHandle() const
{
	return handle;
}

const VkSurfaceKHR Window::getSurfaceHandle() const
{
	return surface;
}

const Platform &Window::getPlatform() const
{
	return platform;
}

VkExtent2D Window::getWindowExtent()
{
	VkExtent2D extent = { WIDTH, HEIGHT };
	return extent;
}

bool Window::shouldClose() const
{
	return glfwWindowShouldClose(handle);
}

void Window::processEvents() const
{
	glfwPollEvents();
}

void Window::updateTitle(std::string title) const
{
	glfwSetWindowTitle(handle, title.c_str());
}

} // namespace vulkr