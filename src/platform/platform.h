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

#include "window.h"
#include "application.h"

namespace vulkr {

class InputEvent;

class Platform
{
public:
	Platform() = default;
	virtual ~Platform() = default;

	Platform(const Platform &) = delete;
	Platform(Platform &&) = delete;
	Platform &operator=(const Platform &) = delete;
	Platform &operator=(Platform &&) = delete;

	/* Initialize the platform by creating the window and logger */
	void initialize(std::unique_ptr<Application> &&applicationToOwn);

	/* Prepare the application before the main processing loop begins */
	void prepareApplication() const;

	/* The main processing loop that the applications runs on */
	void runMainProcessingLoop() const;

	/* Terminate all dependant components before finally deleting the platform */
	void terminate() const;

	/* Initiate the creation of a window surface */
	void createSurface(VkInstance instance);

	/* Update the title of the window */
	void updateWindowTitle(float fps) const;

	/* Handle window resizes */
	void handleWindowResize(const uint32_t width, const uint32_t height) const;

	/* Handle any window focus changes */
	void handleFocusChange(bool isFocused) const;

	/* Handle any inputs received by the window */
	void handleInputEvents(const InputEvent &inputEvent) const;

	/* Get the window surface */
	const VkSurfaceKHR getSurface() const;

	const Window &getWindow() const;
private:
	std::unique_ptr<Window> window{ nullptr };
	std::unique_ptr<Application> application{ nullptr };

	void processApplication() const;
};

} // namespace vulkr