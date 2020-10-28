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
#include "common/timer.h"
#include "platform/input_event.h"

namespace vulkr
{

class Platform;

class Application
{
public:
	Application(Platform &platform, std::string name);
	virtual ~Application() = default;

	/* Disable unnecessary operators to prevent error prone usages */
	Application(const Application &) = delete;
	Application(Application &&) = delete;
	Application& operator=(const Application &) = delete;
	Application& operator=(Application &&) = delete;

	virtual void prepare();

	void step();

	virtual void update() = 0;

	void finish();

	virtual void recreateSwapchain();

	virtual void handleWindowResize(const uint32_t width, const uint32_t height);

	virtual void handleFocusChange(bool isFocused);

	virtual void handleInputEvents(const InputEvent& inputEvent) const;

	/* Getters */
	bool isFocused() const;
	std::string getName() const;
protected:
	/* A handle to the platform */
	Platform &platform;

	bool focused{ true };
private:
	std::string name;
	Timer timer;
	float fps{ 0.0f };
	uint32_t frameCount{ 0u };
	uint32_t lastFrameCount{ 0u };
};

} // namespace vulkr