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

#include "platform.h"
#include "application.h"


namespace vulkr
{

Application::Application(Platform  &platform, std::string name) : platform{ platform }, name { name }
{}

void Application::prepare()
{
	timer.start();
}

void Application::step()
{
	if (focused)
	{
		update();
	}

	float elapsedTime = static_cast<float>(timer.elapsed<Timer::Seconds>());

	frameCount++;

	if (elapsedTime > 0.5f)
	{
		fps = (frameCount - lastFrameCount) / elapsedTime;
		lastFrameCount = frameCount;
		timer.lap();
		platform.updateWindowTitle(fps);
	}
}

void Application::finish()
{
	double executionTime = timer.stop();
	LOGI("Closing application (Runtime: {:.1f} seconds)", executionTime);
}

void Application::recreateSwapchain()
{
	// Nothing is done by default; applications should override this method if they want to support swapchain recreation for things like window resizing
}

void Application::handleWindowResize(const uint32_t width, const uint32_t height)
{
	// Nothing is done by default; applications should override this method if they want to support window resizing
}

void Application::handleFocusChange(bool isFocused)
{
	focused = isFocused;
}

void Application::handleInputEvents(const InputEvent &inputEvent)
{
	// Nothing is done by default; applications should override this method if they care about hardware inputs
}

bool Application::isFocused() const
{
	return focused;
}

std::string Application::getName() const
{
	return name;
}

float Application::getCurrentFps() const
{
	return fps;
}

} // namespace vulkr