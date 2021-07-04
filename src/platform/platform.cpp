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

namespace vulkr {

void Platform::initialize(std::unique_ptr<Application> &&application)
{
	if (application == nullptr) {
		LOGEANDABORT("Application is not valid");
	}

	this->application = std::move(application);

#ifdef VULKR_DEBUG
	spdlog::set_level(spdlog::level::debug);
#else
	spdlog::set_level(spdlog::level::info);
#endif

	spdlog::set_pattern(LOGGER_FORMAT);
	LOGI("Logger initialized");

	window = std::make_unique<Window>(*this);
}

void Platform::prepareApplication() const
{
	if (application)
	{
		application->prepare();
	}
}

void Platform::runMainProcessingLoop() const
{
	while (!window->shouldClose())
	{
		window->processEvents();
		processApplication();
	}
}

void Platform::processApplication() const
{
	if (application->isFocused()) {
		application->step();
	}
}

void Platform::terminate() const
{
	application->finish();
	spdlog::drop_all();
}

void Platform::createSurface(VkInstance instance)
{
	window->createSurface(instance);
}

void Platform::updateWindowTitle(float fps) const
{
	window->updateTitle(application->getName() + " [FPS: " + std::to_string(fps) + "]");
}

void Platform::handleWindowResize(const uint32_t width, const uint32_t height) const
{
	application->handleWindowResize(width, height);
}

void Platform::handleFocusChange(bool isFocused) const
{
	application->handleFocusChange(isFocused);
}

void Platform::handleInputEvents(const InputEvent &inputEvent) const
{
	application->handleInputEvents(inputEvent);
}

const VkSurfaceKHR Platform::getSurface() const
{
	return window->getSurfaceHandle();
}

const Window &Platform::getWindow() const
{
	return *window;
}

} // namespace vulkr