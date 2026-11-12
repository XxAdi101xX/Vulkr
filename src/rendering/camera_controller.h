/* Copyright (c) 2021 Adithya Venkatarao
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
#include "platform/input_event.h"
#include "camera.h"


namespace vulkr
{

class CameraController
{
public:
	CameraController(int32_t viewportWidth, int32_t viewportHeight);
	virtual ~CameraController() = default;

	CameraController(CameraController &&) = delete;
	CameraController(const CameraController &) = delete;
	CameraController &operator=(const CameraController &) = delete;
	CameraController &operator=(CameraController &&) = delete;

	std::shared_ptr<Camera> getCamera() const;

	void handleInputEvents(const InputEvent &inputEvent);

	/* Advance held keyboard input; call once per frame with the frame delta time in seconds */
	void update(const float deltaTime);

	/* Forget held buttons and stop moving */
	void resetInput();
private:
	/* Movement is measured in multiples of the distance to the orbit pivot, so it feels the same when zoomed in or out */
	static constexpr float keyboardMoveSpeed{ 4.5f };

	/* Velocity approaches the target exponentially; higher values make starting and stopping snappier */
	static constexpr float keyboardResponse{ 9.0f };

	/* A frame that took longer than this (for example after the window was unfocused) must not teleport the camera */
	static constexpr float maxDeltaTime{ 0.1f };

	struct MovementInput
	{
		bool forward{ false };
		bool backward{ false };
		bool left{ false };
		bool right{ false };
		bool up{ false };
		bool down{ false };
	};

	const float zoomStepSize = 1.0f;

	std::shared_ptr<Camera> camera;
	MouseInput activeMouseInput{ MouseInput::None };
	glm::vec2 lastMousePosition{ glm::vec2(0.0f) };
	MovementInput movementInput;
	glm::vec3 movementVelocity{ 0.0f };

	void updateKeyboardMovement(const float deltaTime);
	void handleMouseButtonClick(const MouseInputEvent &mouseInputEvent);
	void handleMouseScroll(const MouseInputEvent &mouseInputEvent);
	void handleCursorPositionChange(const MouseInputEvent &mouseInputEvent);
	void orbit(const MouseInputEvent &mouseInputEvent);
	void zoomOnMouseDrag(const MouseInputEvent &mouseInputEvent);
	void pan(const float dx, const float dy, const float dz);

};

} // namespace vulkr