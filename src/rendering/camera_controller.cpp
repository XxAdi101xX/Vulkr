/* Copyright (c) 2022 Adithya Venkatarao
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

#define _USE_MATH_DEFINES
#include <math.h>

#include "common/helpers.h"
#include "camera_controller.h"

namespace vulkr
{

CameraController::CameraController(int32_t viewportWidth, int32_t viewportHeight)
{
	camera = std::make_shared<Camera>(viewportWidth, viewportHeight);
}

std::shared_ptr<Camera> CameraController::getCamera() const
{
	return camera;
}

void CameraController::update(const float deltaTime)
{
	updateKeyboardMovement(glm::clamp(deltaTime, 0.0f, maxDeltaTime));
}

void CameraController::resetInput()
{
	movementInput = MovementInput{};
	movementVelocity = glm::vec3(0.0f);
	activeMouseInput = MouseInput::None;
}

void CameraController::updateKeyboardMovement(const float deltaTime)
{
	if (deltaTime <= 0.0f)
	{
		return;
	}

	const glm::vec3 position = camera->getPosition();
	const glm::vec3 center = camera->getCenter();
	const glm::vec3 backwardVector = position - center;
	const float distance = glm::length(backwardVector);

	if (distance <= std::numeric_limits<float>::epsilon())
	{
		movementVelocity = glm::vec3(0.0f);
		return;
	}

	// Build the camera basis from the view direction and the up axis, so movement follows where the camera is looking
	const glm::vec3 backward = backwardVector / distance;
	const glm::vec3 forward = -backward;
	glm::vec3 right = glm::cross(forward, camera->getUp());
	const float rightLength = glm::length(right);
	glm::vec3 cameraUp{ 0.0f };

	if (rightLength > std::numeric_limits<float>::epsilon())
	{
		right /= rightLength;
		cameraUp = glm::cross(right, forward);
	}
	else
	{
		// Looking straight up or down leaves no horizontal axis; forward movement still works
		right = glm::vec3(0.0f);
	}

	glm::vec3 direction{ 0.0f };

	if (movementInput.forward)
	{
		direction += forward;
	}
	if (movementInput.backward)
	{
		direction -= forward;
	}
	if (movementInput.right)
	{
		direction += right;
	}
	if (movementInput.left)
	{
		direction -= right;
	}
	if (movementInput.up)
	{
		direction += cameraUp;
	}
	if (movementInput.down)
	{
		direction -= cameraUp;
	}

	const float directionLength = glm::length(direction);
	glm::vec3 targetVelocity{ 0.0f };

	if (directionLength > std::numeric_limits<float>::epsilon())
	{
		// Normalizing keeps diagonals at the same speed as a single key, and scaling by the pivot distance keeps keyboard
		// speed proportional to how far the camera has been zoomed out
		targetVelocity = (direction / directionLength) * (keyboardMoveSpeed * distance);
	}

	// Ease the velocity towards the target so presses and releases do not start or stop the camera abruptly.
	// We use 1 - exp(-k*dt) instead of just k*dt to make the response independent of the frame time; the velocity approaches the target exponentially
	const float response = 1.0f - std::exp(-keyboardResponse * deltaTime);
	movementVelocity += (targetVelocity - movementVelocity) * response;

	// Exponential easing never quite reaches zero, so snap the last negligible amount to a full stop
	if (glm::dot(movementVelocity, movementVelocity) < std::numeric_limits<float>::epsilon())
	{
		movementVelocity = glm::vec3(0.0f);
	}

	const glm::vec3 offset = movementVelocity * deltaTime;
	camera->setView(position + offset, center + offset, camera->getUp());
}

void CameraController::handleInputEvents(const InputEvent &inputEvent)
{
	if (inputEvent.getEventSource() == EventSource::Keyboard)
	{
		const KeyInputEvent &keyInputEvent = dynamic_cast<const KeyInputEvent &>(inputEvent);

		if (keyInputEvent.getAction() == KeyAction::Unknown)
		{
			LOGEANDABORT("Unknown key action encountered");
		}

		// Keys track their held state; movement itself is integrated once per frame in update() so that it is smooth and
		// independent of the keyboard repeat rate
		const bool pressed = keyInputEvent.getAction() == KeyAction::Press || keyInputEvent.getAction() == KeyAction::Repeat;
		const bool released = keyInputEvent.getAction() == KeyAction::Release;

		auto setMovementInput = [&](bool &input) {
			if (pressed)
			{
				input = true;
			}
			else if (released)
			{
				input = false;
			}
		};

		switch (keyInputEvent.getInput())
		{
		case KeyInput::W:
			setMovementInput(movementInput.forward);
			break;
		case KeyInput::S:
			setMovementInput(movementInput.backward);
			break;
		case KeyInput::A:
			setMovementInput(movementInput.left);
			break;
		case KeyInput::D:
			setMovementInput(movementInput.right);
			break;
		case KeyInput::E:
			setMovementInput(movementInput.up);
			break;
		case KeyInput::Q:
			setMovementInput(movementInput.down);
			break;
		case KeyInput::F:
			// F is an alias for Q so both vertical pairs work
			setMovementInput(movementInput.down);
			break;
		}
	}
	else if (inputEvent.getEventSource() == EventSource::Mouse)
	{
		const MouseInputEvent &mouseInputEvent = dynamic_cast<const MouseInputEvent &>(inputEvent);

		if (mouseInputEvent.getAction() == MouseAction::Unknown)
		{
			LOGEANDABORT("Unknown mouse action encountered");
		}

		if (mouseInputEvent.getInput() == MouseInput::None)
		{
			handleCursorPositionChange(mouseInputEvent);
		}
		else if (mouseInputEvent.getInput() == MouseInput::Middle && mouseInputEvent.getAction() == MouseAction::Scroll)
		{
			handleMouseScroll(mouseInputEvent);
		}
		else
		{
			handleMouseButtonClick(mouseInputEvent);
		}
	}
}

void CameraController::handleMouseButtonClick(const MouseInputEvent &mouseInputEvent)
{
	if (mouseInputEvent.getAction() == MouseAction::Click)
	{
		activeMouseInput = mouseInputEvent.getInput();
		lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
	}
	else if (mouseInputEvent.getAction() == MouseAction::Release)
	{
		activeMouseInput = MouseInput::None;
	}
	else
	{
		LOGEANDABORT("Mouse input action is neither click or release");
	}
}

void CameraController::handleMouseScroll(const MouseInputEvent &mouseInputEvent)
{
	camera->setFovY(camera->getFovY() + (zoomStepSize * static_cast<float>(mouseInputEvent.getPositionY())));
}

void CameraController::handleCursorPositionChange(const MouseInputEvent &mouseInputEvent)
{
	if (mouseInputEvent.getAction() != MouseAction::Move)
	{
		LOGEANDABORT("Unexpected action type encountered for a mouse input event of type none");
	}

	switch (activeMouseInput)
	{
	case MouseInput::Left:
		orbit(mouseInputEvent);
		break;
	case MouseInput::Right:
		zoomOnMouseDrag(mouseInputEvent);
		break;
	case MouseInput::Middle:
		pan(float(mouseInputEvent.getPositionX() - lastMousePosition.x), float(mouseInputEvent.getPositionY() - lastMousePosition.y), 0.0f);
		lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
		break;
	}
}

void CameraController::orbit(const MouseInputEvent &mouseInputEvent)
{
	// Calculate the rotation magnitude along the x and y axis
	const float deltaAngleX = static_cast<float>(2.0f * M_PI / camera->getViewport().x);
	const float deltaAngleY = static_cast<float>(M_PI / camera->getViewport().y);
	const float rotationAngleX = static_cast<float>(lastMousePosition.x - mouseInputEvent.getPositionX()) * deltaAngleX;
	float rotationAngleY = static_cast<float>(lastMousePosition.y - mouseInputEvent.getPositionY()) * deltaAngleY;

	// If the viewing direction is the same as the up direction, we will get rapid model flipping so we want to avoid that
	if (glm::dot(camera->getViewDirection(), camera->getUp()) * sgn(rotationAngleY) > 0.99f)
	{
		rotationAngleY = 0.0f;
	}

	glm::vec4 newPosition{ camera->getPosition(), 1.0f };
	glm::vec4 pivot{ camera->getCenter(), 1.0f };
	glm::mat4 rotationMatrixX{ 1.0f };
	glm::mat4 rotationMatrixY{ 1.0f };

	// Rotate the camera around the pivot with respect to the x axis
	rotationMatrixX = glm::rotate(rotationMatrixX, rotationAngleX, camera->getUp());
	newPosition = (rotationMatrixX * (newPosition - pivot)) + pivot;

	// Rotate the camera around the pivot with respect to the y axis
	rotationMatrixY = glm::rotate(rotationMatrixY, rotationAngleY, camera->getRight());
	newPosition = (rotationMatrixY * (newPosition - pivot)) + pivot;

	// Update the camera view
	camera->setView(newPosition, camera->getCenter(), camera->getUp());

	// Update the last mouse position
	lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
}

void CameraController::zoomOnMouseDrag(const MouseInputEvent &mouseInputEvent)
{
	// Increase or decrease the fovy depending on the delta of the mouse change
	if (mouseInputEvent.getPositionX() > lastMousePosition.x)
	{
		camera->setFovY(camera->getFovY() - zoomStepSize);
	}
	else
	{
		camera->setFovY(camera->getFovY() + zoomStepSize);
	}

	// Update the last mouse position
	lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
}

void CameraController::pan(const float dx, const float dy, const float dz)
{
	const float dxNormalized = dx / float(camera->getViewport().x);
	const float dyNormalized = dy / float(camera->getViewport().y);

	glm::vec3 z{ camera->getPosition() - camera->getCenter() };
	float length = static_cast<float>(glm::length(z));
	z = glm::normalize(z);
	glm::vec3 x = glm::normalize(glm::cross(camera->getUp(), z));
	glm::vec3 y = glm::normalize(glm::cross(z, x));

	x *= -dxNormalized * length;
	y *= dyNormalized * length;
	z *= dz;

	camera->setPosition(camera->getPosition() + x + y + -z);
	camera->setCenter(camera->getCenter() + x + y + -z);
}

} // namespace vulkr