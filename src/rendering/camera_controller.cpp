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

#include "camera_controller.h"

namespace
{

float calculateZoom(float delta, float positionCoordinate, float centerCoordinate)
{
	if (delta < 0.0f && positionCoordinate > centerCoordinate)
	{
		if ((delta + positionCoordinate) <= centerCoordinate)
		{
			return centerCoordinate + 0.001f;
		}
	}
	else if (delta > 0.0f && positionCoordinate < centerCoordinate)
	{
		if ((delta + positionCoordinate) >= centerCoordinate)
		{
			return centerCoordinate - 0.001f;
		}
	}

	return positionCoordinate + delta;
}

} // namespace

namespace vulkr
{

CameraController::CameraController()
{
	camera = std::make_shared<Camera>();
}

std::shared_ptr<Camera> CameraController::getCamera() const
{
	return camera;
}

void CameraController::handleInputEvents(const InputEvent &inputEvent)
{
    if (inputEvent.getEventSource() == EventSource::Keyboard)
    {
        const KeyInputEvent &keyInputEvent = static_cast<const KeyInputEvent &>(inputEvent);

        if (keyInputEvent.getAction() == KeyAction::Unknown)
        {
            LOGEANDABORT("Unknown key action encountered");
        }

        switch (keyInputEvent.getInput())
        {
        case KeyInput::Up:
            if (keyInputEvent.getAction() == KeyAction::Press || keyInputEvent.getAction() == KeyAction::Repeat)
            {
                // TODO
            }
            break;
        case KeyInput::Down:
            if (keyInputEvent.getAction() == KeyAction::Press || keyInputEvent.getAction() == KeyAction::Repeat)
            {
                // TODO
            }
            break;
        }
    }
    else if (inputEvent.getEventSource() == EventSource::Mouse)
    {
        const MouseInputEvent &mouseInputEvent = static_cast<const MouseInputEvent &>(inputEvent);

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
	zoom(glm::vec3(0.20f * static_cast<float>(mouseInputEvent.getPositionY())));
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
        // TODO: implement rotation with respect to the center
        break;
    case MouseInput::Right:
        if (abs(mouseInputEvent.getPositionX()) > abs(mouseInputEvent.getPositionY()))
        {
            zoom(glm::vec3(zoomDragStepSize * static_cast<float>(mouseInputEvent.getPositionX() - lastMousePosition.x)));
        }
        else
        {
            zoom(glm::vec3(zoomDragStepSize * static_cast<float>(lastMousePosition.y - mouseInputEvent.getPositionY())));
        }

        lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
        break;
    case MouseInput::Middle:
        // TODO: implement translatation of the camera
        break;
    }
}

void CameraController::zoom(glm::vec3 magnitude) const
{
	glm::vec3 newPosition;
	newPosition.x = calculateZoom(magnitude.x, camera->getPosition().x, camera->getCenter().x);
	newPosition.y = calculateZoom(magnitude.y, camera->getPosition().y, camera->getCenter().y);
	newPosition.z = calculateZoom(magnitude.z, camera->getPosition().z, camera->getCenter().z);

	camera->setPosition(newPosition);
}

} // namespace vulkr