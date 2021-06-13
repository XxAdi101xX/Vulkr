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

void CameraController::handleInputEvents(const InputEvent &inputEvent)
{
    if (inputEvent.getEventSource() == EventSource::Keyboard)
    {
        const KeyInputEvent &keyInputEvent = static_cast<const KeyInputEvent &>(inputEvent);

        if (keyInputEvent.getAction() == KeyAction::Unknown)
        {
            LOGEANDABORT("Unknown key action encountered");
        }

        // Handle key input
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
        pan(mouseInputEvent);
        break;
    }
}

void CameraController::orbit(const MouseInputEvent &mouseInputEvent)
{
    // Calculate the rotation magnitude along the x and y axis
    const float deltaAngleX = (2.0f * M_PI / camera->getViewport().x);
    const float deltaAngleY = (M_PI / camera->getViewport().y);
    float rotationAngleX = (lastMousePosition.x - mouseInputEvent.getPositionX()) * deltaAngleX;
    float rotationAngleY = (lastMousePosition.y - mouseInputEvent.getPositionY()) * deltaAngleY;

    // If the viewing direction is the same as the up direction, we will get rapid model flipping so we want to avoid that
    if (glm::dot(camera->getViewDirection(), camera->getUp()) * sgn(rotationAngleY) > 0.99f)
    {
        rotationAngleY = 0;
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

void CameraController::pan(const MouseInputEvent &mouseInputEvent)
{
    float dx = float(mouseInputEvent.getPositionX() - lastMousePosition.x) / float(camera->getViewport().x);
    float dy = float(mouseInputEvent.getPositionY() - lastMousePosition.y) / float(camera->getViewport().y);

    glm::vec3 z{ camera->getPosition() - camera->getCenter() };
    float length = static_cast<float>(glm::length(z));
    z = glm::normalize(z);
    glm::vec3 x = glm::normalize(glm::cross(camera->getUp(), z));
    glm::vec3 y = glm::normalize(glm::cross(z, x));
    x *= -dx * length;
    y *= dy * length;

    camera->setPosition(camera->getPosition() + x + y);
    camera->setCenter(camera->getCenter() + x + y);

    lastMousePosition = glm::vec2(mouseInputEvent.getPositionX(), mouseInputEvent.getPositionY());
}

} // namespace vulkr