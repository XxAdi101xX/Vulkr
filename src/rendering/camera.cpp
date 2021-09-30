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

#include "camera.h"

namespace vulkr
{

Camera::Camera(int32_t viewportWidth, int32_t viewportHeight) : viewport{glm::vec2(viewportWidth, viewportHeight)}
{}

glm::vec2 Camera::getViewport() const
{
	return viewport;
}

float Camera::getFovY() const
{
	return fovy;
}

float Camera::getAspect() const
{
	return aspect;
}

float Camera::getClipNear() const
{
	return znear;
}

float Camera::getClipFar() const
{
	return zfar;
}

glm::vec3 Camera::getPosition() const
{
	return position;
}

glm::vec3 Camera::getCenter() const
{
	return center;
}

glm::vec3 Camera::getUp() const
{
	return up;
}

glm::mat4 Camera::getProjection() const
{
	return projection;
}

glm::mat4 Camera::getView() const
{
	return view;
}

glm::vec3 Camera::getViewDirection() const
{ 
	return -glm::transpose(view)[2];
}

glm::vec3 Camera::getRight() const
{
	return glm::transpose(view)[0];
}

bool Camera::isUpdated() const
{
	return updated;
}

void Camera::setFovY(float fovy)
{
	this->fovy = std::min(std::max(fovy, 0.01f), 179.0f);
	updatePerspectiveProjection();
}

void Camera::setAspect(float aspect)
{
	this->aspect = aspect;
	updatePerspectiveProjection();
}

void Camera::setPosition(glm::vec3 position)
{
	this->position = position;
	updateView();
}

void Camera::setCenter(glm::vec3 center)
{
	this->center = center;
	updateView();
}

void Camera::setUp(glm::vec3 up)
{
	this->up = up;
	updateView();
}


void Camera::setView(glm::vec3 position, glm::vec3 center, glm::vec3 up)
{
	this->position = position;
	this->center = center;
	this->up = up;

	updateView();
}

void Camera::setPerspectiveProjection(float fovy, float aspect, float znear, float zfar)
{
	this->fovy = std::min(std::max(fovy, 0.01f), 179.0f);
	this->aspect = aspect;
	this->znear = znear;
	this->zfar = zfar;

	updatePerspectiveProjection();
}

void Camera::updateView()
{
	view = glm::lookAt(position, center, up);
	updated = true;
}

void Camera::updatePerspectiveProjection()
{
	projection = glm::perspective(glm::radians(fovy), aspect, znear, zfar);
	projection[1][1] *= -1;
	updated = true;
}

void Camera::resetUpdatedFlag()
{
	updated = false;
}

} // namespace vulkr