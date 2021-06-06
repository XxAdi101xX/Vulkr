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

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "common/vulkan_common.h"

namespace vulkr
{

class Camera
{
public:
	Camera() = default;
	virtual ~Camera() = default;

	/* Disable unnecessary operators to prevent error prone usages */
	Camera(Camera &&) = delete;
	Camera(const Camera &) = delete;
	Camera &operator=(const Camera &) = delete;
	Camera &operator=(Camera &&) = delete;

	/* Getters */
	float getFovY() const;
	float getAspect() const;
	float getClipNear() const;
	float getClipFar() const;
	glm::vec3 getPosition() const;
	glm::vec3 getCenter() const;
	glm::vec3 getUp() const;
	glm::mat4 getProjection() const;
	glm::mat4 getView() const;

	/* Setters */
	void setPosition(glm::vec3 position);
	void setView(glm::vec3 position, glm::vec3 center, glm::vec3 up);
	void setPerspectiveProjection(float fovy, float aspect, float znear, float zfar);

	void updateView();
	void updatePerspectiveProjection();
	void zoom(glm::vec3 magnitude);
private:
	float fovy;
	float aspect;
	float znear;
	float zfar;

	glm::vec3 position;
	glm::vec3 center;
	glm::vec3 up;

	glm::mat4 projection;
	glm::mat4 view;

	float calculateZoom(float delta, float positionCoordinate, float centerCoordinate);
};

} // namespace vulkr