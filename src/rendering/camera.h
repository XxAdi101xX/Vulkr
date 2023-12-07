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

 // GLM_FORCE_RADIANS and GLM_FORCE_DEPTH_ZERO_TO_ONE defined in CMAKE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "common/vulkan_common.h"

namespace vulkr
{

// Note that another intutitve way to implement the camera is displayed in the Vulkan repo by Sascha Willems (camera.hpp)
class Camera
{
public:
	Camera(int32_t viewportWidth, int32_t viewportHeight);
	virtual ~Camera() = default;

	Camera(Camera &&) = delete;
	Camera(const Camera &) = delete;
	Camera &operator=(const Camera &) = delete;
	Camera &operator=(Camera &&) = delete;

	/* Getters */
	glm::vec2 getViewport() const;
	float getFovY() const;
	float getAspect() const;
	float getClipNear() const;
	float getClipFar() const;
	glm::vec3 getPosition() const;
	glm::vec3 getCenter() const;
	glm::vec3 getUp() const;
	glm::mat4 getProjection() const;
	glm::mat4 getView() const;
	glm::vec3 getViewDirection() const;
	glm::vec3 getRight() const;
	bool isUpdated() const;

	/* Setters */
	void setFovY(float fovy);
	void setAspect(float aspect);
	void setPosition(glm::vec3 position);
	void setCenter(glm::vec3 center);
	void setUp(glm::vec3 up);
	void setView(glm::vec3 position, glm::vec3 center, glm::vec3 up);
	void setPerspectiveProjection(float fovy, float aspect, float znear, float zfar);
	void resetUpdatedFlag();
private:
	bool updated{ false };
	glm::vec2 viewport;

	float fovy;
	float aspect;
	float znear;
	float zfar;

	glm::vec3 position;
	glm::vec3 center;
	glm::vec3 up;

	glm::mat4 projection;
	glm::mat4 view;

	void updateView();
	void updatePerspectiveProjection();
};

} // namespace vulkr