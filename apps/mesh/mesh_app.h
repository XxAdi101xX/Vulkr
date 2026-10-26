/* Copyright (c) 2026 Adithya Venkatarao
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

#include <string>
#include <vector>

#include "platform/application.h"

namespace vulkr
{

/* Mesh shader application: dynamic rendering, synchronization2, buffer device address,
   descriptor indexing and mesh shaders. The overrides of prepare(), update() and finish()
   hold the rendering setup and are implemented in mesh_app.cpp */
class MeshApp : public Application
{
public:
	MeshApp(Platform &platform, std::string name);

	~MeshApp();

	virtual void update() override;
private:
	/* IMPORTANT NOTICE: Device features are enabled through the pNext chain in src/core/device.cpp; the mesh
	   shader features are only chained in when VK_EXT_mesh_shader is requested through this list */
	const std::vector<const char *> deviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
	};
};

} // namespace vulkr
