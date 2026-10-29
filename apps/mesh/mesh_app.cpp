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

#include "mesh_app.h"

#include "common/logger.h"
#include "common/helpers.h"
#include "platform/platform.h"

#include "core/instance.h"
#include "core/device.h"
#include "core/queue.h"

namespace vulkr
{

MeshApp::MeshApp(Platform &platform, std::string name) :
	Application(platform, std::move(name))
{
}

MeshApp::~MeshApp()
{
	m_device->waitIdle();
	m_instance.reset();
	m_device.reset();

}

void MeshApp::prepare()
{
	Application::prepare();

	// Create the Vulkan instance and surface for the application
	m_instance = std::make_unique<Instance>(getName());

	platform.createSurface(m_instance->getHandle());
	m_surface = platform.getSurface();

	// Create the Vulkan physical and logical device with the required extensions
	std::unique_ptr<PhysicalDevice> physicalDevice = m_instance->getSuitablePhysicalDevice();

	m_device = std::make_unique<Device>(std::move(physicalDevice), m_surface, deviceExtensions);

	// Find general queue for graphics, compute, transfer and presentation
	int32_t desiredQueueFamilyIndex = m_device->getQueueFamilyIndexByFlags(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, true);

	if (desiredQueueFamilyIndex >= 0)
	{
		m_generalQueue = m_device->getQueue(to_u32(desiredQueueFamilyIndex), 0u);
	}
	else
	{
		LOGEANDABORT("TODO: Devices where a queue supporting graphics, compute and transfer do not exist are not supported yet");
	}

}

void MeshApp::update()
{
}

} // namespace vulkr
