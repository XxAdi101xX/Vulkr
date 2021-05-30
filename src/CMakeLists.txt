# Copyright (c) 2020 Adithya Venkatarao
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

cmake_minimum_required(VERSION 3.10)

project(src LANGUAGES C CXX)

set(COMMON_FILES
    # Header Files
    common/helpers.h
    common/logger.h
    common/vulkan_common.h
    common/strings.h
    common/fence_pool.h
    common/semaphore_pool.h
    common/timer.h
    # Source Files
    common/vulkan_common.cpp
    common/strings.cpp
    common/fence_pool.cpp
    common/semaphore_pool.cpp
    common/timer.cpp
)

set(CORE_FILES
    # Header Files
    core/device.h
    core/instance.h
    core/physical_device.h
    core/queue.h
    core/swapchain.h
    core/image_view.h
    core/image.h
    core/render_pass.h
    core/pipeline.h
    core/pipeline_layout.h
    core/framebuffer.h
    core/command_pool.h
    core/command_buffer.h
    core/buffer.h
    core/descriptor_set_layout.h
    core/descriptor_pool.h
    core/descriptor_set.h
    core/sampler.h
    # Source Files
    core/device.cpp
    core/instance.cpp
    core/physical_device.cpp
    core/queue.cpp
    core/swapchain.cpp
    core/image_view.cpp
    core/image.cpp
    core/render_pass.cpp
    core/pipeline.cpp
    core/pipeline_layout.cpp
    core/framebuffer.cpp
    core/command_pool.cpp
    core/command_buffer.cpp
    core/buffer.cpp
    core/descriptor_set_layout.cpp
    core/descriptor_pool.cpp
    core/descriptor_set.cpp
    core/sampler.cpp
)

set(PLATFORM_FILES
    # Header Files
    platform/input_event.h
    platform/window.h
    platform/platform.h
    platform/application.h
    # Source Files
    platform/input_event.cpp
    platform/window.cpp
    platform/platform.cpp
    platform/application.cpp
)

set(RENDERING_FILES
    # Header Files
    rendering/subpass.h
    rendering/shader_module.h
    rendering/pipeline_state.h
    # Source Files
    rendering/subpass.cpp
    rendering/shader_module.cpp
    rendering/pipeline_state.cpp
)

source_group("common\\" FILES ${COMMON_FILES})
source_group("core\\" FILES ${CORE_FILES})
source_group("platform\\" FILES ${PLATFORM_FILES})
source_group("rendering\\" FILES ${RENDERING_FILES})

set(PROJECT_FILES
    ${PLATFORM_FILES}
    ${COMMON_FILES}
    ${CORE_FILES}
    ${RENDERING_FILES}
)

add_library(${PROJECT_NAME} STATIC ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    volk
    glm
    glfw
    spdlog
    VulkanMemoryAllocator
    stb
    tinyobjloader
)