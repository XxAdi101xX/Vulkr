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
    common/vk_common.h
    common/strings.h
    # Source Files
    common/vk_common.cpp
    common/strings.cpp
)

set(CORE_FILES
    # Header Files
    core/device.h
    core/instance.h
    core/physical_device.h
    core/queue.h
    core/swapchain.h
    # Source Files
    core/device.cpp
    core/instance.cpp
    core/physical_device.cpp
    core/queue.cpp
    core/swapchain.cpp
)

set(PLATFORM_FILES
    # Header Files
    platform/window.h
    # Source Files
    platform/window.cpp)

source_group("common\\" FILES ${COMMON_FILES})
source_group("core\\" FILES ${CORE_FILES})
source_group("platform\\" FILES ${PLATFORM_FILES})

set(PROJECT_FILES
    ${PLATFORM_FILES}
    ${COMMON_FILES}
    ${CORE_FILES})

add_library(${PROJECT_NAME} STATIC ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    volk
    glm
    glfw
    spdlog)