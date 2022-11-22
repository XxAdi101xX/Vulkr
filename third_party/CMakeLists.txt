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

project(third_party LANGUAGES C CXX)

# tinyobjloader
add_library(tinyobjloader INTERFACE)
set(TINYOBJLOADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)
target_sources(tinyobjloader INTERFACE ${TINYOBJLOADER_DIR}/tiny_obj_loader.h)
target_include_directories(tinyobjloader INTERFACE ${TINYOBJLOADER_DIR})

# vulkan
add_library(vulkan INTERFACE)
set(VULKAN_INCLUDE_DIR $ENV{VULKAN_SDK}/Include)
target_sources(vulkan INTERFACE ${VULKAN_INCLUDE_DIR}/vulkan/vulkan.h)
target_include_directories(vulkan INTERFACE ${VULKAN_INCLUDE_DIR})

target_compile_definitions(vulkan INTERFACE VK_NO_PROTOTYPES)

# glm
add_library(glm INTERFACE)
set(GLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glm)
target_sources(glm INTERFACE ${GLM_DIR}/glm/glm.hpp)
target_include_directories(glm INTERFACE ${GLM_DIR})

target_compile_definitions(glm INTERFACE
    GLM_FORCE_SWIZZLE
    GLM_FORCE_RADIANS
    GLM_FORCE_CTOR_INIT
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_DEPTH_ZERO_TO_ONE # Use 0 to 1 as the clip space in the z-axis instead of the OpenGL default of -1 to 1
)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_definitions(glm INTERFACE GLM_FORCE_CXX14)
endif()

# volk
set(VOLK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/volk")
set(VOLK_FILES
    "${VOLK_DIR}/volk.c"
    "${VOLK_DIR}/volk.h"
)

add_library(volk STATIC ${VOLK_FILES})

target_link_libraries(volk PUBLIC vulkan)

target_include_directories(volk PUBLIC ${VOLK_DIR})

set_property(TARGET volk PROPERTY FOLDER "ThirdParty")

if(ANDROID)
    # native_app_glue
    set(NATIVE_APP_GLUE_DIR "${CMAKE_ANDROID_NDK}/sources/android/native_app_glue")

    file(GLOB NATIVE_APP_GLUE_FILES "${NATIVE_APP_GLUE_DIR}/*.c" "${NATIVE_APP_GLUE_DIR}/*.h")
    
    add_library(native_app_glue STATIC ${NATIVE_APP_GLUE_FILES})
        
    target_include_directories(native_app_glue PUBLIC ${NATIVE_APP_GLUE_DIR})
    
    set_property(TARGET native_app_glue PROPERTY FOLDER "ThirdParty")
else()
    if (NOT DIRECT_TO_DISPLAY)
        # GLFW
        option(GLFW_BUILD_DOCS OFF)
        option(GLFW_BUILD_TESTS OFF)
        option(GLFW_BUILD_EXAMPLES OFF)
        option(GLFW_BUILD_TESTS OFF)
        option(GLFW_INSTALL OFF)

        add_subdirectory(glfw)

        set_property(TARGET glfw PROPERTY FOLDER "ThirdParty")
    endif()
endif()

# spdlog
add_subdirectory(spdlog)
set_property(TARGET spdlog PROPERTY FOLDER "ThirdParty")

# vma
add_library(VulkanMemoryAllocator INTERFACE)
set(VMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator/include)
target_sources(VulkanMemoryAllocator INTERFACE ${VMA_DIR}/vk_mem_alloc.h)
target_include_directories(VulkanMemoryAllocator INTERFACE ${VMA_DIR})
target_link_libraries(VulkanMemoryAllocator INTERFACE vulkan)

# stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/stb")

# imgui
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set(IMGUI_FILES
    "${IMGUI_DIR}/imconfig.h"
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui.h"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_internal.h"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imstb_rectpack.h"
    "${IMGUI_DIR}/imstb_textedit.h"
    "${IMGUI_DIR}/imstb_truetype.h"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.h"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp")

add_library(imgui STATIC ${IMGUI_FILES})

target_link_libraries(imgui PUBLIC vulkan glfw)
target_include_directories(imgui PUBLIC ${IMGUI_DIR})

set_property(TARGET imgui PROPERTY FOLDER "ThirdParty")
