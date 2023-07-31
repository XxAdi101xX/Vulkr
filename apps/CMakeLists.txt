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

cmake_minimum_required(VERSION 3.20)

add_executable(vulkr_app)
target_sources(vulkr_app PRIVATE vulkr.cpp vulkr.h)
target_link_libraries(vulkr_app PRIVATE src)
target_compile_features(vulkr_app PRIVATE cxx_std_20)
set_target_properties(vulkr_app PROPERTIES CXX_EXTENSIONS OFF)

add_executable(fluid_simulation)
target_sources(fluid_simulation PRIVATE fluid_simulation.cpp fluid_simulation.h)
target_link_libraries(fluid_simulation PRIVATE src)
target_compile_features(fluid_simulation PRIVATE cxx_std_20)
set_target_properties(fluid_simulation PROPERTIES CXX_EXTENSIONS OFF)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()