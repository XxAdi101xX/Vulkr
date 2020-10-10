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

#pragma once

#include "common/vulkan_common.h"

namespace vulkr
{

class Device;

class ShaderSource
{
public:
	ShaderSource(const std::string &filename);

	const std::string &getFileName() const;

	const std::vector<char> &getData() const;

private:
	/* The filename of the compiled spirv */
	const std::string fileName;

	/* The contents of the spirv file */
	std::vector<char> data;
};

class ShaderModule
{
public:
	ShaderModule(
		Device& device,
		VkShaderStageFlagBits stage,
		const ShaderSource &shaderSource,
		const char *entryPoint = "main"
	);

	/* Disable unnecessary operators to prevent error prone usages */
	ShaderModule(const ShaderModule &) = delete;
	ShaderModule(ShaderModule &&) = delete;
	ShaderModule& operator=(const ShaderModule &) = delete;
	ShaderModule& operator=(ShaderModule &&) = delete;

	VkShaderStageFlagBits getStage() const;

	const std::string &getEntryPoint() const;
	const ShaderSource &getShaderSource() const;

private:
	Device &device;

	// Stage of the shader (vertex, fragment, etc)
	VkShaderStageFlagBits stage{};

	// Name of the main function
	const std::string entryPoint;

	// Shader source information
	const ShaderSource &shaderSource;
};

}