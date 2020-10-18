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

#include <fstream>

#include "shader_module.h"
#include "common/helpers.h"
#include "common/logger.h"

namespace vulkr
{

// ShaderSource implementations
ShaderSource::ShaderSource(const std::string &fileName) :
	fileName{ fileName },
	data{ readFile(fileName) }
{}

const std::string& ShaderSource::getFileName() const
{
	return fileName;
}

const std::vector<char>& ShaderSource::getData() const
{
	return data;
}

std::vector<char> ShaderSource::readFile(const std::string& filename) const
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

// ShaderModule implementations
ShaderModule::ShaderModule(Device &device, VkShaderStageFlagBits stage, const ShaderSource &shaderSource, const char *entryPoint) :
	device{ device },
	stage{ stage },
	shaderSource{ shaderSource },
	entryPoint{ entryPoint }
{
	// Check if the SPIR-V that's passed in is empty
	if (shaderSource.getData().empty())
	{
		LOGEANDABORT("Empty spirv file encountered");
	}
}

VkShaderStageFlagBits ShaderModule::getStage() const
{
	return stage;
}

const std::string &ShaderModule::getEntryPoint() const
{
	return entryPoint;
}

const ShaderSource &ShaderModule::getShaderSource() const
{
	return shaderSource;
}

} // namespace vulkr