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

#include <tiny_obj_loader.h>
#include <glm/glm.hpp>

#include <unordered_map>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

 /* Obj vertex structure */
struct VertexObj
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 textureCoordinate;

	bool operator==(const VertexObj &other) const
	{
		return position == other.position && color == other.color && textureCoordinate == other.textureCoordinate;
	}
};

namespace std
{
template <> struct hash<VertexObj>
{
	size_t operator()(VertexObj const &vertex) const
	{
		return ((hash<glm::vec3>()(vertex.position) ^
			(hash<glm::vec3>()(vertex.color) << 1)) >>
			1) ^
			(hash<glm::vec2>()(vertex.textureCoordinate) << 1);
	}
};
} // namespace std

/* Obj material structure */
struct MaterialObj
{
	glm::vec3 ambient = glm::vec3(0.1f, 0.1f, 0.1f);
	glm::vec3 diffuse = glm::vec3(0.7f, 0.7f, 0.7f);
	glm::vec3 specular = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec3 transmittance = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 emission = glm::vec3(0.0f, 0.0f, 0.10);
	float shininess = 0.f;
	float ior = 1.0f;	   // index of refraction
	float dissolve = 1.0f; // 1 == opaque; 0 == fully transparent
						   // illumination model (see http://www.fileformat.info/format/material/)
	int illum = 0;
	int textureID = -1;
};

namespace vulkr
{

class ObjLoader
{
public:
	void loadModel(const char *filename);

	std::vector<VertexObj>   vertices;
	std::vector<uint32_t>    indices;
	std::vector<MaterialObj> materials;
	std::vector<int32_t>     materialIndices;
	std::vector<std::string> textures;
};

} // namespace vulkr