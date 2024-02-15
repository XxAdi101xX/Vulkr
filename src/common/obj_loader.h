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

#include <glm/gtx/hash.hpp> // requires GLM_ENABLE_EXPERIMENTAL

 /* Obj vertex structure; used in shader so needs to follow alignment rules */
struct VertexObj
{
	glm::vec3 position;
	int padding1;
	glm::vec3 normal;
	int padding2;
	glm::vec3 color;
	int padding3;
	glm::vec2 textureCoordinate;
	glm::vec2 padding4;

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

/* Obj material structure; used in shader so needs to follow alignment rules */
struct MaterialObj
{
	glm::vec3 ambient = glm::vec3(0.1f, 0.1f, 0.1f);
	float shininess = 0.f;
	glm::vec3 diffuse = glm::vec3(0.7f, 0.7f, 0.7f);
	float ior = 1.0f;	   // index of refraction
	glm::vec3 specular = glm::vec3(1.0f, 1.0f, 1.0f);
	float dissolve = 1.0f; // 1 == opaque; 0 == fully transparent
	glm::vec3 transmittance = glm::vec3(0.0f, 0.0f, 0.0f);
	int illum = 0;
	glm::vec3 emission = glm::vec3(0.0f, 0.0f, 0.10);
	// illumination model (see http://www.fileformat.info/format/material/)
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