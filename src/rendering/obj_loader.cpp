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

#define TINYOBJLOADER_IMPLEMENTATION
#include "obj_loader.h"
#include "common/logger.h"
#include "common/helpers.h"

namespace vulkr
{

void ObjLoader::loadModel(const char *filename)
{
	tinyobj::ObjReader reader;
	reader.ParseFromFile(filename);
	if (!reader.Valid())
	{
		LOGE(reader.Error().c_str());
		std::abort();
	}

	// Attrib contains the vertex arrays of the file
	const tinyobj::attrib_t &objAttrib = reader.GetAttrib();
	// Shapes contain the info for each separate object in the file
	const std::vector<tinyobj::shape_t> &objShapes = reader.GetShapes();
	// Materials contains the information about the material of each shape
	const std::vector<tinyobj::material_t> &objMaterials = reader.GetMaterials();

	std::unordered_map<VertexObj, uint32_t> uniqueVertices{};

	// Loop over shapes to populate the vertices and indices
	for (const auto &shape : objShapes)
	{
		vertices.reserve(shape.mesh.indices.size() + vertices.size());
		indices.reserve(shape.mesh.indices.size() + indices.size());
		materialIndices.insert(materialIndices.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());

		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
		{
			// Hardcode loading to triangles
			int fv = 3;

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++)
			{
				// Access to vertex
				tinyobj::index_t index = shape.mesh.indices[index_offset + v];

				VertexObj newVertex;

				// Vertex position
				tinyobj::real_t vx = objAttrib.vertices[3 * index.vertex_index + 0];
				tinyobj::real_t vy = objAttrib.vertices[3 * index.vertex_index + 1];
				tinyobj::real_t vz = objAttrib.vertices[3 * index.vertex_index + 2];
				newVertex.position = { vx, vy, vz };

				// Vertex normal (will be computed separately if they're not providede)
				if (!objAttrib.normals.empty() && index.normal_index >= 0)
				{
					tinyobj::real_t nx = objAttrib.normals[3 * index.normal_index + 0];
					tinyobj::real_t ny = objAttrib.normals[3 * index.normal_index + 1];
					tinyobj::real_t nz = objAttrib.normals[3 * index.normal_index + 2];
					newVertex.normal = { nx, ny, nz };
				}

				// Vertex colors
				if (!objAttrib.colors.empty())
				{
					tinyobj::real_t cx = objAttrib.colors[3 * index.vertex_index + 0];
					tinyobj::real_t cy = objAttrib.colors[3 * index.vertex_index + 1];
					tinyobj::real_t cz = objAttrib.colors[3 * index.vertex_index + 2];
					newVertex.color = { cx, cy, cz };
				}

				// Texture coordinates
				if (!objAttrib.texcoords.empty() && index.texcoord_index >= 0)
				{
					tinyobj::real_t ux = objAttrib.texcoords[2 * index.texcoord_index + 0];
					tinyobj::real_t uy = objAttrib.texcoords[2 * index.texcoord_index + 1];
					newVertex.textureCoordinate = { ux, 1.0f - uy };
				}

				if (uniqueVertices.count(newVertex) == 0)
				{
					uniqueVertices[newVertex] = to_u32(vertices.size());
					vertices.push_back(newVertex);
				}

				indices.push_back(uniqueVertices[newVertex]);
			}
			index_offset += fv;
		}
	}

	// Loop over the materials
	for (const auto &material : objMaterials)
	{
		MaterialObj matObj;
		matObj.ambient = glm::vec3(material.ambient[0], material.ambient[1], material.ambient[2]);
		matObj.diffuse = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
		matObj.specular = glm::vec3(material.specular[0], material.specular[1], material.specular[2]);
		matObj.emission = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
		matObj.transmittance = glm::vec3(material.transmittance[0], material.transmittance[1], material.transmittance[2]);
		matObj.dissolve = material.dissolve;
		matObj.ior = material.ior;
		matObj.shininess = material.shininess;
		matObj.illum = material.illum;

		if (!material.diffuse_texname.empty())
		{
			textures.push_back(material.diffuse_texname);
			matObj.textureID = static_cast<int32_t>(textures.size()) - 1;
		}

		materials.emplace_back(matObj);
	}

	// Add a default materials if none exist
	if (materials.empty())
	{
		materials.emplace_back(MaterialObj());
	}

	// Fixing material indices
	bool readujustedMaterialIndices = false;
	for (auto &mi : materialIndices)
	{
		if (mi < 0 || mi > materials.size())
		{
			readujustedMaterialIndices = true;
			mi = 0;
		}
	}
	if (readujustedMaterialIndices) LOGW("Material indices had to be readjusted to 0");

	// Compute normals if none are provided
	if (objAttrib.normals.empty())
	{
		for (size_t i = 0; i < indices.size(); i += 3)
		{
			VertexObj &v0 = vertices[indices[i + 0]];
			VertexObj &v1 = vertices[indices[i + 1]];
			VertexObj &v2 = vertices[indices[i + 2]];

			glm::vec3 n = glm::normalize(glm::cross((v1.position - v0.position), (v2.position - v0.position)));
			v0.normal = n;
			v1.normal = n;
			v2.normal = n;
		}
	}
}

} // namespace vulkr
