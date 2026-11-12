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

// The tinygltf implementation lives in src/rendering/gltf_loader.cpp, which includes stb_image.h because stb_image's
// implementation is provided by the app (vulkr.h does it for the other app). Defining it here keeps tinygltf linkable.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "gltf_mesh.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include "common/logger.h"
#include "common/helpers.h"

namespace vulkr
{

namespace
{

glm::mat4 getNodeMatrix(const tinygltf::Node &node)
{
	if (node.matrix.size() == 16)
	{
		return glm::make_mat4(node.matrix.data());
	}

	glm::mat4 matrix{ 1.0f };

	if (node.translation.size() == 3)
	{
		matrix = glm::translate(matrix, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
	}

	if (node.rotation.size() == 4)
	{
		const glm::quat rotation{ static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]), static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]) };
		matrix *= glm::mat4_cast(rotation);
	}

	if (node.scale.size() == 3)
	{
		matrix = glm::scale(matrix, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
	}

	return matrix;
}

/* Reads an accessor element by element so interleaved buffer views with a byte stride are handled as well. elementSize
   is the size of one accessor element (component count times component size) and is only used for tightly packed views */
template <typename T>
const T *getAccessorElement(const tinygltf::Model &model, const tinygltf::Accessor &accessor, size_t index, size_t elementSize = sizeof(T))
{
	const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
	const size_t stride = bufferView.byteStride != 0 ? bufferView.byteStride : elementSize;
	const unsigned char *data = model.buffers[bufferView.buffer].data.data() + bufferView.byteOffset + accessor.byteOffset;
	return reinterpret_cast<const T *>(data + index * stride);
}

uint32_t readIndex(const tinygltf::Model &model, const tinygltf::Accessor &accessor, size_t index)
{
	switch (accessor.componentType)
	{
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
		return *getAccessorElement<unsigned char>(model, accessor, index);
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
		return *getAccessorElement<unsigned short>(model, accessor, index);
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
		return *getAccessorElement<unsigned int>(model, accessor, index);
	default:
		LOGEANDABORT("Unsupported glTF index component type {}", accessor.componentType);
	}
}

/* The slot of the material's base colour texture in the bindless array, which is the index of the image it decodes from */
uint32_t getBaseColorTextureSlot(const tinygltf::Model &model, int32_t materialIndex)
{
	if (materialIndex < 0 || materialIndex >= static_cast<int32_t>(model.materials.size()))
	{
		return 0u;
	}

	const int32_t textureIndex = model.materials[materialIndex].pbrMetallicRoughness.baseColorTexture.index;
	if (textureIndex < 0 || textureIndex >= static_cast<int32_t>(model.textures.size()))
	{
		return 0u;
	}

	const int32_t imageIndex = model.textures[textureIndex].source;
	return imageIndex >= 0 ? to_u32(imageIndex) : 0u;
}

void collectNode(const tinygltf::Model &model, int32_t nodeIndex, const glm::mat4 &parentMatrix, GltfMeshData &meshData)
{
	const tinygltf::Node &node = model.nodes[nodeIndex];
	const glm::mat4 worldMatrix = parentMatrix * getNodeMatrix(node);

	if (node.mesh >= 0)
	{
		for (const tinygltf::Primitive &primitive : model.meshes[node.mesh].primitives)
		{
			// Only triangle lists with positions and an index buffer are supported for now
			if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
			{
				continue;
			}

			const auto positionAttribute = primitive.attributes.find("POSITION");
			if (positionAttribute == primitive.attributes.end() || primitive.indices < 0)
			{
				continue;
			}

			const tinygltf::Accessor &positionAccessor = model.accessors[positionAttribute->second];
			const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
			const tinygltf::Accessor *texCoordAccessor{ nullptr };
			const auto texCoordAttribute = primitive.attributes.find("TEXCOORD_0");
			if (texCoordAttribute != primitive.attributes.end())
			{
				texCoordAccessor = &model.accessors[texCoordAttribute->second];
			}

			const uint32_t textureIndex = getBaseColorTextureSlot(model, primitive.material);

			// One vertex per index entry, in the index buffer's winding; the vertex remap collapses the duplicates later.
			// Emitting per corner keeps the texture slot of each corner faithful to the primitive that produced it
			for (size_t index = 0; index < indexAccessor.count; ++index)
			{
				const uint32_t sourceVertex = readIndex(model, indexAccessor, index);

				// position is a vec3 of floats, so a tightly packed view advances 12 bytes per vertex
				const float *position = getAccessorElement<float>(model, positionAccessor, sourceVertex, sizeof(float) * 3u);
				const glm::vec3 localPosition{ position[0], position[1], position[2] };

				glm::vec2 texCoord{ 0.0f };
				if (texCoordAccessor != nullptr && sourceVertex < texCoordAccessor->count)
				{
					const float *texCoordValues = getAccessorElement<float>(model, *texCoordAccessor, sourceVertex, sizeof(float) * 2u);
					texCoord = glm::vec2(texCoordValues[0], texCoordValues[1]);
				}

				meshData.vertices.emplace_back(GltfVertex{
					.position = glm::vec3(worldMatrix * glm::vec4(localPosition, 1.0f)),
					.texCoord = texCoord,
					.textureIndex = textureIndex,
				});
				meshData.indices.emplace_back(to_u32(meshData.vertices.size()) - 1u);
			}
		}
	}

	for (int32_t childIndex : node.children)
	{
		collectNode(model, childIndex, worldMatrix, meshData);
	}
}

/* Recentre the mesh on the origin and scale it to fit a unit cube so a fixed camera can frame it */
void normalizeMesh(GltfMeshData &meshData)
{
	if (meshData.vertices.empty())
	{
		return;
	}

	glm::vec3 minimum{ meshData.vertices[0].position };
	glm::vec3 maximum{ meshData.vertices[0].position };
	for (const GltfVertex &vertex : meshData.vertices)
	{
		minimum = glm::min(minimum, vertex.position);
		maximum = glm::max(maximum, vertex.position);
	}

	const glm::vec3 center = (minimum + maximum) * 0.5f;
	const glm::vec3 extent = maximum - minimum;
	const float largestExtent = glm::max(extent.x, glm::max(extent.y, extent.z));
	const float scale = largestExtent > 0.0f ? 2.0f / largestExtent : 1.0f;

	for (GltfVertex &vertex : meshData.vertices)
	{
		vertex.position = (vertex.position - center) * scale;
	}
}

/* The images come back decoded by tinygltf, in the order the asset declares them, which is what the shader indexes */
std::vector<GltfTextureData> extractTextures(const tinygltf::Model &model)
{
	std::vector<GltfTextureData> textures;
	textures.reserve(model.images.size());

	for (const tinygltf::Image &image : model.images)
	{
		if (image.bits != 8 || (image.component != 4 && image.component != 3))
		{
			LOGEANDABORT("The image {} is {}-bit with {} components, only 8-bit RGB and RGBA are supported", image.uri, image.bits, image.component);
		}

		GltfTextureData texture;
		texture.width = to_u32(image.width);
		texture.height = to_u32(image.height);

		// Expand RGB to RGBA so every texel is 4 bytes and the copy can use one format for all textures
		if (image.component == 4)
		{
			texture.pixels = image.image;
		}
		else
		{
			texture.pixels.resize(static_cast<size_t>(image.width) * image.height * 4u);
			for (size_t texel = 0; texel < static_cast<size_t>(image.width) * image.height; ++texel)
			{
				texture.pixels[texel * 4u + 0u] = image.image[texel * 3u + 0u];
				texture.pixels[texel * 4u + 1u] = image.image[texel * 3u + 1u];
				texture.pixels[texel * 4u + 2u] = image.image[texel * 3u + 2u];
				texture.pixels[texel * 4u + 3u] = 255u;
			}
		}

		textures.emplace_back(std::move(texture));
	}

	return textures;
}



} // namespace

GltfAsset loadGltfAsset(const std::string &filePath)
{
	tinygltf::TinyGLTF loader;
	tinygltf::Model model;
	std::string error;
	std::string warning;

	const bool isBinary = filePath.size() >= 4 && filePath.compare(filePath.size() - 4, 4, ".glb") == 0;
	const bool loaded = isBinary ? loader.LoadBinaryFromFile(&model, &error, &warning, filePath) : loader.LoadASCIIFromFile(&model, &error, &warning, filePath);

	if (!warning.empty())
	{
		LOGW("tinygltf warning while loading {}: {}", filePath, warning);
	}

	if (!loaded)
	{
		LOGEANDABORT("Failed to load the glTF asset {}: {}", filePath, error);
	}

	GltfAsset asset;
	GltfMeshData &meshData = asset.mesh;
	const int32_t sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
	if (sceneIndex >= static_cast<int32_t>(model.scenes.size()))
	{
		LOGEANDABORT("The glTF asset {} does not define a scene", filePath);
	}

	for (int32_t nodeIndex : model.scenes[sceneIndex].nodes)
	{
		collectNode(model, nodeIndex, glm::mat4(1.0f), meshData);
	}

	if (meshData.indices.empty())
	{
		LOGEANDABORT("The glTF asset {} does not contain any indexed triangles", filePath);
	}

	normalizeMesh(meshData);
	asset.textures = extractTextures(model);

	LOGI("Loaded {} vertices, {} indices and {} textures from {}", meshData.vertices.size(), meshData.indices.size(),
		asset.textures.size(), filePath);

	return asset;
}

} // namespace vulkr
