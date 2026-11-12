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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace vulkr
{

/* One vertex as it comes out of the asset, laid out for meshoptimizer: the position occupies the first 12 bytes, which is
   what the clusterizer requires, followed by the texture coordinates and the slot of the primitive's base colour texture
   in the bindless texture array */
struct GltfVertex
{
	glm::vec3 position;
	glm::vec2 texCoord;
	uint32_t textureIndex{ 0u };
};

/* A flat indexed triangle mesh extracted from a glTF asset: every primitive of every mesh in the default scene is
   transformed into world space and merged, and the result is recentred and scaled to fit a unit cube so a fixed
   camera can frame it */
struct GltfMeshData
{
	std::vector<GltfVertex> vertices;
	std::vector<uint32_t> indices;
};

/* One decoded image of the asset, always expanded to RGBA8 */
struct GltfTextureData
{
	std::vector<unsigned char> pixels;
	uint32_t width{ 0u };
	uint32_t height{ 0u };
};

/* Everything a model needs from its asset: the merged geometry and the images its materials reference, in the order the
   asset declares them, so the image index doubles as the slot inside the model */
struct GltfAsset
{
	GltfMeshData mesh;
	std::vector<GltfTextureData> textures;
};

/* Load a .gltf or .glb asset; aborts when the file cannot be read or holds no indexed triangles */
GltfAsset loadGltfAsset(const std::string &filePath);

} // namespace vulkr
