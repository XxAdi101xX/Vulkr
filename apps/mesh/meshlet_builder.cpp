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

#include "meshlet_builder.h"

#include "common/logger.h"

namespace vulkr
{

MeshletData buildMeshlets(const std::vector<GltfVertex> &vertices, const std::vector<uint32_t> &indices,
	size_t maxVertices, size_t maxTriangles, float coneWeight)
{
	MeshletData meshletData;

	// The whole GltfVertex structure is the unique key, so two corners only merge when position, texture coordinates and texture
	// slot are all the same
	const size_t vertexSize = sizeof(GltfVertex);

	// Indexing; glTF splits vertices per attribute seam, so the same position is often stored several times; the
	// remap table collapses byte identical vertices and rewrites both buffers to use the compact array
	std::vector<uint32_t> remap(vertices.size());
	const size_t vertexCount = meshopt_generateVertexRemap(remap.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), vertexSize);

	std::vector<uint32_t> remappedIndices(indices.size());
	meshopt_remapIndexBuffer(remappedIndices.data(), indices.data(), indices.size(), remap.data());

	std::vector<GltfVertex> remappedVertices(vertexCount);
	meshopt_remapVertexBuffer(remappedVertices.data(), vertices.data(), vertices.size(), vertexSize, remap.data());

	// Vertex cache optimization; Reorders triangles so vertices are referenced in the order a post transform
	// cache would retain them. A mesh shader has no such cache for its cluster local lookups, so the real gain here is
	// that the clustering below sees a spatially coherent triangle stream and builds tighter meshlets
	meshopt_optimizeVertexCache(remappedIndices.data(), remappedIndices.data(), remappedIndices.size(), vertexCount);

	// Overdraw optimization; Reorders triangles so the surfaces the mesh's own convexity puts in front are drawn first,
	// which saves the pixels the depth test would otherwise shade and then discard. The sort key is built from cluster
	// normals and centroids rather than from a camera, so it survives the model rotating, and it needs the vertex cache
	// optimized index buffer as input. The threshold lets it trade up to 5% vertex cache efficiency for less overdraw
	meshopt_optimizeOverdraw(remappedIndices.data(), remappedIndices.data(), remappedIndices.size(),
		reinterpret_cast<const float *>(remappedVertices.data()), vertexCount, vertexSize, 1.05f);

	// Index filtering; Removes degenerate triangles and duplicates (same post transform position and winding),
	// which would otherwise consume one of a meshlet's 124 triangle slots without producing a fragment. Filtering runs
	// before the fetch optimization so the vertices only those triangles used are dropped instead of uploaded
	const size_t unfilteredIndexCount = remappedIndices.size();
	const size_t filteredIndexCount = meshopt_filterIndexBuffer(
		remappedIndices.data(),
		remappedIndices.data(),
		remappedIndices.size(),
		remappedVertices.data(),
		vertexCount,
		sizeof(glm::vec3),
		vertexSize);
	remappedIndices.resize(filteredIndexCount);

	// Vertex fetch optimization; Compacts away vertices the filtered index buffer no longer references and
	// reorders the rest so the index buffer walks the vertex buffer almost sequentially, giving the mesh shader's
	// random access a denser working set. The returned count is the compacted vertex count, not the input count
	std::vector<GltfVertex> optimizedVertices(vertexCount);
	const size_t fetchedVertexCount = meshopt_optimizeVertexFetch(optimizedVertices.data(), remappedIndices.data(),
		remappedIndices.size(), remappedVertices.data(), vertexCount, vertexSize);
	optimizedVertices.resize(fetchedVertexCount);

	// Clustering, which the documented pipeline has no equivalent for because it describes a single indexed draw: the
	// triangle list becomes fixed size clusters, and cone_weight trades slightly worse packing for tighter normal cones
	const size_t maxMeshlets = meshopt_buildMeshletsBound(remappedIndices.size(), maxVertices, maxTriangles);
	meshletData.meshlets.resize(maxMeshlets);
	meshletData.meshletVertices.resize(remappedIndices.size());
	std::vector<unsigned char> meshletTriangles(remappedIndices.size());

	const size_t meshletCount = meshopt_buildMeshlets(
		meshletData.meshlets.data(),
		meshletData.meshletVertices.data(),
		meshletTriangles.data(),
		remappedIndices.data(),
		remappedIndices.size(),
		reinterpret_cast<const float *>(optimizedVertices.data()),
		optimizedVertices.size(),
		vertexSize,
		maxVertices,
		maxTriangles,
		coneWeight);

	meshletData.meshlets.resize(meshletCount);
	meshletData.meshletVertices.resize(meshletData.meshlets.back().vertex_offset + meshletData.meshlets.back().vertex_count);

	// Reorder the vertices and triangles inside every cluster for locality
	for (meshopt_Meshlet &meshlet : meshletData.meshlets)
	{
		meshopt_optimizeMeshlet(
			&meshletData.meshletVertices[meshlet.vertex_offset],
			&meshletTriangles[meshlet.triangle_offset],
			meshlet.triangle_count,
			meshlet.vertex_count);
	}

	meshletData.meshletTriangles.assign(meshletTriangles.begin(), meshletTriangles.end());

	// Vertex quantization. This has to run after the clustering because meshopt_buildMeshlets 
	// and the per meshlet bounds for culling need float positions. Two 16 bit signed normalized components per word,
	// 8 bytes per vertex instead of 16, decoded by the shader without needing a device feature
	meshletData.vertices.resize(optimizedVertices.size());
	for (size_t vertexIndex = 0; vertexIndex < optimizedVertices.size(); ++vertexIndex)
	{
		const GltfVertex &vertex = optimizedVertices[vertexIndex];
		const uint32_t x = static_cast<uint16_t>(meshopt_quantizeSnorm(vertex.position.x, 16));
		const uint32_t y = static_cast<uint16_t>(meshopt_quantizeSnorm(vertex.position.y, 16));
		const uint32_t z = static_cast<uint16_t>(meshopt_quantizeSnorm(vertex.position.z, 16));
		meshletData.vertices[vertexIndex] = MeshletVertex{
			.packedPosition = glm::uvec2(x | (y << 16), z),
			.texCoord = vertex.texCoord,
			.textureIndex = vertex.textureIndex,
		};
	}
	LOGW("Meshlet vertex quantization is ENABLED; 32 bit position attributes are quantized to 16 bits each");

	// TODO: Shadow indexing; it builds a second index buffer for depth only passes so attribute seams do not multiply vertices
	// std::vector<uint32_t> shadowIndices(remappedIndices.size());
	// meshopt_generateShadowIndexBuffer(shadowIndices.data(), remappedIndices.data(), remappedIndices.size(),
	//     optimizedVertices.data(), optimizedVertices.size(), sizeof(glm::vec3), vertexSize);

	size_t triangleCount = 0;
	size_t largestMeshlet = 0;
	for (const meshopt_Meshlet &meshlet : meshletData.meshlets)
	{
		triangleCount += meshlet.triangle_count;
		largestMeshlet = std::max<size_t>(largestMeshlet, meshlet.triangle_count);
	}

	LOGI("Index filtering removed {} of {} indices", unfilteredIndexCount - filteredIndexCount, unfilteredIndexCount);
	LOGI("Built {} meshlets for {} triangles ({} vertices, {} bytes each), largest meshlet holds {} triangles",
		meshletData.meshlets.size(), triangleCount, meshletData.vertices.size(), sizeof(MeshletVertex), largestMeshlet);

	return meshletData;
}

} // namespace vulkr
