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

#include <vector>

#include <glm/glm.hpp>
#include <meshoptimizer.h>

namespace vulkr
{

/* A mesh split into meshlets by meshoptimizer, laid out for one mesh shader workgroup per meshlet.
   The 8 bit triangle indices meshoptimizer produces are expanded to 32 bit so the shader does not need the
   8 bit storage feature. */
struct MeshletData
{
	/* One entry per meshlet; the offsets index meshletVertices and meshletTriangles */
	std::vector<meshopt_Meshlet> meshlets;

	/* Global vertex index for every meshlet vertex, indexed by meshlet.vertex_offset + local index */
	std::vector<uint32_t> meshletVertices;

	/* Local vertex indices, three consecutive entries per triangle, indexed by meshlet.triangle_offset */
	std::vector<uint32_t> meshletTriangles;

	/* Vertex positions that meshletVertices index into, quantized to two 16 bit signed normalized components per word
	   (x and y in .x, z in .y) so a vertex costs 8 bytes instead of 16 */
	std::vector<glm::uvec2> packedPositions;
};

/* Split an indexed triangle mesh into meshlets, optimizing it for the GPU first and the individual clusters after */
MeshletData buildMeshlets(const std::vector<glm::vec3> &positions, const std::vector<uint32_t> &indices,
	size_t maxVertices = 64, size_t maxTriangles = 124, float coneWeight = 0.25f);

} // namespace vulkr
