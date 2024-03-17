/* Copyright (c) 2018-2023, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 * Note that this file has been modified to integrate into the Vulkr engine, but the core logic remains the same and is covered under the license above.
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../common.glsl"
#define MAX_NUM_JOINTS 128

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec4 inJoint0;
layout (location = 5) in vec4 inWeight0;
layout (location = 6) in vec4 inColor0;


layout(set = 0, binding = 0) uniform CurrentFrameCameraBuffer {
    mat4 view;
    mat4 proj;
} currentFrameCameraBuffer;

layout(std140, set = 1, binding = 1) readonly buffer CurrentFrameGltfInstanceBuffer {
	GltfInstance gltfInstances[];
} currentFrameGltfInstanceBuffer;

layout (set = 3, binding = 0) uniform UBONode {
	mat4 matrix;
	mat4 jointMatrix[MAX_NUM_JOINTS];
	float jointCount;
} node;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV0;
layout (location = 3) out vec2 outUV1;
layout (location = 4) out vec4 outColor0;

void main() 
{
	mat4 modelMatrix = currentFrameGltfInstanceBuffer.gltfInstances[gl_BaseInstance].transform;

	outColor0 = inColor0;

	vec4 locPos;
	if (node.jointCount > 0.0) {
		// Mesh is skinned
		mat4 skinMat = 
			inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
			inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
			inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
			inWeight0.w * node.jointMatrix[int(inJoint0.w)];

		locPos = modelMatrix * node.matrix * skinMat * vec4(inPos, 1.0);
		outNormal = normalize(transpose(inverse(mat3(modelMatrix * node.matrix * skinMat))) * inNormal);
	} else {
		locPos = modelMatrix * node.matrix * vec4(inPos, 1.0);
		outNormal = normalize(transpose(inverse(mat3(modelMatrix * node.matrix))) * inNormal);
	}

	outWorldPos = locPos.xyz / locPos.w;
	outUV0 = inUV0;
	outUV1 = inUV1;
	gl_Position =  currentFrameCameraBuffer.proj * currentFrameCameraBuffer.view * vec4(outWorldPos, 1.0); // Perspective divide is done early hence we pass in 1.0 so that when it's automatically done in rasterizer, the results are unchanged.
}