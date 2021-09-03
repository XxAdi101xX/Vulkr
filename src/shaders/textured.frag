#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "common.glsl"

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) flat in int baseInstance;

layout(location = 0) out vec4 outColor;

layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MaterialIndices {int i[]; }; // Material ID for each triangle

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;
layout(set = 2, binding = 0) uniform sampler2D[] textureSamplers;

void main() {
    ObjInstance objResource = objectBuffer.objects[baseInstance];
    Materials materials = Materials(objResource.materials);
    MaterialIndices matIndices = MaterialIndices(objResource.materialIndices);

    int materialIndex = matIndices.i[gl_PrimitiveID];
    WaveFrontMaterial mat = materials.m[materialIndex];

    uint txtId = uint(mat.textureId + objResource.textureOffset);

    outColor = texture(textureSamplers[nonuniformEXT(txtId)], fragTexCoord);
}