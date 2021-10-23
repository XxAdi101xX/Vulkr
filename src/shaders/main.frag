#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
//#extension GL_EXT_debug_printf : enable

#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) flat in int baseInstance;
layout(location = 4) in vec3 worldPos;
layout(location = 5) in vec3 viewDir;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform RasterizationPushConstant
{
    layout(offset = 16)
	vec3 lightPosition;
	float lightIntensity;
	int lightType; // 0: point, 1: infinite
} pushConstant;

layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MaterialIndices {int i[]; }; // Material ID for each triangle

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;
layout(set = 2, binding = 0) uniform sampler2D[] textureSamplers;

void main() {
    // debugPrintfEXT("Example print of float is is %f", pushConstant.lightIntensity);
    ObjInstance objResource = objectBuffer.objects[baseInstance];
    Materials materials = Materials(objResource.materials);
    MaterialIndices matIndices = MaterialIndices(objResource.materialIndices);

    int materialIndex = matIndices.i[gl_PrimitiveID];
    WaveFrontMaterial mat = materials.m[materialIndex];

    vec3 N = normalize(fragNormal);

    // Vector toward light
    vec3  L;
    float lightIntensity = pushConstant.lightIntensity;
    if (pushConstant.lightType == 0)
    {
        vec3  lDir     = pushConstant.lightPosition - worldPos;
        float d        = length(lDir);
        lightIntensity = pushConstant.lightIntensity / (d * d);
        L              = normalize(lDir);
    }
    else
    {
        L = normalize(pushConstant.lightPosition - vec3(0));
    }

    // Diffuse
    vec3 diffuse = computeDiffuse(mat, L, N);
    if (mat.textureId >= 0)
    {
        uint txtId  = uint(mat.textureId + objResource.textureOffset);
        vec3 diffuseTxt = texture(textureSamplers[nonuniformEXT(txtId)], fragTexCoord).xyz;
        diffuse *= diffuseTxt;
    }

    // Specular
    vec3 specular = computeSpecular(mat, viewDir, L, N);

    // Result
    outColor = vec4(lightIntensity * (diffuse + specular), 1);
}