#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "../common.glsl"

layout(constant_id = 0) const uint maxLightCount = 100;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) flat in int baseInstance;
layout(location = 4) in vec3 worldPos;
layout(location = 5) in vec3 viewDir;
layout(location = 6) in vec4 currentFramePosition;
layout(location = 7) in vec4 previousFramePosition;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outColorCopy;
layout(location = 2) out vec4 outVelocity;

layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MaterialIndices {int i[]; }; // Material ID for each triangle

layout(std140, set = 0, binding = 1) uniform LightBuffer {
	LightData lights[maxLightCount];
} lightBuffer;
layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;
layout(set = 2, binding = 0) uniform sampler2D[] textureSamplers;

layout(push_constant) uniform RasterizationPushConstant
{
    layout(offset = 16) int lightCount;
} pushConstant;

void main() {
    //debugPrintfEXT("the lightcount is %d", pushConstant.lightCount);
    //debugPrintfEXT("the fragcoord is: %1.4v2f", gl_FragCoord.xy);
    ObjInstance objResource = objectBuffer.objects[baseInstance];
    Materials materials = Materials(objResource.materials);
    MaterialIndices matIndices = MaterialIndices(objResource.materialIndices);

    int materialIndex = matIndices.i[gl_PrimitiveID];
    WaveFrontMaterial mat = materials.m[materialIndex];

    vec3 N = normalize(fragNormal);

    vec3 outputValue = vec3(0.0);

    // Vector toward light
    vec3  L;
    float lightIntensity;
    for (int lightIndex = 0; lightIndex < pushConstant.lightCount; ++lightIndex)
    {
        lightIntensity = lightBuffer.lights[lightIndex].intensity;
        if (lightBuffer.lights[lightIndex].type == 0)
        {
            vec3  lightDir     = lightBuffer.lights[lightIndex].position - worldPos;
            float d        = length(lightDir);
            lightIntensity = lightBuffer.lights[lightIndex].intensity / (d * d);
            L              = normalize(lightDir);
        }
        else if (lightBuffer.lights[lightIndex].type == 1)
        {   
            vec3 lightDir = vec3(
		        sin(radians(lightBuffer.lights[lightIndex].rotation.x)) * cos(radians(lightBuffer.lights[lightIndex].rotation.y)),
		        sin(radians(lightBuffer.lights[lightIndex].rotation.y)),
		        cos(radians(lightBuffer.lights[lightIndex].rotation.x)) * cos(radians(lightBuffer.lights[lightIndex].rotation.y))
	        );
            L = normalize(lightDir);
            lightIntensity *= 0.01;
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
        outputValue += vec3(lightIntensity * lightBuffer.lights[lightIndex].color * (diffuse + specular));
    }
    outColor = vec4(outputValue, 1);
    outColorCopy = vec4(outputValue, 1);

    // Populate the velocity image
    vec2 newPos = (currentFramePosition.xy / currentFramePosition.w) * 0.5 + 0.5;
	vec2 prePos = (previousFramePosition.xy / previousFramePosition.w) * 0.5 + 0.5;
    vec2 velocity = newPos - prePos;
    outVelocity = vec4(velocity, 0.0f, 1.0f);
}