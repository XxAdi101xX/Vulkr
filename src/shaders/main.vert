#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_debug_printf : enable
#include "common.glsl"

layout(set = 0, binding = 0) uniform CurrentFrameCameraBuffer {
    mat4 view;
    mat4 proj;
} currentFrameCameraBuffer;

layout(std140, set = 1, binding = 0) readonly buffer CurrentFrameObjectBuffer {
	ObjInstance objects[];
} currentFrameObjectBuffer;

layout(set = 3, binding = 0) uniform PreviousFrameCameraBuffer {
    mat4 view;
    mat4 proj;
} previousFrameCameraBuffer;

layout(std140, set = 3, binding = 1) readonly buffer PreviousFrameObjectBuffer {
	ObjInstance objects[];
} previousFrameObjectBuffer;

layout(push_constant) uniform TaaPushConstant
{
    int frameSinceViewChange;
    vec2 jitter;
    int blank;
} pushConstant;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out int baseInstance;
layout(location = 4) out vec3 worldPos;
layout(location = 5) out vec3 viewDir;
layout(location = 6) out vec4 currentFramePosition;
layout(location = 7) out vec4 previousFramePosition;

void main() {
    mat4 currentFrameModelMatrix = currentFrameObjectBuffer.objects[gl_BaseInstance].transform;
    mat4 previousFrameModelMatrix = previousFrameObjectBuffer.objects[gl_BaseInstance].transform;
    vec3 origin = vec3(inverse(currentFrameCameraBuffer.view) * vec4(0, 0, 0, 1));
    
    // Setting pixel shader inputs
    fragColor = inColor;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    baseInstance = gl_BaseInstance;
    worldPos = vec3(currentFrameModelMatrix * vec4(inPosition, 1.0));
    viewDir = vec3(worldPos - origin);

    vec4 currentClipPosUnjittered = currentFrameCameraBuffer.proj * currentFrameCameraBuffer.view * currentFrameModelMatrix * vec4(inPosition, 1.0f);
    currentFramePosition = currentClipPosUnjittered;

    // jitter will be set to 0 if taa is disabled
    vec4 currentClipPosJittered = currentClipPosUnjittered + vec4(pushConstant.jitter, 0.0f, 0.0f);    
    gl_Position = currentClipPosJittered;

    vec4 previousClipPositionUnjittered = previousFrameCameraBuffer.proj * previousFrameCameraBuffer.view * previousFrameModelMatrix * vec4(inPosition, 1.0f);
    previousFramePosition = previousClipPositionUnjittered;
}