#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable
//#extension GL_EXT_debug_printf : enable

#include "common.glsl"
#include "random.glsl"

layout(set = 0, binding = 0) uniform CurrentFrameCameraBuffer
{
    mat4 view;
    mat4 proj;
} camera;

layout(std140, set = 0, binding = 1) readonly buffer CurrentFrameObjectBuffer {
	ObjInstance objects[];
} currentFrameObjectBuffer;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(push_constant) uniform PostProcessingPushConstant
{
    vec2 jitter;
	int frameSinceViewChange;
} pushConstant;

void main() {
//debugPrintfEXT("jitter is %f %f", pushConstant.jitter.x, pushConstant.jitter.y);
    mat4 modelMatrix = currentFrameObjectBuffer.objects[gl_BaseInstance].transform;
    vec4 clipPos = camera.proj * camera.view * modelMatrix * vec4(inPosition, 1.0f);
    gl_Position = clipPos;
}