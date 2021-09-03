#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"

layout(set = 0, binding = 0) uniform CameraBuffer {
    mat4 view;
    mat4 proj;
} camera;

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out int baseInstance;

void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].transform;
    gl_Position = camera.proj * camera.view * modelMatrix * vec4(inPosition, 1.0f);

    fragColor = inColor;
    fragTexCoord = inTexCoord;
    baseInstance = gl_BaseInstance;
}