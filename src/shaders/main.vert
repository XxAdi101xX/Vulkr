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
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out int baseInstance;
layout(location = 4) out vec3 worldPos;
layout(location = 5) out vec3 viewDir;

void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].transform;
    vec3 origin = vec3(inverse(camera.view) * vec4(0, 0, 0, 1));
    
    // Setting pixel shader inputs
    fragColor = inColor;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    baseInstance = gl_BaseInstance;
    worldPos = vec3(modelMatrix * vec4(inPosition, 1.0));
    viewDir = vec3(worldPos - origin);

    gl_Position = camera.proj * camera.view * modelMatrix * vec4(inPosition, 1.0f);
}