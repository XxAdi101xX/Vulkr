#version 460

layout(set = 0, binding = 0) uniform CameraBuffer {
    mat4 view;
    mat4 proj;
} camera;

struct ObjectData {
	mat4 model;
};

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjectData objects[];
} objectBuffer;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
    gl_Position = camera.proj * camera.view * modelMatrix * vec4(inPosition, 1.0f);

    fragColor = inColor;
    fragTexCoord = inTexCoord;
}