#version 460
#extension GL_GOOGLE_include_directive : require

#include "../common.glsl"

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec4 inColor0;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outUV0;
layout (location = 3) out vec4 outUV1;
layout (location = 4) out vec4 outColor0;
layout (location = 5) out vec4 outMaterialIndex;

layout (push_constant) uniform PushConstants {
	int materialIndex;
} pushConstants;

void main() 
{
	// TODO: this data is not packed together efficiently (eg. uv0 and uv1 can be put into a single output of size vec4' the material index can be packed with the position or normal)
	outPosition = vec4(inWorldPos, 0.0f);
	outNormal = vec4(inNormal, 0.0f);
	outUV0 = vec4(inUV0, 0.0f, 0.0f);
	outUV1 = vec4(inUV1, 0.0f, 0.0f);
	outColor0 = inColor0;
	outMaterialIndex = vec4(float(pushConstants.materialIndex), 0.0f, 0.0f, 0.0f);
}