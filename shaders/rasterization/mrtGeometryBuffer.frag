#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : enable

#include "../common.glsl"

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec4 inColor0;

layout (location = 0) out vec3 outPosition;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV0;
layout (location = 3) out vec2 outUV1;
layout (location = 4) out vec4 outColor0;
layout (location = 5) out int outMaterialIndex;

layout (push_constant) uniform PushConstants {
	int materialIndex;
} pushConstants;

void main() 
{
	// TODO: this data is not packed together efficiently (eg. uv0 and uv1 can be put into a single output of size vec4)
	outPosition = inWorldPos;
	outNormal = inNormal;
	outUV0 = inUV0;
	outUV1 = inUV1;
	outColor0 = inColor0;
	// TODO investigate why changing line 18 from int to float and casting this to a float results in 1 being returned in deferredShading.frag and 0 when it's not changed...
	outMaterialIndex = pushConstants.materialIndex;//vec4(float(pushConstants.materialIndex), 3.0f, 3.0f, 3.0f);

	//debugPrintfEXT("outMaterialIndex is %f, %f", outMaterialIndex, float(outMaterialIndex));
}