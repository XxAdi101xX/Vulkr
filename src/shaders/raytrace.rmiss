#version 460
#extension GL_EXT_ray_tracing : require
// #extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

layout(push_constant) uniform Constants
{
	vec4 clearColor;
	vec3 lightPosition;
	float lightIntensity;
	int lightType; // 0: point, 1: infinite
};

void main()
{
	payload.hitValue = clearColor.xyz * 0.8;
}