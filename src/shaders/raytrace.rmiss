#version 460
#extension GL_EXT_ray_tracing : require
// #extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
	payload.hitValue = vec3(0.8f, 0.8f, 0.8f);
}