#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "../common.glsl"
#include "raycommon.glsl"

hitAttributeEXT vec2 attribs;

layout(constant_id = 0) const uint maxLightCount = 100;

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices { Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices { int i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MaterialIndices {int i[]; }; // Material ID for each triangle

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(std140, set = 1, binding = 1) uniform LightBuffer {
	LightData lights[maxLightCount];
} lightBuffer;
layout(std140, set = 2, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;
layout(set = 3, binding = 0) uniform sampler2D[] textureSamplers;

layout(push_constant) uniform RaytracingPushConstant
{
	int lightCount;
	int frameSinceViewChange;
} pushConstant;

void main()
{
    // Object data
    ObjInstance objResource = objectBuffer.objects[gl_InstanceCustomIndexEXT];
    Vertices vertices = Vertices(objResource.vertices);
    Indices indices = Indices(objResource.indices);
    Materials materials = Materials(objResource.materials);
    MaterialIndices matIndices = MaterialIndices(objResource.materialIndices);
  
    // Vertex of the triangle
    Vertex v0 = vertices.v[indices.i[gl_PrimitiveID * 3]];
    Vertex v1 = vertices.v[indices.i[gl_PrimitiveID * 3 + 1]];
    Vertex v2 = vertices.v[indices.i[gl_PrimitiveID * 3 + 2]];

    // The hitpoint's barycentric coordinates
    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Computing the normal at hit position
    vec3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    // Transforming the normal to world space
    normal = normalize(vec3(objResource.transformIT * vec4(normal, 0.0)));

    // Computing the coordinates of the hit position
    vec3 worldPos = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;
    // Transforming the position to world space
    worldPos = vec3(objResource.transform * vec4(worldPos, 1.0));

    // Vector toward the light
    vec3 outputColor = vec3(0.0);
    vec3 L;
    float lightDistance  = 100000.0;
    float lightIntensity;
    
    for (int lightIndex = 0; lightIndex < pushConstant.lightCount; ++lightIndex)
    {
    
        lightIntensity = lightBuffer.lights[lightIndex].intensity;

        // Point light
        if (lightBuffer.lights[lightIndex].type == 0)
        {
            vec3 lightDir  = lightBuffer.lights[lightIndex].position - worldPos;
            lightDistance  = length(lightDir);
            lightIntensity = lightBuffer.lights[lightIndex].intensity / (lightDistance * lightDistance);
            L              = normalize(lightDir);
        }
        else if (lightBuffer.lights[lightIndex].type == 1) // Directional light
        {
            vec3 lightDir = vec3(
		        sin(radians(lightBuffer.lights[lightIndex].rotation.x)) * cos(radians(lightBuffer.lights[lightIndex].rotation.y)),
		        sin(radians(lightBuffer.lights[lightIndex].rotation.y)),
		        cos(radians(lightBuffer.lights[lightIndex].rotation.x)) * cos(radians(lightBuffer.lights[lightIndex].rotation.y))
	        );
            L = normalize(lightDir);
            lightIntensity *= 0.01;
        }


        // Material of the object
        int materialIndex = matIndices.i[gl_PrimitiveID];
        WaveFrontMaterial mat = materials.m[materialIndex];

        // Diffuse
        vec3 diffuse = computeDiffuse(mat, L, normal);
        if (mat.textureId >= 0)
        {
            uint txtId = uint(mat.textureId + objResource.textureOffset);
            vec2 texCoord = v0.textureCoordinate * barycentrics.x + v1.textureCoordinate * barycentrics.y + v2.textureCoordinate * barycentrics.z;
            diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
        }

        vec3 specular = vec3(0);
        float attenuation = 1.0;

        // Tracing shadow ray only if the light is visible from the surface
        if (dot(normal, L) > 0)
        {
            float tMin   = 0.001;
            float tMax   = lightDistance;
            vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
            vec3  rayDir = L;
            uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
            isShadowed   = true;

            traceRayEXT(
                topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex (0 is regular miss while 1 is the shadow miss)
                origin,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
            );

            if (isShadowed)
            {
                attenuation = 0.3;
            }
            else
            {
                // Compute specular since we're not in the shadow
                specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
            }
        }

        outputColor += vec3(lightIntensity * lightBuffer.lights[lightIndex].color * attenuation * (diffuse + specular));
    }

    payload.hitValue = outputColor;
}
