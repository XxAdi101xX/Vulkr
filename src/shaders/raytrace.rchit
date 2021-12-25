#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"
#include "raycommon.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices { Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices { ivec3 i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MaterialIndices {int i[]; }; // Material ID for each triangle

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
//layout(set = 1, binding = 0) uniform LightBuffer {
//	LightData lights[];
//} lightBuffer;
layout(std140, set = 2, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;
layout(set = 3, binding = 0) uniform sampler2D[] textureSamplers;

layout(push_constant) uniform RaytracingPushConstant
{
	vec3  lightPosition;
	float lightIntensity;
	int   lightType;
	int   frameSinceViewChange;
} pushConstant;

void main()
{
    // Object data
    ObjInstance objResource = objectBuffer.objects[gl_InstanceCustomIndexEXT];
    Vertices vertices = Vertices(objResource.vertices);
    Indices indices = Indices(objResource.indices);
    Materials materials = Materials(objResource.materials);
    MaterialIndices matIndices = MaterialIndices(objResource.materialIndices);
  
    // Indices of the triangle
    ivec3 ind = indices.i[gl_PrimitiveID];
  
    // Vertex of the triangle
    Vertex v0 = vertices.v[ind.x];
    Vertex v1 = vertices.v[ind.y];
    Vertex v2 = vertices.v[ind.z];

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
    vec3  L;
    float lightIntensity = pushConstant.lightIntensity;
    float lightDistance  = 100000.0;

    // Point light
    if (pushConstant.lightType == 0)
    {
        vec3 lDir      = pushConstant.lightPosition - worldPos;
        lightDistance  = length(lDir);
        lightIntensity = pushConstant.lightIntensity / (lightDistance * lightDistance);
        L              = normalize(lDir);
    }
    else // Directional light
    {
        L = normalize(pushConstant.lightPosition - vec3(0));
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

    payload.hitValue = vec3(lightIntensity * attenuation * (diffuse + specular));
}
