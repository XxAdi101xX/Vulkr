#version 460
//#extension GL_EXT_debug_printf : enable
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "common.glsl"
#include "raycommon.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(buffer_reference, scalar) buffer Vertices { Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices { ivec3 i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MaterialIndices {int i[]; }; // Material ID for each triangle

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
// set 1 is the global buffer
layout(std140, set = 2, binding = 0) readonly buffer ObjectBuffer {
	ObjInstance objects[];
} objectBuffer;

layout(push_constant) uniform Constants
{
	vec4  clearColor;
	vec3  lightPosition;
	float lightIntensity;
	int   lightType;
}
pushC;

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
    float lightIntensity = pushC.lightIntensity;
    float lightDistance  = 100000.0;

    // Point light
    if (pushC.lightType == 0)
    {
        vec3 lDir      = pushC.lightPosition - worldPos;
        lightDistance  = length(lDir);
        lightIntensity = pushC.lightIntensity / (lightDistance * lightDistance);
        L              = normalize(lDir);
    }
    else // Directional light
    {
        L = normalize(pushC.lightPosition - vec3(0));
    }

    float dotNL = max(dot(normal, L), 0.2);

    prd.hitValue = vec3(dotNL);
/*
    // Material of the object
    int materialIndex = matIndices.i[gl_PrimitiveID];
    WaveFrontMaterial mat = materials.m[materialIndex];

    // Diffuse
    vec3 diffuse = computeDiffuse(mat, L, normal);
    if(mat.textureId >= 0)
    {
        uint txtId = mat.textureId + scnDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
        vec2 texCoord =
            v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
        diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
    }

    // Specular
    vec3 specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
    prd.hitValue = vec3(lightIntensity * (diffuse + specular));
*/
}
