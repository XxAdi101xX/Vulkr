// Meshlet rendering: one mesh shader workgroup per meshlet, dispatched by vkCmdDrawMeshTasksEXT(meshletCount, 1, 1).
// Each invocation writes one vertex of the meshlet and every 64th triangle's worth of work is spread over the group.

struct Meshlet
{
    uint vertexOffset;      // offset into meshletVertices
    uint triangleOffset;    // offset into meshletTriangles, three entries per triangle
    uint vertexCount;       // vertices used by this meshlet
    uint triangleCount;     // triangles emitted by this meshlet
};

// The vertex the mesh shader reads: the position quantized into two words, the texture coordinate and the slot this
// vertex samples from the bindless texture array. The padding keeps the stride at 24 bytes in std430 layout
struct Vertex
{
    uint2 packedPosition;
    float2 texCoord;
    uint textureIndex;
    uint padding;
};

// The meshlet buffers are reached through buffer device addresses held in the push constant block, so no buffer needs a
// descriptor. Images cannot be addressed that way, so the textures stay in one bindless descriptor set: an unsized array
// whose descriptor count is decided when the set is allocated
struct PushConstants
{
    float4x4 viewProjection;
    float4x4 model;
    Vertex* vertices;
    Meshlet* meshlets;
    uint* meshletVertices;
    uint* meshletTriangles;
};

struct VertexOut
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
    nointerpolation uint textureIndex : TEXCOORD2;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

// Bindless textures
[[vk::binding(0, 0)]] SamplerState textureSampler;
[[vk::binding(1, 0)]] Texture2D<float4> textures[];

// Two signed normalized 16 bit components per word: x and y in .x, z in .y. The sign extension relies on the
// arithmetic shift of the signed int, so no 16 bit integer type (and no extra feature) is needed
float3 decodePosition(uint2 packed)
{
    const int x = int(packed.x << 16) >> 16;
    const int y = int(packed.x & 0xffff0000) >> 16;
    const int z = int(packed.y << 16) >> 16;
    return float3(x, y, z) * (1.0 / 32767.0);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void meshMain(uint threadIndex : SV_GroupThreadID, uint groupIndex : SV_GroupID,
    out vertices VertexOut verts[64], out indices uint3 tris[124])
{
    const Meshlet meshlet = pushConstants.meshlets[groupIndex];
    SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);

    if (threadIndex < meshlet.vertexCount)
    {
        const uint vertexIndex = pushConstants.meshletVertices[meshlet.vertexOffset + threadIndex];
        const Vertex vertex = pushConstants.vertices[vertexIndex];
        const float4 worldPosition = mul(pushConstants.model, float4(decodePosition(vertex.packedPosition), 1.0));

        verts[threadIndex].position = mul(pushConstants.viewProjection, worldPosition);
        verts[threadIndex].worldPosition = worldPosition.xyz;
        verts[threadIndex].texCoord = vertex.texCoord;
        verts[threadIndex].textureIndex = vertex.textureIndex;
    }

    // A meshlet can hold up to 124 triangles, so the invocations cover them in strides of the workgroup size
    for (uint triangleIndex = threadIndex; triangleIndex < meshlet.triangleCount; triangleIndex += 64)
    {
        const uint corner = meshlet.triangleOffset + triangleIndex * 3;
        tris[triangleIndex] = uint3(pushConstants.meshletTriangles[corner], pushConstants.meshletTriangles[corner + 1], pushConstants.meshletTriangles[corner + 2]);
    }
}

[shader("fragment")]
float4 fragMain(VertexOut input) : SV_Target
{
    // Flat shading from the screen space derivatives of the world position. The absolute value keeps both sides of
    // the surface lit, so the model does not turn black depending on which side of the mesh faces the camera
    const float3 normal = normalize(cross(ddx(input.worldPosition), ddy(input.worldPosition)));
    const float3 lightDirection = normalize(float3(0.4, 0.7, 0.5));
    const float diffuse = saturate(abs(dot(normal, lightDirection)));

    // The slot varies per triangle, so neighbouring fragments inside one wave can reference different descriptors;
    // NonUniformResourceIndex tells the compiler not to hoist a single descriptor load out of the divergent path
    const float4 albedo = textures[NonUniformResourceIndex(input.textureIndex)].Sample(textureSampler, input.texCoord);

    return float4(albedo.rgb * (0.15 + 0.85 * diffuse), 1.0);
}
