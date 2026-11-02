// Two triangles from a single mesh shader workgroup: one that covers the whole viewport and a smaller one
// drawn on top of it. There is no task (amplification) stage yet, so the workgroup count comes from
// vkCmdDrawMeshTasksEXT on the host.


struct VertexOut
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};


[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(out vertices VertexOut verts[6], out indices uint3 tris[2])
{
    SetMeshOutputCounts(6, 2);

    // Primitive 0: positions outside the clip volume so the triangle covers the whole viewport
    const float4 orange = float4(1.0, 0.4, 0.1, 1.0);

    verts[0].position = float4(-1.0, -1.0, 0.0, 1.0);
    verts[1].position = float4(3.0, -1.0, 0.0, 1.0);
    verts[2].position = float4(-1.0, 3.0, 0.0, 1.0);

    verts[0].color = orange;
    verts[1].color = orange;
    verts[2].color = orange;

    // Primitive 1: a smaller triangle inset inside the viewport, rasterized after primitive 0
    const float4 green = float4(0.1, 0.9, 0.2, 1.0);

    verts[3].position = float4(-0.5, -0.4, 0.0, 1.0);
    verts[4].position = float4(0.5, -0.4, 0.0, 1.0);
    verts[5].position = float4(0.0, 0.6, 0.0, 1.0);

    verts[3].color = green;
    verts[4].color = green;
    verts[5].color = green;

    tris[0] = uint3(0, 1, 2);
    tris[1] = uint3(3, 4, 5);
}

[shader("fragment")]
float4 fragMain(VertexOut input) : SV_Target
{
    return input.color;
}