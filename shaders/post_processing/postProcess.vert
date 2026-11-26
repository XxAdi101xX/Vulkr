#version 460

// The post process pass covers the entire render target with a single oversized triangle so that every
// pixel is resolved, not just the pixels that happen to be covered by scene geometry. This also lets the
// fragment shader run before ImGui is composited, keeping UI pixels out of the temporal history.
void main()
{
	vec2 position = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
	gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
