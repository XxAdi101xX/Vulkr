struct Vertex
{
  vec3 position;
  vec3 normal;
  vec3 color;
  vec2 textureCoordinate;
};

struct ObjectData
{
	mat4 model;
	mat4 modelIT;
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
};