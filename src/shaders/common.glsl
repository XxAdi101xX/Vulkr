struct Vertex
{
    vec3 position;
    vec3 normal;
    vec3 color;
    vec2 textureCoordinate;
};

struct ObjInstance
{
	mat4 transform;           // Model transform
	mat4 transformIT;         // Transpose of the inverse of the model transformation
    uint64_t objIndex;        // Object index
    uint64_t textureOffset;   // Offset of texture
    uint64_t vertices;        // BufferDeviceAddress
    uint64_t indices;         // BufferDeviceAddress
    uint64_t materials;       // BufferDeviceAddress
    uint64_t materialIndices; // BufferDeviceAddress
};

struct WaveFrontMaterial
{
    vec3  ambient;
    vec3  diffuse;
    vec3  specular;
    vec3  transmittance;
    vec3  emission;
    float shininess;
    float ior;       // index of refraction
    float dissolve;  // 1 == opaque; 0 == fully transparent
    int   illum;     // illumination model (see http://www.fileformat.info/format/material/)
    int   textureId;
};

vec3 computeDiffuse(WaveFrontMaterial mat, vec3 lightDir, vec3 normal)
{
    // Lambertian BRDF
    float dotNL = max(dot(normal, lightDir), 0.0);
    vec3  c     = mat.diffuse * dotNL;

    if (mat.illum >= 1)
    {
        c += mat.ambient;
    }
    return c;
}

vec3 computeSpecular(WaveFrontMaterial mat, vec3 viewDir, vec3 lightDir, vec3 normal)
{
    if (mat.illum < 2)
    {
        return vec3(0);
    }

    const float kPi        = 3.14159265;
    const float kShininess = max(mat.shininess, 4.0);

    // Specular
    const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
    vec3        V                   = normalize(-viewDir);
    vec3        R                   = reflect(-lightDir, normal);
    float       specular            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

    return vec3(mat.specular * specular);
}