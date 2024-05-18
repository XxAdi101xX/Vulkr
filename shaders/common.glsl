#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct Vertex
{
    vec3 position;
    int padding1;
    vec3 normal;
    int padding2;
    vec3 color;
    int padding3;
    vec2 textureCoordinate;
    vec2 padding4;
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

struct GltfInstance
{
	mat4 transform;           // Model transform
    uint64_t modelIndex;      // Object model index
    uint64_t blank;
};

struct WaveFrontMaterial
{
    vec3  ambient;
    float shininess;
    vec3  diffuse;
    float ior;       // index of refraction
    vec3  specular;
    float dissolve;  // 1 == opaque; 0 == fully transparent
    vec3  transmittance;
    int   illum;     // illumination model (see http://www.fileformat.info/format/material/)
    vec3  emission;
    int   textureId;
};

struct Particle
{
	vec4 position; // xyz = position, w = mass
	vec4 velocity; // xyz = velocity, w = gradient texture position
};

struct LightData
{
    vec3 position; // used for point light calculation
    float intensity;
    vec3 color;
    int type; // 0: point, 1: directional (infinite)
    vec2 rotation; // Used for directional lights; represents horizontal (azimuth) and vertical (elevation) rotation
    vec2 blank; // padding
};

struct GltfMaterial
{
	vec4 baseColorFactor;
	vec4 emissiveFactor;
	vec4 diffuseFactor;
	vec4 specularFactor;
	float workflow;
	int baseColorTextureSet;
	int physicalDescriptorTextureSet;
	int normalTextureSet;	
	int occlusionTextureSet;
	int emissiveTextureSet;
	float metallicFactor;	
	float roughnessFactor;	
	float alphaMask;	
	float alphaMaskCutoff;
	float emissiveStrength;
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