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

// Trowbridge-Reitz (GGX) Normal Distribution Function (NDF)
float D_TrowbridgeReitzGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265 * denom * denom;
    return a2 / denom;
}

// Geometry shadowing term (Schlick-GGX approximation)
float G_SchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float denom = NdotV * (1.0 - k) + k;
    return NdotV / denom;
}

// Smith geometry attenuation function combining light and view terms
float G_Smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = G_SchlickGGX(NdotV, roughness);
    float ggx2 = G_SchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Fresnel reflectance term using Schlick's approximation
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Cook-Torrance specular microfacet BRDF implementation
vec3 computeSpecularCookTorrance(WaveFrontMaterial mat, vec3 viewDir, vec3 lightDir, vec3 normal) {
    if (mat.illum < 2) {
        return vec3(0.0);
    }

    vec3 V = normalize(-viewDir);
    vec3 L = normalize(lightDir);
    vec3 N = normalize(normal);
    vec3 H = normalize(V + L);

    // Map shininess [4,256] to roughness [1.0, 0.05]
    float roughness = clamp(1.0 - (mat.shininess - 4.0) / 252.0, 0.05, 1.0);

    vec3 F0 = mat.specular;

    float D = D_TrowbridgeReitzGGX(N, H, roughness);
    float G = G_Smith(N, V, L, roughness);
    vec3 F = F_Schlick(max(dot(H, V), 0.0), F0);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.001; // prevent division by zero

    vec3 specular = numerator / denominator;

    return specular * NdotL;
}

// Blinn-Phong specular BRDF component
vec3 computeSpecularBlinnPhong(WaveFrontMaterial mat, vec3 viewDir, vec3 lightDir, vec3 normal)
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

// Lambertian diffuse BRDF component 
vec3 computeDiffuseLambertian(WaveFrontMaterial mat, vec3 lightDir, vec3 normal)
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

