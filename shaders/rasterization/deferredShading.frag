#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : enable

#include "../common.glsl"

layout(constant_id = 0) const uint maxLightCount = 100;
layout(constant_id = 1) const uint totalMaterialCount = 10000;

layout (push_constant) uniform PushConstants {
	vec3 cameraPos;
	int lightCount;
} pushConstants;

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;
// TODO populate and test ourColor and outVelocity for postprocessing
//layout(location = 1) out vec4 outColorCopy;
//layout(location = 2) out vec4 outVelocity;

// Scene bindings
layout(set = 0, binding = 0) uniform CurrentFrameCameraBuffer {
    mat4 view;
    mat4 proj;
} currentFrameCameraBuffer;

layout(std140, set = 0, binding = 1) uniform LightBuffer {
	LightData lights[maxLightCount];
} lightBuffer;

// Geometry buffer
layout (set = 1, binding = 0) uniform sampler2D samplerPosition;
layout (set = 1, binding = 1) uniform sampler2D samplerNormal;
layout (set = 1, binding = 2) uniform sampler2D samplerUV0;
layout (set = 1, binding = 3) uniform sampler2D samplerUV1;
layout (set = 1, binding = 4) uniform sampler2D samplerColor0;
layout (set = 1, binding = 5) uniform sampler2D samplerMaterialIndex;

// Texture samplers
layout (set = 2, binding = 0) uniform sampler2D colorMap[totalMaterialCount];
layout (set = 2, binding = 1) uniform sampler2D physicalDescriptorMap[totalMaterialCount];
layout (set = 2, binding = 2) uniform sampler2D normalMap[totalMaterialCount];
layout (set = 2, binding = 3) uniform sampler2D aoMap[totalMaterialCount];
layout (set = 2, binding = 4) uniform sampler2D emissiveMap[totalMaterialCount];

// Material buffer
layout(std430, set = 3, binding = 0) buffer SSBO
{
   GltfMaterial materials[];
};

/*
layout (set = 0, binding = 1) uniform UBOParams {
	vec4 lightDir;
	float exposure;
	float gamma;
	float prefilteredCubeMipLevels;
	float scaleIBLAmbient;
	float debugViewInputs;
	float debugViewEquation;
} uboParams;
*/

float exposure = 4.5;
float gamma = 2.2;
float debugViewInputs = 0.0;
float debugViewEquation = 0.0;

/*
layout (set = 0, binding = 2) uniform samplerCube samplerIrradiance;
layout (set = 0, binding = 3) uniform samplerCube prefilteredMap;
layout (set = 0, binding = 4) uniform sampler2D samplerBRDFLUT;
*/


// Encapsulate the various inputs used by the various functions in the shading equation
// We store values in this struct to simplify the integration of alternative implementations
// of the shading terms, outlined in the Readme.MD Appendix.
struct PBRInfo
{
	float NdotL;                  // cos angle between normal and light direction
	float NdotV;                  // cos angle between normal and view direction
	float NdotH;                  // cos angle between normal and half vector
	float LdotH;                  // cos angle between light direction and half vector
	float VdotH;                  // cos angle between view direction and half vector
	float perceptualRoughness;    // roughness value, as authored by the model creator (input to shader)
	float metalness;              // metallic value at the surface
	vec3 reflectance0;            // full reflectance color (normal incidence angle)
	vec3 reflectance90;           // reflectance color at grazing angle
	float alphaRoughness;         // roughness mapped to a more linear change in the roughness (proposed by [2])
	vec3 diffuseColor;            // color contribution from diffuse lighting
	vec3 specularColor;           // color contribution from specular lighting
};

const float M_PI = 3.141592653589793;
const float c_MinRoughness = 0.04;

const float PBR_WORKFLOW_METALLIC_ROUGHNESS = 0.0;
const float PBR_WORKFLOW_SPECULAR_GLOSINESS = 1.0f;

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
	#define MANUAL_SRGB 1
	#ifdef MANUAL_SRGB
	#ifdef SRGB_FAST_APPROXIMATION
	vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
	#else //SRGB_FAST_APPROXIMATION
	vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
	vec3 linOut = mix( srgbIn.xyz/vec3(12.92), pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
	#endif //SRGB_FAST_APPROXIMATION
	return vec4(linOut,srgbIn.w);;
	#else //MANUAL_SRGB
	return srgbIn;
	#endif //MANUAL_SRGB
}

vec3 Uncharted2Tonemap(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec4 tonemap(vec4 color)
{
	vec3 outcol = Uncharted2Tonemap(color.rgb * exposure);
	outcol = outcol * (1.0f / Uncharted2Tonemap(vec3(11.2f)));	
	return vec4(pow(outcol, vec3(1.0f / gamma)), color.a);
}

// Find the normal for this fragment, pulling either from a predefined normal map
// or from the interpolated mesh normal and tangent attributes.
vec3 getNormal(GltfMaterial material)
{
	// Get info from geometry buffer
	vec3 sampledPosition = texture(samplerPosition, inUV).rgb;
	vec3 sampledNormal = texture(samplerNormal, inUV).rgb;
	vec2 sampledUV0 = texture(samplerUV0, inUV).rg;
	vec2 sampledUV1 = texture(samplerUV1, inUV).rg;
	int sampledMaterialIndex = int(texture(samplerMaterialIndex, inUV).r);

	// Perturb normal, see http://www.thetenthplanet.de/archives/1180
	vec3 tangentNormal = texture(normalMap[sampledMaterialIndex], material.normalTextureSet == 0 ? sampledUV0 : sampledUV1).xyz * 2.0 - 1.0;

	vec3 q1 = dFdx(sampledPosition);
	vec3 q2 = dFdy(sampledPosition);
	vec2 st1 = dFdx(sampledUV0);
	vec2 st2 = dFdy(sampledUV0);

	vec3 N = normalize(sampledNormal);
	vec3 T = normalize(q1 * st2.t - q2 * st1.t);
	vec3 B = -normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	return normalize(TBN * tangentNormal);
}

/*
// Calculation of the lighting contribution from an optional Image Based Light source.
// Precomputed Environment Maps are required uniform inputs and are computed as outlined in [1].
// See our README.md on Environment Maps [3] for additional discussion.
vec3 getIBLContribution(PBRInfo pbrInputs, vec3 n, vec3 reflection)
{
	float lod = (pbrInputs.perceptualRoughness * prefilteredCubeMipLevels);
	// retrieve a scale and bias to F0. See [1], Figure 3
	vec3 brdf = (texture(samplerBRDFLUT, vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness))).rgb;
	vec3 diffuseLight = SRGBtoLINEAR(tonemap(texture(samplerIrradiance, n))).rgb;

	vec3 specularLight = SRGBtoLINEAR(tonemap(textureLod(prefilteredMap, reflection, lod))).rgb;

	vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
	vec3 specular = specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);

	// For presentation, this allows us to disable IBL terms
	// For presentation, this allows us to disable IBL terms
	diffuse *= scaleIBLAmbient;
	specular *= scaleIBLAmbient;

	return diffuse + specular;
}*/

// Basic Lambertian diffuse
// Implementation from Lambert's Photometria https://archive.org/details/lambertsphotome00lambgoog
// See also [1], Equation 1
vec3 diffuse(PBRInfo pbrInputs)
{
	return pbrInputs.diffuseColor / M_PI;
}

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 specularReflection(PBRInfo pbrInputs)
{
	return pbrInputs.reflectance0 + (pbrInputs.reflectance90 - pbrInputs.reflectance0) * pow(clamp(1.0 - pbrInputs.VdotH, 0.0, 1.0), 5.0);
}

// This calculates the specular geometric attenuation (aka G()),
// where rougher material will reflect less light back to the viewer.
// This implementation is based on [1] Equation 4, and we adopt their modifications to
// alphaRoughness as input as originally proposed in [2].
float geometricOcclusion(PBRInfo pbrInputs)
{
	float NdotL = pbrInputs.NdotL;
	float NdotV = pbrInputs.NdotV;
	float r = pbrInputs.alphaRoughness;

	float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
	float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
	return attenuationL * attenuationV;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float microfacetDistribution(PBRInfo pbrInputs)
{
	float roughnessSq = pbrInputs.alphaRoughness * pbrInputs.alphaRoughness;
	float f = (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH + 1.0;
	return roughnessSq / (M_PI * f * f);
}

// Gets metallic factor from specular glossiness workflow inputs 
float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular) {
	float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
	if (perceivedSpecular < c_MinRoughness) {
		return 0.0;
	}
	float a = c_MinRoughness;
	float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
	float c = c_MinRoughness - perceivedSpecular;
	float D = max(b * b - 4.0 * a * c, 0.0);
	return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

void main()
{
	// Get info from geometry buffer
	vec3 sampledPosition = texture(samplerPosition, inUV).rgb;
	vec3 sampledNormal = texture(samplerNormal, inUV).rgb;
	vec2 sampledUV0 = texture(samplerUV0, inUV).rg;
	vec2 sampledUV1 = texture(samplerUV1, inUV).rg;
	vec4 sampledColor0 = texture(samplerColor0, inUV).rgba;
	int sampledMaterialIndex = int(texture(samplerMaterialIndex, inUV).r);

	//if (sampledMaterialIndex != 0) debugPrintfEXT("SampledMaterialIndex is %f, %f, the int value of .r is %d", texture(samplerMaterialIndex, inUV).r, texture(samplerMaterialIndex, inUV).g, sampledMaterialIndex);
	//vec2 texCoords = gl_FragCoord.xy / vec2(1280, 720); // Same value as inUV
	//debugPrintfEXT("inuv is (%f, %f), texCoords is (%f, %f)", inUV.x, inUV.y, texCoords.x, texCoords.y);

	GltfMaterial material = materials[sampledMaterialIndex];

	float perceptualRoughness;
	float metallic;
	vec3 diffuseColor;
	vec4 baseColor;
	vec3 color = vec3(0.0);

	// Overriden each loop but are put here to debug
	vec3 diffuseContrib;
	vec3 specContrib;
	vec3 F;
	float G;
	float D;

	vec3 f0 = vec3(0.04);
	for (int lightIndex = 0; lightIndex < pushConstants.lightCount; ++lightIndex)
	{
		if (material.alphaMask == 1.0f) {
			if (material.baseColorTextureSet > -1) {
				baseColor = SRGBtoLINEAR(texture(colorMap[sampledMaterialIndex], material.baseColorTextureSet == 0 ? sampledUV0 : sampledUV1)) * material.baseColorFactor;
			} else {
				baseColor = material.baseColorFactor;
			}
			if (baseColor.a < material.alphaMaskCutoff) {
				discard;
			}
		}

		if (material.workflow == PBR_WORKFLOW_METALLIC_ROUGHNESS) {
			// Metallic and Roughness material properties are packed together
			// In glTF, these factors can be specified by fixed scalar values
			// or from a metallic-roughness map
			perceptualRoughness = material.roughnessFactor;
			metallic = material.metallicFactor;
			if (material.physicalDescriptorTextureSet > -1) {
				// Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
				// This layout intentionally reserves the 'r' channel for (optional) occlusion map data
				vec4 mrSample = texture(physicalDescriptorMap[sampledMaterialIndex], material.physicalDescriptorTextureSet == 0 ? sampledUV0 : sampledUV1);
				perceptualRoughness = mrSample.g * perceptualRoughness;
				metallic = mrSample.b * metallic;
			} else {
				perceptualRoughness = clamp(perceptualRoughness, c_MinRoughness, 1.0);
				metallic = clamp(metallic, 0.0, 1.0);
			}
			// Roughness is authored as perceptual roughness; as is convention,
			// convert to material roughness by squaring the perceptual roughness [2].

			// The albedo may be defined from a base texture or a flat color
			if (material.baseColorTextureSet > -1) {
				baseColor = SRGBtoLINEAR(texture(colorMap[sampledMaterialIndex], material.baseColorTextureSet == 0 ? sampledUV0 : sampledUV1)) * material.baseColorFactor;
			} else {
				baseColor = material.baseColorFactor;
			}
		}

		if (material.workflow == PBR_WORKFLOW_SPECULAR_GLOSINESS) {
			// Values from specular glossiness workflow are converted to metallic roughness
			if (material.physicalDescriptorTextureSet > -1) {
				perceptualRoughness = 1.0 - texture(physicalDescriptorMap[sampledMaterialIndex], material.physicalDescriptorTextureSet == 0 ? sampledUV0 : sampledUV1).a;
			} else {
				perceptualRoughness = 0.0;
			}

			const float epsilon = 1e-6;

			vec4 diffuse = SRGBtoLINEAR(texture(colorMap[sampledMaterialIndex], sampledUV0));
			vec3 specular = SRGBtoLINEAR(texture(physicalDescriptorMap[sampledMaterialIndex], sampledUV0)).rgb;

			float maxSpecular = max(max(specular.r, specular.g), specular.b);

			// Convert metallic value from specular glossiness inputs
			metallic = convertMetallic(diffuse.rgb, specular, maxSpecular);

			vec3 baseColorDiffusePart = diffuse.rgb * ((1.0 - maxSpecular) / (1 - c_MinRoughness) / max(1 - metallic, epsilon)) * material.diffuseFactor.rgb;
			vec3 baseColorSpecularPart = specular - (vec3(c_MinRoughness) * (1 - metallic) * (1 / max(metallic, epsilon))) * material.specularFactor.rgb;
			baseColor = vec4(mix(baseColorDiffusePart, baseColorSpecularPart, metallic * metallic), diffuse.a);

		}

		baseColor *= sampledColor0;

		diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
		diffuseColor *= 1.0 - metallic;
		
		float alphaRoughness = perceptualRoughness * perceptualRoughness;

		vec3 specularColor = mix(f0, baseColor.rgb, metallic);

		// Compute reflectance.
		float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);

		// For typical incident reflectance range (between 4% to 100%) set the grazing reflectance to 100% for typical fresnel effect.
		// For very low reflectance range on highly diffuse objects (below 4%), incrementally reduce grazing reflecance to 0%.
		float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
		vec3 specularEnvironmentR0 = specularColor.rgb;
		vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;

		vec3 n = (material.normalTextureSet > -1) ? getNormal(material) : normalize(sampledNormal);
		vec3 v = normalize(pushConstants.cameraPos - sampledPosition);    // Vector from surface point to camera
		vec3 l;     // Vector from surface point to light

		float lightIntensity = lightBuffer.lights[lightIndex].intensity;
		if (lightBuffer.lights[lightIndex].type == 0)
        {
            vec3  lightDir = lightBuffer.lights[lightIndex].position - sampledPosition;
            float d        = length(lightDir);
            lightIntensity = lightBuffer.lights[lightIndex].intensity / (d * d);
            l              = normalize(lightDir);
        }
        else if (lightBuffer.lights[lightIndex].type == 1)
        {   
            vec3 lightDir = vec3(
		        sin(radians(lightBuffer.lights[lightIndex].rotation.x)) * cos(radians(lightBuffer.lights[lightIndex].rotation.y)),
		        sin(radians(lightBuffer.lights[lightIndex].rotation.y)),
		        cos(radians(lightBuffer.lights[lightIndex].rotation.x)) * cos(radians(lightBuffer.lights[lightIndex].rotation.y))
	        );
            l = normalize(lightDir);
            lightIntensity *= 0.01;
        }

		vec3 h = normalize(l+v);                        // Half vector between both l and v
		vec3 reflection = -normalize(reflect(v, n));
		reflection.y *= -1.0f;

		float NdotL = clamp(dot(n, l), 0.001, 1.0);
		float NdotV = clamp(abs(dot(n, v)), 0.001, 1.0);
		float NdotH = clamp(dot(n, h), 0.0, 1.0);
		float LdotH = clamp(dot(l, h), 0.0, 1.0);
		float VdotH = clamp(dot(v, h), 0.0, 1.0);

		PBRInfo pbrInputs = PBRInfo(
			NdotL,
			NdotV,
			NdotH,
			LdotH,
			VdotH,
			perceptualRoughness,
			metallic,
			specularEnvironmentR0,
			specularEnvironmentR90,
			alphaRoughness,
			diffuseColor,
			specularColor
		);

		// Calculate the shading terms for the microfacet specular shading model
		F = specularReflection(pbrInputs);
		G = geometricOcclusion(pbrInputs);
		D = microfacetDistribution(pbrInputs);

		// Calculation of analytical lighting contribution
		diffuseContrib = (1.0 - F) * diffuse(pbrInputs);
		specContrib = F * G * D / (4.0 * NdotL * NdotV);
		// Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
		color += NdotL * lightIntensity * lightBuffer.lights[lightIndex].color * (diffuseContrib + specContrib);

		// Calculate lighting contribution from image based lighting source (IBL)
		//color += getIBLContribution(pbrInputs, n, reflection);

		const float u_OcclusionStrength = 1.0f;
		// Apply optional PBR terms for additional (optional) shading
		if (material.occlusionTextureSet > -1) {
			float ao = texture(aoMap[sampledMaterialIndex], (material.occlusionTextureSet == 0 ? sampledUV0 : sampledUV1)).r;
			color = mix(color, color * ao, u_OcclusionStrength);
		}

		vec3 emissive = material.emissiveFactor.rgb * material.emissiveStrength;
		if (material.emissiveTextureSet > -1) {
			emissive *= SRGBtoLINEAR(texture(emissiveMap[sampledMaterialIndex], material.emissiveTextureSet == 0 ? sampledUV0 : sampledUV1)).rgb;
		};
		color += emissive;
	}
	
	float ambientFactor = 3.0;
	outColor = vec4(color * ambientFactor, baseColor.a);

	// Shader inputs debug visualization
	if (debugViewInputs > 0.0) {
		int index = int(debugViewInputs);
		switch (index) {
			case 1:
				outColor.rgba = material.baseColorTextureSet > -1 ? texture(colorMap[sampledMaterialIndex], material.baseColorTextureSet == 0 ? sampledUV0 : sampledUV1) : vec4(1.0f);
				break;
			case 2:
				outColor.rgb = (material.normalTextureSet > -1) ? texture(normalMap[sampledMaterialIndex], material.normalTextureSet == 0 ? sampledUV0 : sampledUV1).rgb : normalize(sampledNormal);
				break;
			case 3:
				outColor.rgb = (material.occlusionTextureSet > -1) ? texture(aoMap[sampledMaterialIndex], material.occlusionTextureSet == 0 ? sampledUV0 : sampledUV1).rrr : vec3(0.0f);
				break;
			case 4:
				outColor.rgb = (material.emissiveTextureSet > -1) ? texture(emissiveMap[sampledMaterialIndex], material.emissiveTextureSet == 0 ? sampledUV0 : sampledUV1).rgb : vec3(0.0f);
				break;
			case 5:
				outColor.rgb = texture(physicalDescriptorMap[sampledMaterialIndex], sampledUV0).bbb;
				break;
			case 6:
				outColor.rgb = texture(physicalDescriptorMap[sampledMaterialIndex], sampledUV0).ggg;
				break;
		}
		outColor = SRGBtoLINEAR(outColor);
	}

	// PBR equation debug visualization
	// "none", "Diff (l,n)", "F (l,h)", "G (l,v,h)", "D (h)", "Specular"
	if (debugViewEquation > 0.0) {
		int index = int(debugViewEquation);
		switch (index) {
			case 1:
				outColor.rgb = diffuseContrib;
				break;
			case 2:
				outColor.rgb = F;
				break;
			case 3:
				outColor.rgb = vec3(G);
				break;
			case 4: 
				outColor.rgb = vec3(D);
				break;
			case 5:
				outColor.rgb = specContrib;
				break;				
		}
	}
}
