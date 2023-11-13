/**
 * Vulkan glTF model and texture loading class based on tinyglTF (https://github.com/syoyo/tinygltf)
 *
 * Copyright (C) 2018-2022 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 * 
 * Note that this file has been modified to integrate into the Vulkr engine, but the core logic remains the same and is covered under the license above.
 */

#pragma once

#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>

#include "vulkan_common.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ERROR is already defined in wingdi.h and collides with a define in the Draco headers
#if defined(_WIN32) && defined(ERROR) && defined(TINYGLTF_ENABLE_DRACO) 
#undef ERROR
#pragma message ("ERROR constant already defined, undefining")
#endif

#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "tiny_gltf.h"

// Changing this value here also requires changing it in the vertex shader
#define MAX_NUM_JOINTS 128u

namespace vulkr
{
// Forward declarations
class Device;
class CommandPool;
class Buffer;
class Image;
}

namespace vulkr::gltf
{

struct Node;

struct BoundingBox
{
	glm::vec3 min;
	glm::vec3 max;
	bool valid = false;
	BoundingBox();
	BoundingBox(glm::vec3 min, glm::vec3 max);
	BoundingBox getAABB(glm::mat4 m);
};

struct TextureSampler
{
	VkFilter magFilter;
	VkFilter minFilter;
	VkSamplerAddressMode addressModeU;
	VkSamplerAddressMode addressModeV;
	VkSamplerAddressMode addressModeW;
};

struct Texture
{
	vulkr::Device *device;
	VkImage image;
	VmaAllocation allocation{ VK_NULL_HANDLE };
	VkImageLayout imageLayout;
	VkImageView view;
	uint32_t width, height;
	uint32_t mipLevels;
	uint32_t layerCount;
	VkDescriptorImageInfo descriptor;
	VkSampler sampler;
	void updateDescriptor();
	void destroy();
	// Load a texture from a glTF image (stored as vector of chars loaded via stb_image) and generate a full mip chaing for it
	void fromglTfImage(tinygltf::Image &gltfimage, TextureSampler textureSampler, vulkr::Device *device,  vulkr::CommandPool *commandPool, VkQueue copyQueue);
};

struct Material
{
	enum AlphaMode { ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
	AlphaMode alphaMode = ALPHAMODE_OPAQUE;
	float alphaCutoff = 1.0f;
	float metallicFactor = 1.0f;
	float roughnessFactor = 1.0f;
	glm::vec4 baseColorFactor = glm::vec4(1.0f);
	glm::vec4 emissiveFactor = glm::vec4(0.0f);
	Texture *baseColorTexture;
	Texture *metallicRoughnessTexture;
	Texture *normalTexture;
	Texture *occlusionTexture;
	Texture *emissiveTexture;
	bool doubleSided = false;
	struct TexCoordSets {
		uint8_t baseColor = 0;
		uint8_t metallicRoughness = 0;
		uint8_t specularGlossiness = 0;
		uint8_t normal = 0;
		uint8_t occlusion = 0;
		uint8_t emissive = 0;
	} texCoordSets;
	struct Extension {
		Texture *specularGlossinessTexture;
		Texture *diffuseTexture;
		glm::vec4 diffuseFactor = glm::vec4(1.0f);
		glm::vec3 specularFactor = glm::vec3(0.0f);
	} extension;
	struct PbrWorkflows {
		bool metallicRoughness = true;
		bool specularGlossiness = false;
	} pbrWorkflows;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
	int index = 0;
	bool unlit = false;
	float emissiveStrength = 1.0f;
};

struct Primitive
{
	uint32_t firstIndex;
	uint32_t indexCount;
	uint32_t vertexCount;
	Material &material;
	bool hasIndices;
	BoundingBox bb;
	Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material &material);
	void setBoundingBox(glm::vec3 min, glm::vec3 max);
};

struct Mesh
{
	Device *device;
	std::vector<Primitive *> primitives;
	BoundingBox bb;
	BoundingBox aabb;
	struct UniformBuffer {
		std::unique_ptr<vulkr::Buffer> buffer;
		VkDescriptorBufferInfo descriptor;
		VkDescriptorSet descriptorSet; // Will be filled in in the vulkr.cpp code
	} uniformBuffer;
	struct UniformBlock {
		glm::mat4 matrix;
		glm::mat4 jointMatrix[MAX_NUM_JOINTS]{};
		float jointcount{ 0 };
	} uniformBlock;
	Mesh(Device *device, glm::mat4 matrix);
	~Mesh();
	void setBoundingBox(glm::vec3 min, glm::vec3 max);
};

struct Skin
{
	std::string name;
	Node *skeletonRoot = nullptr;
	std::vector<glm::mat4> inverseBindMatrices;
	std::vector<Node *> joints;
};

struct Node
{
	Node *parent;
	uint32_t index;
	std::vector<Node *> children;
	glm::mat4 matrix;
	std::string name;
	Mesh *mesh;
	Skin *skin;
	int32_t skinIndex = -1;
	glm::vec3 translation{};
	glm::vec3 scale{ 1.0f };
	glm::quat rotation{};
	BoundingBox bvh;
	BoundingBox aabb;
	glm::mat4 localMatrix();
	glm::mat4 getMatrix();
	void update();
	~Node();
};

struct AnimationChannel
{
	enum PathType { TRANSLATION, ROTATION, SCALE };
	PathType path;
	Node *node;
	uint32_t samplerIndex;
};

struct AnimationSampler
{
	enum InterpolationType { LINEAR, STEP, CUBICSPLINE };
	InterpolationType interpolation;
	std::vector<float> inputs;
	std::vector<glm::vec4> outputsVec4;
};

struct Animation
{
	std::string name;
	std::vector<AnimationSampler> samplers;
	std::vector<AnimationChannel> channels;
	float start = std::numeric_limits<float>::max();
	float end = std::numeric_limits<float>::min();
};

struct Model
{
	Device *device;

	struct Vertex
	{
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv0;
		glm::vec2 uv1;
		glm::vec4 joint0;
		glm::vec4 weight0;
		glm::vec4 color;
	};

	std::unique_ptr<vulkr::Buffer> vertexBuffer;
	std::unique_ptr<vulkr::Buffer> indexBuffer;

	glm::mat4 aabb;

	std::vector<Node *> nodes;
	std::vector<Node *> linearNodes;

	std::vector<Skin *> skins;

	std::vector<Texture> textures;
	std::vector<TextureSampler> textureSamplers;
	std::vector<Material> materials;
	std::vector<Animation> animations;
	std::vector<std::string> extensions;

	struct Dimensions
	{
		glm::vec3 min = glm::vec3(FLT_MAX);
		glm::vec3 max = glm::vec3(-FLT_MAX);
	} dimensions;

	struct LoaderInfo
	{
		uint32_t *indexBuffer;
		Vertex *vertexBuffer;
		size_t indexPos = 0;
		size_t vertexPos = 0;
	};

	void destroy(Device *device);
	void loadNode(Node *parent, const tinygltf::Node &node, uint32_t nodeIndex, const tinygltf::Model &model, LoaderInfo &loaderInfo, float globalscale);
	void getNodeProps(const tinygltf::Node &node, const tinygltf::Model &model, size_t &vertexCount, size_t &indexCount);
	void loadSkins(tinygltf::Model &gltfModel);
	void loadTextures(tinygltf::Model &gltfModel, vulkr::Device *device, vulkr::CommandPool *commandPool, VkQueue transferQueue);
	VkSamplerAddressMode getVkWrapMode(int32_t wrapMode);
	VkFilter getVkFilterMode(int32_t filterMode);
	void loadTextureSamplers(tinygltf::Model &gltfModel);
	void loadMaterials(tinygltf::Model &gltfModel);
	void loadAnimations(tinygltf::Model &gltfModel);
	void loadFromFile(std::string filename, vulkr::Device *device, vulkr::CommandPool *commandPool, VkQueue transferQueue, float scale = 1.0f);
	void drawNode(Node *node, VkCommandBuffer commandBuffer);
	void draw(VkCommandBuffer commandBuffer);
	void calculateBoundingBox(Node *node, Node *parent);
	void getSceneDimensions();
	void updateAnimation(uint32_t index, float time);
	Node *findNode(Node *parent, uint32_t index);
	Node *nodeFromIndex(uint32_t index);
};

} // vulkr