#version 460

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 3, rgba8_snorm) uniform image2D historyImage;

void main() {
	// Load the contents of the history buffer which will then be blended with the existing output image according to the parameters set in the pipeline
    outColor = imageLoad(historyImage, ivec2(gl_FragCoord.xy));
	//outColor = vec4(0.2f, 0.2f, 0.2f, 1.0f);
}