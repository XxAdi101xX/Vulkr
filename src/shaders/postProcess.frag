#version 460

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 2, rgba8_snorm) uniform image2D historyImage;
layout(set = 0, binding = 3, rgba8_snorm) uniform image2D velocityImage;

void main() {
	vec2 velocity = imageLoad(velocityImage, ivec2(gl_FragCoord.xy)).xy;
	vec2 previousPixelPos = gl_FragCoord.xy - velocity;
	// Load the contents of the history buffer which will then be blended with the existing output image according to the parameters set in the pipeline
    //outColor = imageLoad(historyImage, ivec2(gl_FragCoord.xy));
	// enable this for dynamic TAA
    outColor = imageLoad(historyImage, ivec2(previousPixelPos));
	//outColor = vec4(0.2f, 0.2f, 0.2f, 1.0f);
}