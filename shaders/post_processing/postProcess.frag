#version 460

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 2, rgba8_snorm) uniform image2D historyImage;
layout(set = 0, binding = 3, rgba8_snorm) uniform image2D velocityImage;
layout(set = 0, binding = 4, rgba8_snorm) uniform image2D copyOutputImage;

layout(push_constant) uniform PostProcessPushConstant
{
    vec2 imageExtent;
} pushConstant;

void main() {
	vec2 velocity = imageLoad(velocityImage, ivec2(gl_FragCoord.xy)).xy;
	vec2 previousPixelPos = gl_FragCoord.xy - velocity;

	vec4 minColor = imageLoad(copyOutputImage, ivec2(gl_FragCoord.xy));
	vec4 maxColor = imageLoad(copyOutputImage, ivec2(gl_FragCoord.xy));

	// Check top, bot, left and right colour values to clamp against
	if (gl_FragCoord.x > 0)
	{
		// Left pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x - 1, gl_FragCoord.y)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x - 1, gl_FragCoord.y)));
	}
	if (gl_FragCoord.x < pushConstant.imageExtent.x - 1)
	{
		// Right pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x + 1, gl_FragCoord.y)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x + 1, gl_FragCoord.y)));
	}
	if (gl_FragCoord.y > 0)
	{
		// Top pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x, gl_FragCoord.y - 1)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x, gl_FragCoord.y - 1)));
	}
	if (gl_FragCoord.y < pushConstant.imageExtent.y - 1)
	{
		// Bottom pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x, gl_FragCoord.y + 1)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x, gl_FragCoord.y + 1)));
	}

	// Check against the diagnoal pixels to clamp colour against, completing the 3x3 neighbourhood kernel check
	if (gl_FragCoord.x > 0 && gl_FragCoord.y > 0)
	{
		// Top left pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x - 1, gl_FragCoord.y - 1)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x - 1, gl_FragCoord.y - 1)));
		// Top right pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x + 1, gl_FragCoord.y - 1)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x + 1, gl_FragCoord.y - 1)));
	}
	if (gl_FragCoord.x < pushConstant.imageExtent.x - 1 && gl_FragCoord.y < pushConstant.imageExtent.y - 1)
	{
		// Bottom left pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x - 1, gl_FragCoord.y + 1)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x - 1, gl_FragCoord.y + 1)));
		// Bottom right pixel
		minColor = min(minColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x + 1, gl_FragCoord.y + 1)));
		maxColor = max(maxColor, imageLoad(copyOutputImage, ivec2(gl_FragCoord.x + 1, gl_FragCoord.y + 1)));
	}

	// Load the contents of the history buffer (clamped to prevent ghosting artifacts) which will then be blended with the existing output image according to the parameters set in the pipeline
    outColor = clamp(imageLoad(historyImage, ivec2(previousPixelPos)), minColor, maxColor);
}