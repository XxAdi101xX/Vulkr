#version 460

// Temporal anti-aliasing resolve.
//
// The scene color is sampled from the pre-resolve color image, while the previous frame's resolved color is
// retrieved from the history image using the per-pixel motion vector. The history is validated against the
// current frame's local neighborhood before being blended in. The result is written to the post process
// attachment; the history image is copied from that attachment before ImGui is rendered on top of it.

layout(set = 0, binding = 0) uniform sampler2D historyImage;
layout(set = 0, binding = 1) uniform sampler2D velocityImage;
layout(set = 0, binding = 2) uniform sampler2D sceneColorImage;

layout(push_constant) uniform PostProcessPushConstant
{
	vec2 imageExtent;     // Render target size in pixels
	vec2 currentJitter;   // This frame's sub-pixel jitter in pixels (x right, y down)
	vec2 previousJitter;  // Jitter that was used to render the history image, in pixels (x right, y down)
	float historyWeight;  // Weight of the history sample; 0 disables temporal accumulation
	float blank;          // Alignment
} pushConstant;

layout(location = 0) out vec4 outColor;

void main()
{
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	ivec2 maxPixel = ivec2(pushConstant.imageExtent) - 1;

	vec3 currentColor = texelFetch(sceneColorImage, clamp(pixel, ivec2(0), maxPixel), 0).rgb;

	// Clamp the history against the current frame's 3x3 neighborhood so stale colors from disocclusions and
	// moving geometry cannot bleed into the resolved image
	vec3 minColor = currentColor;
	vec3 maxColor = currentColor;
	for (int y = -1; y <= 1; ++y)
	{
		for (int x = -1; x <= 1; ++x)
		{
			ivec2 sampleCoord = clamp(pixel + ivec2(x, y), ivec2(0), maxPixel);
			vec3 sampleColor = texelFetch(sceneColorImage, sampleCoord, 0).rgb;
			minColor = min(minColor, sampleColor);
			maxColor = max(maxColor, sampleColor);
		}
	}

	vec3 resolvedColor = currentColor;
	if (pushConstant.historyWeight > 0.0)
	{
		// Velocity is stored in normalized screen space with the y axis pointing down, so scaling it by the
		// render target size gives the surface motion in pixels. Subtracting it from the current pixel walks
		// back to where this surface point was in the previous frame; the jitter delta accounts for the
		// history image having been rendered with a different sub-pixel offset.
		vec2 velocity = texelFetch(velocityImage, clamp(pixel, ivec2(0), maxPixel), 0).xy * pushConstant.imageExtent;
		if (!(dot(velocity, velocity) < 1.0e12)) // Guard against unpopulated velocity targets
		{
			velocity = vec2(0.0);
		}

		vec2 previousPixel = gl_FragCoord.xy - velocity + (pushConstant.previousJitter - pushConstant.currentJitter);
		previousPixel = clamp(previousPixel, vec2(0.5), vec2(pushConstant.imageExtent) - 0.5);
		vec2 historyUv = previousPixel / pushConstant.imageExtent;

		vec3 historyColor = clamp(texture(historyImage, historyUv).rgb, minColor, maxColor);
		resolvedColor = mix(currentColor, historyColor, pushConstant.historyWeight);
	}

	outColor = vec4(resolvedColor, 1.0);
}
