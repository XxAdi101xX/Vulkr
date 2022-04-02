# Vulkr

Vulkr is a personal Vulkan renderer project that I've taken up in order to solidify my knowledge of Vulkan and graphics programming as a whole.

## Build
Clone the project with the `--recurse-submodules` option to grab all the necessary third party dependencies.

CMake is the build tool of choice for this project running the following commands from the root folder level will build the project.
```
cmake -S . -B build
cmake --build build
```

## Features
The Vulkr renderer is able to accept wavefront object and material files (*.obj + *.mtl) and render them in a 3D environment. Some features are:
- Choosing between a traditional rasterization pipeline (forward rendering) and raytracing (VK_KHR_ray_tracing) as rendering methods (you can switch between them at runtime)
- Triple buffering (pending GPU exposes atleast 3 swapchain images) with two sets of frame resources
- Instanced object rendering
- Navigating within the world using mouse controls you would see on typical graphics software like Blender
- A single global light source that can be set as a directional (infinite) or point light with customizable intensity and position values
- Temporal Anti-Aliasing for static and dynamic scenes (rasterization only so far); implemented using techniques mentioned in the "[A Survey of Temporal Antialising Techniques](http://behindthepixels.io/assets/files/TemporalAA.pdf)" white paper and [Ziyad Barakat's TAA Blog Post](https://ziyadbarakat.wordpress.com/2020/07/28/temporal-anti-aliasing-step-by-step/)
- A imgui debug window that provides access to altering the camera and light settings
- Shadows (ray tracing only so far)

## Credits
A special thanks to Alexander Overvoorde's [Vulkan Tutorial](https://vulkan-tutorial.com/) for providing a great introduction to Vulkan as a whole and providing the base knowledge required to get started on this project. Additionally, Sascha Willems' [Vulkan Demos](https://github.com/SaschaWillems/Vulkan), [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) provided by the KhronosGroup and the [Vulkan Ray Tracing Samples](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR) provided by NVIDIA have played an instrumental role into providing guidance into best practices, techniques and code samples for implementing various features using Vulkan. Finally, the discord and reddit communities on Vulkan and Graphics Programming have been very helpful with answering many of the questions that I've gotten along the way!
