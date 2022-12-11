# Vulkr

A Vulkan rendering engine that I've taken upon building in order to solidify my knowledge of Vulkan and graphics programming as a whole by implementing interesting graphics features and applications.

The engine runs on `Vulkan 1.3` using `VK_KHR_synchronization2` and the code is written for the `C++20` standard. This project is being built on `Windows 10, x64` using `Visual Studio 2022` and as such, can not be expected to work on Linux or MacOS. However, all third-party libraries used are cross platform and platform-specific components like the windowing system could be easily supported in the future. 

https://user-images.githubusercontent.com/18451835/173965410-b90e12f6-1137-4c0d-bb6b-b521f90e2a7d.mp4

## Build
Clone the project with the `--recurse-submodules` option to grab all the necessary third party dependencies.

With CMake installed, running the following commands from the root folder level will build the project.
```
cmake -S . -B build
cmake --build build
```
For optimal performance, ensure that the `VULKR_DEBUG` flag is not defined in `vulkan_common.h`. If you wish to debug the engine using RenderDoc, be sure to enable the `RENDERDOC_DEBUG` flag also located in `vulkan_common.h` so that unsupported extensions like `VK_KHR_ray_tracing` are disabled to prevent RenderDoc from complaining about them. NSight Systems and NSight Graphics should work without this flag.

## Features
The Vulkr renderer is able to accept wavefront object and material files (*.obj + *.mtl) and render them in a 3D environment. Some core features are:
- Choosing between a traditional rasterization pipeline (forward rendering) and raytracing (VK_KHR_ray_tracing) as rendering methods (you can switch between them at runtime)
- Triple buffering (pending GPU exposes atleast 3 swapchain images) with two sets of frame resources
- Instanced object rendering
- Navigating within the world using mouse controls you would see on typical graphics software like Blender
- Multiple global light sources that can be set as a directional (infinite) or point light with customizable intensity and position values
- Temporal Anti-Aliasing for static and dynamic scenes (rasterization only so far); implemented using techniques mentioned in the "[A Survey of Temporal Antialising Techniques](http://behindthepixels.io/assets/files/TemporalAA.pdf)" white paper and [Ziyad Barakat's TAA Blog Post](https://ziyadbarakat.wordpress.com/2020/07/28/temporal-anti-aliasing-step-by-step/)
- A imgui debug window that provides access to altering the camera and light settings
- Shadows (ray tracing only so far)
- A particle system simulation

https://user-images.githubusercontent.com/18451835/173991616-5825f922-1a45-4556-9cf3-51c6615e918b.mp4

- A 2D fluid simulator that implements velocity & density advection, gaussian splat on mouse click + drag, and a projection steps that solve for the poisson pressure equation and subtracts the gradient of the pressure from the velocity field; implemented using the techniques mentioned in [Mark Harris' "Fluid Dynamics Simulation on the GPU" GPU Gems article](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu) (`#define FLUID_SIMULATION` should enable the simulation)

https://user-images.githubusercontent.com/18451835/206916607-4497a86a-4377-497f-a8b3-40adfadd5b77.mp4


## Credits
A special thanks to `Alexander Overvoorde's` [Vulkan Tutorial](https://vulkan-tutorial.com/) for providing a great introduction to Vulkan as a whole and providing the base knowledge required to get started on this project. Additionally, `Sascha Willems'` [Vulkan Demos](https://github.com/SaschaWillems/Vulkan), [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) provided by the `Khronos Group` and the [Vulkan Ray Tracing Samples](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR) provided by `NVIDIA` have played an instrumental role into providing guidance into best practices, techniques and code samples for implementing various features using Vulkan. Finally, the discord and reddit communities on Vulkan and Graphics Programming have been very helpful with answering many of the questions that I've gotten along the way!
