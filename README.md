# Vulkr

A Vulkan rendering engine that I've taken upon building in order to solidify my knowledge of Vulkan and graphics programming as a whole by implementing interesting graphics features and applications.

The engine runs on `Vulkan 1.3` using `VK_KHR_synchronization2` and the code is written for the `C++20` standard.

https://user-images.githubusercontent.com/18451835/173965410-b90e12f6-1137-4c0d-bb6b-b521f90e2a7d.mp4

## Features
The Vulkr renderer is able to accept GLTF (.gltf or glb) or OBJ asset formats and render them in a 3D environment. Some core features are:
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

- A 2D fluid simulator that implements velocity & density advection, gaussian splat on mouse click + drag, and a projection steps that solve for the poisson pressure equation and subtracts the gradient of the pressure from the velocity field; implemented using the techniques mentioned in [Mark Harris' "Fluid Dynamics Simulation on the GPU" GPU Gems article](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)

https://user-images.githubusercontent.com/18451835/206916607-4497a86a-4377-497f-a8b3-40adfadd5b77.mp4

## Build
This project requires the latest `Vulkan v1.3` SDK and the latest `CMake` installed (v3.10 or above) installed on the host's computer. All other third-party dependencies should be imported when cloning the project with the `--recurse-submodules` option. I developed Vulkr running `Windows 10, x64` using `Visual Studio 2022` and as such, an not be expected to work on Linux or MacOS. However, all third-party libraries used are cross platform and OS platform specific components like the windowing system could be supported in the future. 

The easiest way to run the project is to load it using `Visual Studio 2022` and building the project. You could then select any application executable file (`vulkr_app.exe` or `fluid_simulation.exe`) as a startup item using the x64-Release configuration and run the project. The build files can be found under `build/x64-Release` if desired. Alternatively, running the following CMAKE commands from the root folder level will build the project.
```
cmake -S . -B build
cmake --build build
```


For optimal performance, ensure that the `VULKR_DEBUG` flag is not defined in `vulkan_common.h`. Building in release mode should disable this flag. If you wish to debug the engine using RenderDoc, be sure to enable the `RENDERDOC_DEBUG` flag; also located in `vulkan_common.h` so that unsupported extensions like `VK_KHR_ray_tracing` are disabled to prevent RenderDoc from complaining about them. NSight Systems and NSight Graphics should work without that flag.

## Project Structure
```bash
Vulkr
|-- apps                        # All applcation classes are located here
|   |-- Vulkr                   # Main Vulkr Application
|   |-- Fluid Simulation        # 2D Fluid Simulation Application
|-- assets              
|   |-- glTF models             # All glTF files (.gltf & .glb)
|   |-- OBJ model               # All OBJ related files (.obj and .mtl)
|   |-- textures                # Textures for both gltf and obj models (note that some glb files embed texture information)
|-- src                         # Source files
|   |-- common                
|   |-- core                  
|   |-- platform              
|   |-- rendering             
|   |-- shaders               
|-- third_party                 # All external dependencies
|   |-- VulkanMemoryAllocator               
|   |-- glfw           
|   |-- glm 
|   |-- imgui            
|   |-- spdlog 
|   |-- tinygltf 
|   |-- tinyobjloader 
|   |-- volk
```
## Dependencies
My goal with Vulkr is to create a sandbox engine where I could spend most of my time implement interesting graphics related features while focusing on the rendering specific code to accomplish the task correctly and in a peformant manner. As such, I've relied on a variety to amazing third party libraries to help me do all of the other things that are required for making a rendering engine but are not related to those core goals mentioned before.

As mentioned before, there are two external dependencies that need to be installed separately by the user
- [Vulkan](https://vulkan.lunarg.com/) -> Core graphics and compute API
- [CMake](https://cmake.org/download/) -> C++ Build System

Here are the rest of the external libraries used that are included as submodules on this project
- [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) -> Memory allocation library for Vulkan
- [GLFW](https://github.com/glfw/glfw) -> Multi-platform general windowing system
- [GLM](https://github.com/g-truc/glm) -> Math library based on the GLSL sepcifications
- [Dear ImGui](https://github.com/ocornut/imgui) -> Immediate mode graphics user interface; mainly used for debugging
- [spdlog](https://github.com/gabime/spdlog) -> Fast/low-overhead logging library
- [stb](https://github.com/nothings/stb) -> Single file public domain libraries; mainly used for image loading helpers
- [tiny glTF](https://github.com/syoyo/tinygltf) -> Peformant glTF library
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) -> Performant OBJ loading library
- [volk](https://github.com/zeux/volk) -> Meta-loader for Vulkan
  
## Credits
A special thanks to `Alexander Overvoorde's` [Vulkan Tutorial](https://vulkan-tutorial.com/) for providing a great introduction to Vulkan as a whole and providing the base knowledge required to get started on this project. Additionally, `Sascha Willems'` [Vulkan Demos](https://github.com/SaschaWillems/Vulkan), [Vulkan-glTF-PBR](https://github.com/SaschaWillems/Vulkan-glTF-PBR), [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) provided by the `Khronos Group` and the [Vulkan Ray Tracing Samples](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR) provided by `NVIDIA` have played an instrumental role into providing guidance into best practices, techniques and code samples for implementing various features using Vulkan. The discord and reddit communities on Vulkan and Graphics Programming have been very helpful with answering many of the questions that I've gotten along the way!
