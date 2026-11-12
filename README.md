# Vulkr

A Vulkan rendering engine that I've taken upon building in order to solidify my knowledge of Vulkan and graphics programming as a whole by implementing interesting graphics features and applications.

The engine runs on `Vulkan 1.4` and the code is written for the `C++20` standard.

![vulkr_helmet_models](https://github.com/user-attachments/assets/0e05f436-ae09-4942-ad00-b4f700c09c97)

https://github.com/user-attachments/assets/9b803162-2994-4b58-b20e-57a44af1821a

## Features

Vulkr is three sample applications on one engine, and each one targets a different slice of the API, so the features are grouped by the executable that implements them.

### `vulkr_app` — forward, deferred and ray traced renderer
- Runtime switchable forward rendering, deferred rendering and ray tracing (`VK_KHR_ray_tracing_pipeline`) over glTF (`.gltf` and `.glb`) and OBJ assets
- Instanced object rendering of several loaded models in the same scene
- Multiple global light sources, each either directional (infinite) or point with its own intensity and position
- Temporal Anti-Aliasing for static and dynamic scenes (rasterization only so far); implemented using techniques mentioned in the "[A Survey of Temporal Antialising Techniques](http://behindthepixels.io/assets/files/TemporalAA.pdf)" white paper and [Ziyad Barakat's TAA Blog Post](https://ziyadbarakat.wordpress.com/2020/07/28/temporal-anti-aliasing-step-by-step/)
- Shadows (ray tracing only so far)
- A particle system implemented using compute shaders for the position and velocity calculations

https://user-images.githubusercontent.com/18451835/173991616-5825f922-1a45-4556-9cf3-51c6615e918b.mp4

- A free flight camera with similar controls to blender, and an imgui debug window that provides access to altering the camera and light settings

### `vulkr_mesh` — mesh shading sandbox
- A scene of glTF models (`.gltf` and `.glb`) rendered as meshes built with [meshoptimizer](https://github.com/zeux/meshoptimizer)
- Duplicate vertices are collapsed and each cluster is compacted to at most 64 vertices, so a workgroup streams a small local subset instead of the whole mesh. Positions are then quantized to two 16-bit signed normalized components per word and the per-cluster vertex order is tuned for locality
- Renders using a task and mesh shader pipeline that rejects meshlets through frustum and normal cone culling before any geometry work happens
- Indirect mesh task draws: the task workgroup count lives in a buffer instead of the draw call, so a compute pass can rewrite it
- Buffer device address: every meshlet buffer is reached through a 64 bit address in the push constants, so there are no buffer descriptors at all
- Bindless textures: one descriptor set holds a variable count array of sampled images that the fragment shader indexes non-uniformly
- Dynamic rendering, synchronization2 and Slang shaders compiled at runtime

### `fluid_simulation` — 2D GPU fluid
- Velocity and density advection, a gaussian splat on mouse click + drag, and a projection step that solves the poisson pressure equation and subtracts the gradient of the pressure from the velocity field; implemented using the techniques mentioned in [Mark Harris' "Fluid Dynamics Simulation on the GPU" GPU Gems article](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- Field visualization modes with an imgui overlay to display velocity direction/magnitude, pressure field, diveregence and curl

### Shared engine (`src/`)
- Vulkan 1.4 through volk, with core synchronization2 and VMA for memory allocation
- Two frames in flight with per-frame fences, semaphores and command pools, over a triple buffered swapchain
- Camera and camera controller, logging, and glTF/OBJ loading (the mesh application has its own single-parse glTF loader that also decodes the textures)
- GLSL compiled to committed SPIR-V for `vulkr_app` and `fluid_simulation`; `vulkr_mesh` compiles Slang at startup instead

## Build
This project requires the `Vulkan 1.4` SDK (or newer; the SDK must include Slang, which `vulkr_mesh` compiles shaders with at runtime) and `CMake` v3.20 or above installed on the host's computer. All other third-party dependencies should be imported when cloning the project with the `--recurse-submodules` option. I developed Vulkr running `Windows 11, x64` using `Visual Studio 2026` and as such, it cannot be expected to work on Linux or MacOS. However, all third-party libraries used are cross platform and OS platform specific components like the windowing system could be supported in the future. 

The easiest way to run the project is to load it using `Visual Studio 2026` and building the project. You could then select any application executable file (`vulkr_app.exe`, `fluid_simulation.exe`, or `vulkr_mesh.exe`) as a startup item using the x64-Release configuration and run the project. The build files can be found under `build/x64-Release` if desired. Alternatively, running the following CMAKE commands from the root folder level will build the project.
```
cmake -S . -B build/x64-Release
cmake --build build/x64-Release
```
The executables resolve assets and shaders relative to the working directory (`../../assets` and `../../shaders`), so they have to be launched from a two level deep build directory such as `build/x64-Release`.

`vulkr_mesh` differs from the other two applications in one important way: its shaders are Slang sources under `shaders/mesh/` that the app compiles at startup, so there is nothing to rebuild and no committed SPIR-V to fall out of date. If a shader fails to compile, the error comes from Slang in the app's log rather than from the build.

For optimal performance, ensure that the `VULKR_DEBUG` flag is not defined in `vulkan_common.h`. Building in release mode should disable this flag.

## Project Structure
```bash
Vulkr
|-- apps                        # All application classes are located here
|   |-- vulkr                   # Main Vulkr application that implements the forward, deferred and ray tracing rendering techniques
|   |-- fluid_simulation        # 2D fluid simulation application
|   |-- mesh                    # Mesh shader application: glTF loading, meshlet building, task culling and bindless textures
|-- assets
|   |-- gltf_models             # All glTF files (.gltf & .glb)
|   |-- obj_models              # All OBJ related files (.obj and .mtl)
|   |-- textures                # Textures for both gltf and obj models (note that some glb files embed texture information)
|-- shaders
|   |-- fluid_simulation        # All shaders related to the 2D fluid simulation app
|   |-- particle_system         # Shaders required to calculate and render particle positions
|   |-- forward_rendering       # Shaders for the forward rasterization technique
|   |-- deferred_rendering      # Shaders for the deferred rasterization technique
|   |-- ray_tracing             # Ray tracing related shaders
|   |-- post_processing         # Shaders for all post processing techniques eg. TAA
|   |-- mesh                    # Slang shaders for the mesh shader app; compiled at runtime by the app itself
|   |-- spv                     # Committed SPIR-V output for the GLSL shaders above
|   |-- common.glsl             # Contains common structs that need to be synced across CPU/GPU and are used in various shaders
|   |-- random.glsl             # RNG helpers
|   |-- animate.comp            # Model animation compute shader (not currently used)
|   |-- compile_shaders.bat     # The script to compile GLSL shaders into spirv. If you don't execute this after any shader changes, the spirv WILL NOT change
|-- src
|   |-- common                  # Common utils and helpers are located here
|   |-- core                    # Core Vulkan API classes
|   |-- platform                # Everything related to setting up the platform (eg. windowing system, processing key and mouse inputs)
|   |-- rendering               # All other rendering related classes
|-- third_party                 # All external dependencies; details on each library mentioned in the dependencies section below
```
## Dependencies
My goal with Vulkr is to create a sandbox engine where I could spend most of my time implement interesting graphics related features while focusing on the rendering specific code to accomplish the task correctly and in a peformant manner. As such, I've relied on a variety to amazing third party libraries to help me do all of the other things that are required for making a rendering engine but are not related to those core goals mentioned before.

As mentioned before, there are two external dependencies that need to be installed separately by the user
- [Vulkan](https://vulkan.lunarg.com/) -> Core graphics and compute API. The SDK also ships [Slang](https://github.com/shader-slang/slang), which `vulkr_mesh` uses to compile its shaders at runtime
- [CMake](https://cmake.org/download/) -> C++ Build System

Here are the rest of the external libraries used that are included as submodules on this project
- [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) -> Memory allocation library for Vulkan
- [GLFW](https://github.com/glfw/glfw) -> Multi-platform general windowing system
- [GLM](https://github.com/g-truc/glm) -> Math library based on the GLSL specifications
- [Dear ImGui](https://github.com/ocornut/imgui) -> Immediate mode graphics user interface; mainly used for debugging
- [spdlog](https://github.com/gabime/spdlog) -> Fast/low-overhead logging library
- [stb](https://github.com/nothings/stb) -> Single file public domain libraries; mainly used for image loading helpers
- [tiny glTF](https://github.com/syoyo/tinygltf) -> Performant glTF library
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) -> Performant OBJ loading library
- [volk](https://github.com/zeux/volk) -> Meta-loader for Vulkan
- [Basis Universal](https://github.com/BinomialLLC/basis_universal) -> Texture transcoder used for the compressed glTF materials
- [meshoptimizer](https://github.com/zeux/meshoptimizer) -> Mesh and meshlet optimization used by the mesh shader application
  
## Credits
A special thanks to `Alexander Overvoorde's` [Vulkan Tutorial](https://vulkan-tutorial.com/) for providing a great introduction to Vulkan as a whole and providing the base knowledge required to get started on this project. Additionally, `Sascha Willems'` [Vulkan Demos](https://github.com/SaschaWillems/Vulkan), [Vulkan-glTF-PBR](https://github.com/SaschaWillems/Vulkan-glTF-PBR), [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) provided by the `Khronos Group` and the [Vulkan Ray Tracing Samples](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR) provided by `NVIDIA` have played an instrumental role into providing guidance into best practices, techniques and code samples for implementing various features using Vulkan. `Sascha Willems'` [How to Vulkan](https://www.howtovulkan.com) and [meshoptimizer](https://github.com/zeux/meshoptimizer) by `Arseny Kapoulkine` provided the reference for the modern Vulkan patterns and the meshlet tooling used by the mesh shader application. The discord and reddit communities on Vulkan and Graphics Programming have been very helpful with answering many of the questions that I've gotten along the way!

Much of the models and textures that I'm using are from KhronosGroup's [glTF-Sample-Asset repository](https://github.com/KhronosGroup/glTF-Sample-Assets) so please refer to the author credits and license agreements mentioned in that repository.
