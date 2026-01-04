%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 forward_rendering/forward_obj.vert -o spv/forward_rendering/forward_obj.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 forward_rendering/forward_obj.frag -o spv/forward_rendering/forward_obj.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 forward_rendering/forward_gltf.vert -o spv/forward_rendering/forward_gltf.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 forward_rendering/forward_gltf.frag -o spv/forward_rendering/forward_gltf.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 deferred_rendering/mrtGeometryBuffer.vert -o spv/deferred_rendering/mrtGeometryBuffer.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 deferred_rendering/mrtGeometryBuffer.frag -o spv/deferred_rendering/mrtGeometryBuffer.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 deferred_rendering/deferredShading.vert -o spv/deferred_rendering/deferredShading.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 deferred_rendering/deferredShading.frag -o spv/deferred_rendering/deferredShading.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 post_processing/postProcess.vert -o spv/post_processing/postProcess.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 post_processing/postProcess.frag -o spv/post_processing/postProcess.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 ray_tracing/raytrace.rgen -o spv/ray_tracing/raytrace.rgen.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 ray_tracing/raytrace.rmiss -o spv/ray_tracing/raytrace.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 ray_tracing/raytraceShadow.rmiss -o spv/ray_tracing/raytraceShadow.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 ray_tracing/raytrace.rchit -o spv/ray_tracing/raytrace.rchit.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 animate.comp -o spv/animate.comp.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 particle_system/particleCalculate.comp -o spv/particle_system/particleCalculate.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 particle_system/particleIntegrate.comp -o spv/particle_system/particleIntegrate.comp.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/velocityAdvection.comp -o spv/fluid_simulation/velocityAdvection.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/densityAdvection.comp -o spv/fluid_simulation/densityAdvection.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/velocityGaussianSplat.comp -o spv/fluid_simulation/velocityGaussianSplat.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/densityGaussianSplat.comp -o spv/fluid_simulation/densityGaussianSplat.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/divergence.comp -o spv/fluid_simulation/divergence.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/jacobi.comp -o spv/fluid_simulation/jacobi.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.4 fluid_simulation/gradient.comp -o spv/fluid_simulation/gradient.comp.spv -g