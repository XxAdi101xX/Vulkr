%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/main.vert -o spv/rasterization/main.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/main.frag -o spv/rasterization/main.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/pbr.vert -o spv/rasterization/pbr.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/pbr.frag -o spv/rasterization/pbr.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/mrtGeometryBuffer.vert -o spv/rasterization/mrtGeometryBuffer.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/mrtGeometryBuffer.frag -o spv/rasterization/mrtGeometryBuffer.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/deferredShading.vert -o spv/rasterization/deferredShading.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 rasterization/deferredShading.frag -o spv/rasterization/deferredShading.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 post_processing/postProcess.vert -o spv/post_processing/postProcess.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 post_processing/postProcess.frag -o spv/post_processing/postProcess.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 ray_tracing/raytrace.rgen -o spv/ray_tracing/raytrace.rgen.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 ray_tracing/raytrace.rmiss -o spv/ray_tracing/raytrace.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 ray_tracing/raytraceShadow.rmiss -o spv/ray_tracing/raytraceShadow.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 ray_tracing/raytrace.rchit -o spv/ray_tracing/raytrace.rchit.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 animate.comp -o spv/animate.comp.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 particle_system/particleCalculate.comp -o spv/particle_system/particleCalculate.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 particle_system/particleIntegrate.comp -o spv/particle_system/particleIntegrate.comp.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/velocityAdvection.comp -o spv/fluid_simulation/velocityAdvection.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/densityAdvection.comp -o spv/fluid_simulation/densityAdvection.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/velocityGaussianSplat.comp -o spv/fluid_simulation/velocityGaussianSplat.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/densityGaussianSplat.comp -o spv/fluid_simulation/densityGaussianSplat.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/divergence.comp -o spv/fluid_simulation/divergence.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/jacobi.comp -o spv/fluid_simulation/jacobi.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 fluid_simulation/gradient.comp -o spv/fluid_simulation/gradient.comp.spv -g