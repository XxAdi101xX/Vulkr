%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 main.vert -o spv/main.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 main.frag -o spv/main.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 postProcess.vert -o spv/postProcess.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 postProcess.frag -o spv/postProcess.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 raytrace.rgen -o spv/raytrace.rgen.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 raytrace.rmiss -o spv/raytrace.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 raytraceShadow.rmiss -o spv/raytraceShadow.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 raytrace.rchit -o spv/raytrace.rchit.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 animate.comp -o spv/animate.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 particleCalculate.comp -o spv/particleCalculate.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 particleIntegrate.comp -o spv/particleIntegrate.comp.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 fluidAdvection.comp -o spv/fluidAdvection.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.3 jacobi.comp -o spv/jacobi.comp.spv -g