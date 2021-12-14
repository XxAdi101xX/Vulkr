%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 main.vert -o spv/main.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 main.frag -o spv/main.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 postProcess.vert -o spv/postProcess.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 postProcess.frag -o spv/postProcess.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytrace.rgen -o spv/raytrace.rgen.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytrace.rmiss -o spv/raytrace.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytraceShadow.rmiss -o spv/raytraceShadow.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytrace.rchit -o spv/raytrace.rchit.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 animate.comp -o spv/animate.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 particleCalculate.comp -o spv/particleCalculate.comp.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 particleIntegrate.comp -o spv/particleIntegrate.comp.spv -g