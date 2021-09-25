%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 main.vert -o spv/main.vert.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 default.frag -o spv/default.frag.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 textured.frag -o spv/textured.frag.spv -g

%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytrace.rgen -o spv/raytrace.rgen.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytrace.rmiss -o spv/raytrace.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytraceShadow.rmiss -o spv/raytraceShadow.rmiss.spv -g
%VULKAN_SDK%/Bin/glslc.exe --target-env=vulkan1.2 raytrace.rchit -o spv/raytrace.rchit.spv -g