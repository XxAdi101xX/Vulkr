# Vulkr

Vulkr is a personal Vulkan renderer project that I've taken up in order to solidify my knowledge of Vulkan as I learn more about graphics programming in general! The project is currently a work in progress but I aim to provide more documentation once the foundation has been created.

## Build
Clone the project with the `--recurse-submodules` option to grab all the necessary third party dependencies.

CMake is the build tool of choice for this project running the following commands from the root folder level will build the project.
```
cmake -S . -B build
cmake --build build
```

## Credits
A special thanks to Alexander Overvoorde's [Vulkan Tutorial](https://vulkan-tutorial.com/) for providing a great introduction to Vulkan as a whole and provding the base knowledge required to get started on this project. Additionally, Sascha Willems' [Vulkan Demos](https://github.com/SaschaWillems/Vulkan) and the [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) provided by the KhronosGroup have played an instrumental role into providing guidance into best practices and technique for implementing various features using Vulkan.
