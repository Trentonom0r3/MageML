cmake_minimum_required(VERSION 3.23)

# ======================================================
# 1. User-configurable options
# ======================================================
option(MageML_ENABLE_CUDA   "Build with CUDA acceleration" ON)
option(MageML_BUILD_SHARED  "Build MageML as a shared library" OFF)

if (NOT MageML_ENABLE_CUDA)
    set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit ON)
endif()

# ======================================================
# 2. Project declaration
# ======================================================
if (MageML_ENABLE_CUDA)
    project(MageML VERSION 1.0 LANGUAGES CXX CUDA)
else()
    project(MageML VERSION 1.0 LANGUAGES CXX)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ======================================================
# 3. Build acceleration (optional)
# ======================================================
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message(STATUS "Enabling ccache")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    if (MageML_ENABLE_CUDA)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    endif()
endif()

# ======================================================
# 4. Dependencies
# ======================================================
find_package(spdlog CONFIG REQUIRED)
find_package(fmt     CONFIG REQUIRED)   # fmt::fmt

# ---- Torch ----
if (NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR not set. Please define it via environment or CMakePresets.json.")
endif()
find_package(Torch REQUIRED)

# Make sure we have a Torch::Torch target
if (NOT TARGET Torch::Torch)
    if (TARGET torch)
        add_library(Torch::Torch ALIAS torch)
    elseif (TARGET torch_cpu)
        add_library(Torch::Torch ALIAS torch_cpu)
    else()
        message(FATAL_ERROR "LibTorch found but no usable target (torch / torch_cpu) exported.")
    endif()
endif()

if (NOT MageML_ENABLE_CUDA)
    if (TARGET torch_cuda OR TORCH_CUDA_FOUND)
        message(FATAL_ERROR "CPU-only build requested but GPU-enabled libtorch detected. Use a CPU-only LibTorch build.")
    endif()
endif()

# ---- FFmpeg ----
find_package(FFMPEG REQUIRED COMPONENTS avcodec avformat avutil swscale swresample libx264)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad  CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Stb   REQUIRED)

set(VCPKG_INSTALLED "${VCPKG_ROOT}/installed/x64-windows")
include_directories("${VCPKG_INSTALLED}/include")
link_directories("${VCPKG_INSTALLED}/lib")

add_subdirectory(extern/imgui-node-editor/examples)
# ======================================================
# 5. Source Files
# ======================================================
file(GLOB_RECURSE ALL_MageML_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MageML/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MageML/conversion/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MageML/backends/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MageML/filters/*.cpp
)

if (MageML_ENABLE_CUDA)
    file(GLOB_RECURSE MageML_CUDA_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MageML/cuda/*.cu)
    list(APPEND ALL_MageML_SOURCES ${MageML_CUDA_SOURCES})
endif()

# ======================================================
# 6. MageML core library
# ======================================================
set(MageML_LIB_TYPE STATIC)
if (MageML_BUILD_SHARED)
    set(MageML_LIB_TYPE SHARED)
endif()

add_library(MageMLLib ${MageML_LIB_TYPE} ${ALL_MageML_SOURCES})

target_include_directories(MageMLLib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MageML>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/backends>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/filters>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/conversion>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/core>
        $<INSTALL_INTERFACE:include>
        ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(MageMLLib
    PUBLIC
        ${FFMPEG_LIBRARIES}
        Torch::Torch
        spdlog::spdlog_header_only
        fmt::fmt
)

if (WIN32)
    target_link_libraries(MageMLLib PUBLIC ws2_32 bcrypt)
endif()

if (MageML_ENABLE_CUDA)
    target_compile_definitions(MageMLLib PUBLIC MageML_WITH_CUDA)
    set_property(TARGET MageMLLib PROPERTY CUDA_STANDARD 17)
endif()

# ======================================================
# 8. Install rules (core lib)
# ======================================================
include(GNUInstallDirs)

install(TARGETS MageMLLib EXPORT MageMLTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT MageMLTargets NAMESPACE MageML:: FILE MageMLTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MageML)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MageMLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/MageMLConfig.cmake.in")
    configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/MageMLConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/MageMLConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MageML
    )
else()
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/MageMLConfig.cmake
        "# Auto-generated minimal config\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/MageMLTargets.cmake\")\n"
    )
endif()

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MageMLConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MageMLConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MageML)


add_executable(MageMLGUI
    src/MageMLGUI/main.cpp
    src/MageMLGUI/nodes/VideoPipeline.cpp
    src/MageMLGUI/nodes/EncoderPipeline.cpp
    extern/imgui-node-editor/imgui_node_editor.cpp
    extern/imgui-node-editor/imgui_node_editor_api.cpp
    extern/imgui-node-editor/imgui_canvas.cpp
    extern/imgui-node-editor/crude_json.cpp
    extern/imgui-node-editor/external/imgui/imgui_widgets.cpp
    extern/imgui-node-editor/examples/blueprints-example/utilities/drawing.cpp
    extern/imgui-node-editor/examples/blueprints-example/utilities/widgets.cpp
    extern/imgui-node-editor/examples/blueprints-example/utilities/builders.cpp
)

target_include_directories(MageMLGUI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/external/DXSDK
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/external/ScopeGuard
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/external/stb_image
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/examples/application/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/examples/application/source
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui-node-editor/examples/blueprints-example
    ${CMAKE_CURRENT_BINARY_DIR}/include/MageML/error
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MageML
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/filters
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/backends/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/conversion
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MageML/conversion/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MageMLGUI/core
)

target_link_libraries(MageMLGUI PRIVATE
    MageMLLib
    application                 # imgui node editor's helper target
    glfw
    OpenGL::GL
    glad::glad
    spdlog::spdlog_header_only
    fmt::fmt
    onnxruntime                 
)

# ======================================================
# 11. MageMLGUI Install & Bundle (simplified)

# 11.1. MageMLGUI itself
install(TARGETS MageMLGUI
    RUNTIME DESTINATION bin
)

# 11.2. auto-find all the .dlls shipped by LibTorch
# Torch_DIR = …/libtorch/share/cmake/Torch
get_filename_component(_torch_cmake_dir "${Torch_DIR}" DIRECTORY)  # …/libtorch/share
get_filename_component(LIBTORCH_ROOT "${_torch_cmake_dir}" DIRECTORY)  # …/libtorch
file(GLOB TORCH_DLLS      "${LIBTORCH_ROOT}/lib/*.dll")

# 11.3. auto-find all the .dlls shipped by FFmpeg vcpkg
# FFMPEG_LIBRARY_DIRS points at vcpkg’s x64-windows/lib
file(GLOB FFMPEG_DLLS     "${FFMPEG_LIBRARY_DIRS}/*.dll")

# install them all side-by-side with the exe
install(FILES
    ${TORCH_DLLS}
    ${FFMPEG_DLLS}
  DESTINATION bin
)

include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME        "MageMLGUI")
set(CPACK_PACKAGE_VENDOR      "Trentonom0r3")
set(CPACK_PACKAGE_DESCRIPTION "MageML standalone GUI bundle")
set(CPACK_PACKAGE_VERSION     "${PROJECT_VERSION}")
set(CPACK_GENERATOR           "ZIP")
set(CPACK_PACKAGE_FILE_NAME   "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
include(CPack)
