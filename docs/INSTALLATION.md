
## 📦 Installation

# 🛠️ MageML Build & Install Guide (Windows)

**This document will walk you step-by-step through building MageML on Windows using Visual Studio, vcpkg, CMake, and libtorch.**

---

## 1. Prerequisites & Downloads

**You will need:**

- [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) (Community Edition is fine).  
  _Install the **Desktop Development with C++** workload during setup._
- [CMake 3.23 or newer](https://cmake.org/download/)
- [Git](https://git-scm.com/download/win) (for cloning repositories)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 
- About 10GB free disk space

---

### 1.1. Install vcpkg

**vcpkg** is Microsoft's dependency/package manager for C++ libraries.

```sh
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.ootstrap-vcpkg.bat
```
_Note:_ If you prefer, you can use a graphical Git client to clone [https://github.com/microsoft/vcpkg.git](https://github.com/microsoft/vcpkg.git)

---

### 1.2. Install MageML’s dependencies with vcpkg

**From your `vcpkg` directory, run:**

```sh
.cpkg install ffmpeg[avcodec,avformat,avutil,swscale,swresample,libx264] glfw3 glad spdlog fmt stb onnxruntime-gpu
```

_Note: This may take a while on first install!_

---

### 1.3. Download libtorch

- Go to the [official PyTorch website](https://pytorch.org/get-started/locally/)
- Under "Get LibTorch", download the ZIP for your setup:
  - **For CUDA (GPU):** `libtorch-cuda-*.zip` (matching your CUDA version)
- **Extract** the ZIP somewhere permanent (e.g., `C:\libs\libtorch` or next to your project directory)

---

### 1.4. Download MageML Source Code

You can either clone your own fork or the main MageML repository:

```bash
git clone https://github.com/trentonom0r3/MageML.git
cd MageML
git submodule update --init --recursive
```

> 🧱 Be sure to run `git submodule update --init --recursive` after cloning — MageML uses submodules like `imgui-node-editor` and others.

## 2. Project Directory Layout

Here’s what your **directory structure** should look like after you’ve finished steps 1.x:

```
C:\dev\
 ├── MageML\
 │    ├── CMakeLists.txt
 │    ├── CMakePresets.json
 │    ├── src\
 │    ├── include\
 │    └── extern\
 ├── vcpkg\
 │    └── (contains vcpkg.exe, installed/...)
 └── libtorch\
      └── (contains share/cmake/Torch, lib/, bin/, etc)
```

- _If you put vcpkg or libtorch somewhere else, just update your paths below._

---

## 3. Configure CMakePresets.json

**Open the file `CMakePresets.json` in a text editor.**  
**Find and update** these lines with your correct absolute paths (don’t use `~` or relative paths):

```jsonc
"CMAKE_TOOLCHAIN_FILE": "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake",
"VCPKG_ROOT": "C:/dev/vcpkg",
"Torch_DIR": "C:/dev/libtorch/share/cmake/Torch",
```
- Use `/` slashes or double `\\` (not single `\`)
- If your triplet or build settings differ, update them accordingly

---

## 4. Building with Visual Studio

1. **Open Visual Studio 2022**  
2. Click **"Open a Local Folder"** and select your `MageML` project directory.
3. Wait for CMake to finish configuring (look for the spinner in the bottom status bar).
4. At the top, click the **CMake Preset** dropdown (may be labeled `No Configurations`).
    - Choose:
      - `x64-debug` — debug build
      - `x64-release` — release build
5. Go to **Build > Build All** or press `Ctrl+Shift+B`.
6. Your build output (`MageMLGUI.exe`) will appear in  
   `out/build/x64-debug/` (or similar, depending on preset)

---

## 5. Running MageML

After building, find and run `MageMLGUI.exe` in  
`out/build/x64-debug/`  
or (if you installed):  
`out/install/x64-debug/bin/`

---

## 6. Troubleshooting

- **Dependency not found:** Double-check all paths in your `CMakePresets.json` are correct and use `/` slashes.
- **Torch errors:** Ensure `Torch_DIR` is set to the directory containing `TorchConfig.cmake`.
- **CUDA errors:** Confirm the CUDA Toolkit is installed, and that you downloaded the correct libtorch package for CUDA.
- **ffmpeg or other DLLs missing at runtime:** Make sure you installed (`cmake --install ...`) so DLLs get copied.
- **Stale builds/errors:** If things get weird, delete the `out/build` and `out/install` folders and rebuild:

```sh
rd /s /q out\build
rd /s /q out\install
```

---

**Need help?**  
Open an [issue on GitHub](https://github.com/MageML/issues) and include your full error message, `CMakePresets.json`, and your directory structure.

---

# 🎉 You’re done!