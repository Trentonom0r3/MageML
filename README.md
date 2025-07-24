[![License](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1041502781808328704.svg?label=Join%20Us%20on%20Discord&logo=discord&colorB=7289da)](https://discord.gg/hFSHjGyp4p)

# MageML

### 🛠️ Version 1.0.0 BETA — Initial Public Release

[Check out the latest changes](docs/CHANGELOG.md#version-100-beta--initial-public-release)


---

# 📚 Documentation

- [📝 Changelog](docs/CHANGELOG.md)
- [📦 Installation Instructions](docs/INSTALLATION.md)
  - [🛠️ Building from Source](docs/INSTALLATION.md#building-from-source)
- [🤝 Contributing Guide](docs/CONTRIBUTING.md)
- [❓ FAQ](docs/FAQ.md)

---

# 🚀 Features

**MageML** is a blazing-fast, node-based graph editor for video and ML pipelines.

- **Visual, drag-and-drop node editor** for building AI/video flows in seconds
- **ONNX model support** – run any compatible ML model, no code required
- **Chain arbitrary nodes** (decode, filter, ML, encode, and more)
- **High-performance, pure C++/CUDA** backend for real-time throughput
- **Export** for full automation workflows

## For More Information on available nodes, please see [Nodes](docs/NODES.md)

Powered by CeLux (fast video/tensor core), ONNX Runtime, PyTorch, and imgui-node-editor.

---
🎥 [Watch Demo Video](assets/graphvideo.mp4)

---

---

# ⚡ Quick Start Guide

MageML is a real-time, zero-copy machine learning graph editor for video and tensor pipelines. Here’s how to get up and running fast:

---

### 🧩 1. Download the Latest Release  
Grab the latest release from the [Releases page](https://github.com/MageML/releases).  

---

### 🛠️ 2. Extract Files
Extract the Zip File to your preferred location.

---

### 🚀 3. Launch MageML  
Start the app from a shortcut or directly via `MageMLGUI.exe` within the root folder you extracted into.

---

### 🧠 4. (Optional) Add Models  
Place any `.onnx` models into the `models/` folder. Some starter models may already be included.

---

### 🖱️ 5. Start a Project  
Use the UI to load a template graph or create a new one. Drag and connect nodes to begin building your pipeline.

---

# 🧭 UI Overview & Tips

- **Top Left Panel:**  
  Controls for `Live` and `Basic` modes:
  
  - **Live Mode:** All nodes recalculate automatically on changes. *Video pipeline will not push unless running.*
  - **Basic Mode:** Only Video Reader and Preview nodes compute. Ideal for layout/design without processing overhead.

  You can toggle modes **without running the graph** to safely test behaviors.

- **Run / Stop:**  
  Starts/stops the video processing pipeline. You can also modify node parameters during runtime for real-time feedback.

- ⚠️ **Encoding Tip:**  
  Avoid scrubbing during encoding. Let the graph play to completion.  
  Be sure to click `Finalize` in the encoder node to ensure the file is written!

---

## 🧩 Node System

- **Video Reader Node:**  
  Enables timeline control, `Live`/`Basic` toggle, and playback.

- **Active Node List:**  
  Located below the main toolbar. Double-click to rename nodes.

- **Node Inspector Panel:**  
  Appears below the node list. Displays the selected node’s parameters and controls.

- **Right Click Actions:**
  - On **nodes**, see input/output tensor values (great for debugging)
  - On **graph background**, spawn new nodes by category
  - On **pins/links**, access connection options

---

## 💡 Performance Tips

- For max speed: **Remove Preview Nodes** when not needed.
- MageML is **zero-copy**: Tensors only clone during **Preview** or **Encoding**. All other ops run in-place.

---

## ⌨️ Hotkeys

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open graph |
| `Ctrl+S` | Save graph |
| `Ctrl+Shift+S` | Save as |
| `Ctrl+C/V/X` | Copy / Paste / Cut node |
| `Ctrl+A` | Select all |
| `Ctrl+Shift+A` | Deselect all |

Graphs are saved as `.json`, and reusable templates can be added to the `templates/` folder for quick access via the File menu.

---

## 🎨 Style Menu

Use the Style menu to customize the look of:

- Node shapes, colors, and font size
- Link styles (flow map, wire thickness)
- Grid layout, zoom, snapping

---

# 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
See the [LICENSE](LICENSE) file for details.

---

# 🙏 Acknowledgments

- **[CeLux]((https://github.com/trentonom0r3/celux))** – Fast video/tensor core
- **[PyTorch](https://pytorch.org/)** – For tensor ops and CUDA support
- **[ONNX Runtime](https://onnxruntime.ai/)** – Universal ML inference
- **[imgui-node-editor](https://github.com/thedmd/imgui-node-editor)** – Graph UI
- **[vcpkg](https://github.com/microsoft/vcpkg)** – C++ dependency mgmt

---

*For questions, ideas, or just to show off your graph, join our [Discord](https://discord.gg/hFSHjGyp4p)!*
