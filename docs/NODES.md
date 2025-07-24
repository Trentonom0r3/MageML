# MageML Nodes – User Guide (GUI Focus)

> **TL;DR:** This page is for people *using* MageML, not hacking the engine. Each node card tells you what it does, the pins you get, the knobs you can turn in the right panel, and common ways to wire it.

---

## How to Read These Cards

* **Pins** → What shows up on the left (inputs) and right (outputs) of the node.
* **Panel Controls** → Sliders, combos, text boxes in the node’s Inspector/Panel.
* **Typical Uses** → Why you’d drop this in your graph.
* **Notes/Gotchas** → Things that bite people (dtype/layout/etc.).

---

## Categories

* **Basic** – constants and manual tensors
* **Arithmetic** – add/mul/sub/div on tensors & scalars
* **Logic** – comparisons, masking, conditional selects
* **TensorOps** – reshape/permute/concat/reductions/image prep
* **MachineLearning** – ONNX model runner
* **Utility** – IO, preview, debugging, device moves, timing

Use the search/filter bar in the add-node menu or browse by category.

---

## BASIC

### **Tensor**

**Pins:** → Tensor
**Panel Controls:** Rank, shape dims, dtype (float32/int32/uint8/float16), Fill (Zeros/Ones/Random), **Apply** button.
**Typical Uses:** Generate a blank tensor to test a pipeline or feed constants into models.
**Notes:** Nothing happens until you click **Apply**.

### **Float / Int / Bool / String**

**Pins:** → matching scalar/string value
**Panel Controls:** DragFloat / DragInt / Checkbox / Text field.
**Typical Uses:** Quick parameters (scales, toggles, filenames, debug text).

---

## ARITHMETIC

> All four accept **Variant** pins (Tensor/float/int) and spit out the same type.

* **Add** – A + B
* **Subtract** – A − B
* **Multiply** – A × B (shows avg ms/FPS)
* **Divide** – A / B (handles divide-by-zero for scalars)

**Pins:** `A`, `B` → `Result`
**Panel Controls:** None (except perf readout on Multiply).
**Typical Uses:** Scale tensors, combine masks, blend values.

---

## LOGIC

### **Compare**

**Pins:** `A (Variant)`, `B (Variant)` → `Result (Tensor<bool>)`
**Panel Controls:** Operator dropdown: `==`, `!=`, `>`, `<`, `>=`, `<=`
**Typical Uses:** Build boolean masks, branch with `Where`.

### **Where**

**Pins:** `Condition (Tensor<bool>)`, `A (Variant)`, `B (Variant)` → `Result`
**Panel Controls:** None
**Typical Uses:** “If cond then A else B” on tensors.
**Gotcha:** Currently expects tensors for A/B (scalars will no-op).

### **Mask**

**Pins:** `In (Variant)`, `Mask (Tensor<bool>)` → `Out`
**Panel Controls:** None
**Typical Uses:** Pull only the pixels/elements where mask is true.

---

## TENSOROPS

### Layout / Type / Shape

#### **Normalize**

**Pins:** `In (Tensor)` → `Out (Tensor)`
**Panel Controls:** Mode (`Normalize` \[0–255 → 0–1] or `Denormalize` \[0–1 → 0–255]), Precision (float16/float32).
**Typical Uses:** Prep images for ML models, convert back for preview/export.

#### **Clamp**

**Pins:** `In (Tensor)`, `Min (Float)`, `Max (Float)` → `Out (Tensor)`
**Typical Uses:** Prevent overflow, clip HDR, etc.

#### **Permute**

**Pins:** `In` → `Out`
**Panel Controls:** CSV order string (e.g. `2,0,1`).
**Typical Uses:** Switch HWC↔CHW, rearrange dims to match a model.

#### **Type**

**Pins:** `In` → `Out`
**Panel Controls:** Drop-down: `uint8`, `int32`, `float32`, `float16`.
**Typical Uses:** Cast to the dtype a model expects.

#### **Squeeze / Unsqueeze**

**Pins:** `In` → `Out`
**Panel Controls:** Dim index (`-1` = all for Squeeze).
**Typical Uses:** Add/remove batch dims or singleton channels.

#### **Reshape**

**Pins:** `In` → `Out`
**Panel Controls:** CSV shape text (`1,3,224,224`).
**Typical Uses:** Force a tensor to a specific shape (be sure sizes match numel!).

#### **Transpose**

**Pins:** `In` → `Out`
**Panel Controls:** Two dim sliders to swap.
**Typical Uses:** Swap H/W, C/H, etc. when Permute overkill.

#### **Concat**

**Pins:** `In1`, `In2` → `Out`
**Panel Controls:** `dim`.
**Typical Uses:** Stitch tensors together along an axis.

#### **Contiguous**

**Pins:** `In` → `Out`
**Typical Uses:** Fix "non-contiguous" errors before passing to other ops.

#### **Flatten**

**Pins:** `In` → `Out`
**Typical Uses:** Collapse to 1D for reductions or logging.

#### **Split**

**Pins:** `In` → `Out1`, `Out2`
**Panel Controls:** `chunks`, `dim` (currently only two outputs).
**Typical Uses:** Separate channels, heads, etc.

#### **Stack**

**Pins:** `In1`, `In2` → `Out`
**Panel Controls:** `dim`.
**Typical Uses:** Combine tensors into a new axis (e.g., batch them).

#### **Slice**

**Pins:** `In` → `Out`
**Panel Controls:** `X, Y, W, H`.
**Typical Uses:** Crop a region (supports 2D or HWC 3D).

#### **Repeat**

**Pins:** `In` → `Out`
**Panel Controls:** `repeats`, `dim`.
**Typical Uses:** Tile a tensor along one axis.

### Reductions

#### **Sum / Mean / Max / Min**

**Pins:** `In (Variant)` → `Out (Variant)`
**Panel Controls:** `dim` (−1 = all), `keepdim`.
**Typical Uses:** Aggregate across channels/height/width.

#### **Argmax**

**Pins:** `In (Tensor)` → `Out (Tensor<int64>)`
**Panel Controls:** `dim`, `keepdim`.
**Typical Uses:** Get index of maximum values (classification, etc.).

### Image Ops

#### **Pad**

**Pins:** `In` → `Out`
**Panel Controls:** Left/Right/Top/Bottom ints, pad value.
**Typical Uses:** Add borders before a model that needs divisible sizes.

#### **Crop**

**Pins:** `In` → `Out`
**Panel Controls:** X, Y, W, H.
**Typical Uses:** Spatial crop without changing channel layout.

#### **Resize**

**Pins:** `In` → `Out`
**Panel Controls:** Target W/H, Mode (Bilinear/Nearest/Bicubic).
**Typical Uses:** Down/upscale to feed a model or prep for export/preview.
**Gotcha:** Converts layout as needed (HWC↔CHW) under the hood.

---

## MACHINELEARNING

### **Onnx**

**Pins:** Dynamic. 1×1 inputs become Floats. Everything else is Tensor. Outputs are Tensors.
**Panel Controls:**

* Model dropdown (auto-scans `./models/*.onnx`)
* Manual path box (Enter to load)
* Device: CPU / CUDA
* Lists IO shapes/types

**Typical Uses:** Run any ONNX model for inference inside your graph.
**Gotchas:** Make sure your input tensors are the shape/dtype the model expects. Use Normalize/Permute/Resize/etc. around it.

---

## UTILITY

### **Read Video**

**Pins:** → `Frame (Tensor)`
**Panel Controls:** Path, Open/Re-open, timeline slider (seek), FPS stats, EOF indicator.
**Typical Uses:** Bring frames in as tensors for processing.

### **Export Video**

**Pins:** `Frame (Tensor)` → (writes file)
**Panel Controls:** Output path, codec, W/H, bitrate, FPS; Initialize/Re-init/Finalize.
**Typical Uses:** Dump your processed frames to a playable video.

### **Load Image**

**Pins:** → `Image (Tensor)`
**Panel Controls:** Path + Load button.
**Typical Uses:** Bring in still images (PNG/JPG/etc.).

### **Preview**

**Pins:** `In (Tensor)` → (UI only)
**Panel Controls:** Lock Aspect, Node Size slider.
**Typical Uses:** See what your tensor looks like (debug/visualize).

### **Delay**

**Pins:** `In (Tensor)`, `Offset (Int)` → `Out (Tensor)`
**Panel Controls:** Cache Size (GB), live stats & offset slider.
**Typical Uses:** Frame differencing, temporal effects, motion trails.

### **Device**

**Pins:** `In (Tensor)` → `Out (Tensor)`
**Panel Controls:** CPU/CUDA combo.
**Typical Uses:** Force a tensor onto GPU before an ONNX op, or back to CPU for encode.

### **Debug**

**Pins:** `Message (String)`, `Value (Variant)` → `Out (Variant)`
**Panel Controls:** Message text, live formatted value display.
**Typical Uses:** Peek at data without breaking the graph. Passes value through.

---

---

### Need More?

* Want node recipes (e.g., “prepare image for upscale model”)? Ping us or open an issue.
* Added a custom node? Mirror this format and send a PR so everyone benefits.

[![Discord](https://img.shields.io/discord/1041502781808328704.svg?label=Join%20Us%20on%20Discord&logo=discord&colorB=7289da)](https://discord.gg/hFSHjGyp4p)
