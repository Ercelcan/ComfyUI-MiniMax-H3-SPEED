# ⚡ ComfyUI MiniMax-H3 SPEED Sampler

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/Model-MiniMax--H3-orange.svg)](https://huggingface.co/Comfy-Org/MiniMax-H3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An adaptive, progressive-resolution diffusion sampler for **MiniMax-H3** (joint Video + Audio) in ComfyUI.

By generating the early layout and motion at lower spatial resolution (e.g., 50%) and dynamically expanding to full resolution with Discrete Cosine Transform (DCT) alignment, this node achieves **up to a ~40% render time reduction** while keeping full-resolution details and native stereo audio synchronization.

Optimized for **Turbo LoRAs (4, 6, 8, 10, 12 steps)** as well as **Standard Schedules (20–25 steps)**.

## 🎯 Mode Compatibility

| Mode | Compatibility | Progressive SPEED Support |
| :--- | :---: | :---: |
| **Text-to-Video (T2V)** | 🟢 Full | **✅ Yes (~40% Faster)** |
| **Reference-to-Video (img,Audio,) (R2V)** | 🟢 Full | **✅ Yes (~40% Faster)** |
| **Image-to-Video (I2V First/Last Frame Keyframing)** | 🟡 Safe Full-Res | ⚠️ *Runs at 1.0x Full Resolution* |

> 📌 **Note on I2V (First/Last Frame):** MiniMax-H3's FL2V model hardcodes reference frame pixels directly into the DiT spatial token rows (`all_video_rows[~img_update] = cond_video_rows`). Because the token count must strictly match the full canvas grid, keyframe I2V runs safely at 1.0x resolution. For **image-guided generation with ~40% SPEED acceleration**, use **Reference-to-Video (`MiniMaxH3ReferenceToVideo`)** or feed images through the Qwen3-VL multimodal prompt!

---

## 📊 Real-World Benchmark (MiniMax-H3 Turbo)

Tested on a 10-step schedule:

| Mode | Generation Time | Speedup Factor | Time Saved |
| :--- | :--- | :--- | :--- |
| **Standard Baseline** | `286s` | 1.00× | 0s |
| **SPEED Sampler** | **`175s`** | **1.63× (+63% throughput)** | **-111s (~39% faster)** |

---

---

## 🎥 Demos & Comparisons

### ⚡ SPEED Sampler (10 Steps — 175 seconds)

https://github.com/user-attachments/assets/f821907e-b659-4706-a426-993c67f90b42

---

### ⏱️ Standard Baseline (10 Steps — 286 seconds)

https://github.com/user-attachments/assets/13b990f4-2195-40a2-9d00-2ccd0bc0ae0b

---


## 🔌 Workflow Connection Guide

Place the **`MiniMax H3 SPEED — Sampler`** in your sampling block. It replaces the default `KSampler` / `SamplerCustomAdvanced` combo.

### 🖼️ Workflow Screenshot
<!-- Place your workflow screenshot here -->
<img width="1301" height="633" alt="pythonw_nvlxCCxcJV" src="https://github.com/user-attachments/assets/e8252b08-e347-42d8-b68f-8f359882823c" />


---

## 🚀 Preset & VRAM Guidance

| Schedule Type | Steps | Recommended Preset | Auto Step Breakdown | `coarse_steps_override` |
| :--- | :--- | :--- | :--- | :--- |
| **Ultra-Fast Turbo** | **4** | `Half -> Full` | **1 step** @ 50% $\rightarrow$ **3 steps** @ 100% | `1` (or `0`) |
| **Fast Turbo** | **6** | `Half -> Full` | **2 steps** @ 50% $\rightarrow$ **4 steps** @ 100% | `1` or `2` |
| **Balanced Turbo** | **8** | `Half -> Full` | **2 steps** @ 50% $\rightarrow$ **6 steps** @ 100% | `2` |
| **High Quality Turbo** | **10** | `Half -> Full` or `Three-Quarter` | **3 steps** @ 50% $\rightarrow$ **7 steps** @ 100% | `0` (Auto) |
| **Base Model** | **20** | `Half -> Full` | **5 steps** @ 50% $\rightarrow$ **15 steps** @ 100% | `0` (Auto) |
| **Base Model** | **25** | `Half -> Full` | **6 steps** @ 50% $\rightarrow$ **19 steps** @ 100% | `0` (Auto) |

### 💡 Tips for 12GB GPUs (e.g., RTX 4070):
* Use **2-Stage Presets** (`Half -> Full` or `Three-Quarter -> Full`). 
* Keep `noise_policy` set to **`direct_coarse`** to minimize VRAM usage during long video generation (73+ frames).
* Multi-stage presets (`Aggressive` / 3-Stage / 4-Stage) perform multiple resolution transitions that can cause VRAM fragmentation on 12GB cards; they are best suited for 16GB–24GB+ GPUs.