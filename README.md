# ⚡ ComfyUI MiniMax-H3 SPEED Sampler

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/Model-MiniMax--H3-orange.svg)](https://huggingface.co/Comfy-Org/MiniMax-H3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An adaptive, progressive-resolution diffusion sampler for **MiniMax-H3** (joint Video + Audio) in ComfyUI.

By generating the early layout and motion at lower spatial resolution (e.g., 50%) and dynamically expanding to full resolution with Discrete Cosine Transform (DCT) alignment, this node achieves **up to a ~40% render time reduction** while keeping full-resolution details and native stereo audio synchronization.

Optimized for **Turbo LoRAs (4, 6, 8, 10, 12 steps)** as well as **Standard Schedules (20–25 steps)**.

---

## 🎥 Demos & Comparisons

<!-- Replace these video paths with your uploaded video links or repository assets -->
| Standard Sampler (10 Steps) | SPEED Sampler (10 Steps — ~39% Faster) |
| :---: | :---: |
| <video src="Assets/video_without_speed.mp4" controls width="100%"></video> | <video src="Assets/video_with_speed.mp4" controls width="100%"></video> |
| *286 seconds* | *175 seconds (1.63× speedup)* |

---

## 📊 Real-World Benchmark (MiniMax-H3 Turbo)

Tested on a 10-step schedule:

| Mode | Generation Time | Speedup Factor | Time Saved |
| :--- | :--- | :--- | :--- |
| **Standard Baseline** | `286s` | 1.00× | 0s |
| **SPEED Sampler** | **`175s`** | **1.63× (+63% throughput)** | **-111s (~39% faster)** |

---

## 🔌 Workflow Connection Guide

Place the **`MiniMax H3 SPEED — Sampler`** in your sampling block. It replaces the default `KSampler` / `SamplerCustomAdvanced` combo.

### 🖼️ Workflow Screenshot
<!-- Place your workflow screenshot here -->
![Workflow Diagram](assets/workflow_screenshot.png)

---

### 🧩 Node Connections

```text
┌─────────────────┐
│   RandomNoise   │──(NOISE)──────────────┐
└─────────────────┘                       │
┌─────────────────┐                       ▼
│   BasicGuider   │──(GUIDER)──►┌───────────────────────────┐
└─────────────────┘             │                           │──(output)──►┌──────────────┐──► Images ──┐
┌─────────────────┐             │  MiniMax H3 SPEED Sampler │             │  VAE Decode  │             │
│  BasicScheduler │──(SIGMAS)──►│                           │             └──────────────┘             ▼
└─────────────────┘             │                           │             ┌──────────────┐      ┌─────────────┐
┌─────────────────┐             │                           │──(output)──►│  VAE Decode  │──► Audio ──►│Create Video │
│ ImageToVideo /  │──(LATENT)──►└───────────────────────────┘             │    Audio     │      └─────────────┘
│  Empty Latent   │                                                       └──────────────┘             ▲
└─────────────────┘                                                                                    │
                                                                               (fps / bit_depth) ──────┘

                                                                               
