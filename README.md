# ComfyUI MiniMax-H3 SPEED Sampler

An adaptive, progressive-resolution diffusion sampler for **MiniMax-H3** (Audio + Video) in ComfyUI. 

By calculating the initial composition and camera motion at coarse resolution (e.g. 50%) and dynamically DCT-expanding to full resolution for the remaining steps, it significantly reduces GPU compute time while preserving sharp details and joint audio-video synchronization.

Optimized for **Turbo LoRAs (4, 6, 8, 10, 12 steps)** and **Standard Schedules (20–25 steps)**.

---

## ⚡ Features

- **Adaptive Step Allocation**: Automatically scales progressive stage boundaries for any schedule length (from 4-step Turbo LoRAs up to 30+ base steps).
- **Manual Coarse Override**: Direct knob (`coarse_steps_override`) to tune exactly how many low-res steps to run.
- **Audio-Video Consistency**: Preserves MiniMax-H3's native stereo audio track across progressive resolution transitions.
- **Lightweight & Self-Contained**: Pure PyTorch implementation with zero extra dependencies.

---

## 📦 Installation

1. Navigate to your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/YOUR_USERNAME/ComfyUI-MiniMax-H3-SPEED.git