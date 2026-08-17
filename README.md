# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md). Free to use for personal/learning projects. Contact for commercial use.

A ComfyUI node that runs [SPEED](https://github.com/howardhx/speed) (Spectral Progressive Diffusion) on MiniMax-H3's packed video+audio latent. Replaces KSAMPLER + SamplerCustomAdvanced.

## Why use this?

Standard diffusion generates at full resolution the whole time. SPEED starts coarse (half or quarter resolution), then progressively refines up to full. You get similar quality with less VRAM and faster generation because most steps run on smaller buffers.

```
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider (UNETLoader + MiniMaxH3ImageToVideo)
  sigmas       ← BasicScheduler (default 20 steps)
  latent_image ← MiniMaxH3ImageToVideo
  preset       ← "half_then_full" (default)
                ↓
  output → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo
```

## Install

```bash
git clone https://github.com/StanLukuvka/H3-SPEED.git ComfyUI/custom_nodes/H3-SPEED
# restart ComfyUI
```

**Required:** MiniMax-H3 plugin ([ComfyUI-MiniMax-H3](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3), requires ComfyUI 0.32.0+).

## Usage

After cloning, the workflow is at `ComfyUI/custom_nodes/H3-SPEED/workflows/video_minimax_h3_t2v_speed.json`. Load it via ComfyUI's workflow browser (Workflow → Open).

The default `half_then_full` preset works out of the box. No changes needed.

Options:
- `preset` — see table below

### Presets (default 20-step schedule)

Each preset splits denoising across resolutions. More stages = more time at low res = faster but potentially softer mid-frequency detail.

| Preset | Steps @ each stage | Outcome |
|--------|-------------------|---------|
| `half_then_full` | 5 @ 50%, 15 @ 100% | Default. Good balance. |
| `three_quarter_then_full` | 10 @ 75%, 10 @ 100% | Fastest. Fewer coarse steps, but may miss fine detail. |
| `quarter_half_full` | 3 @ 25%, 2 @ 50%, 15 @ 100% | Higher quality. More refinement passes. |
| `aggressive` | 3 @ 25%, 5 @ 75%, 12 @ 100% | Skips 50% stage. Fast but loses mid-frequency detail. |
| `quarter_half_3q_full` | 3 @ 25%, 2 @ 50%, 3 @ 75%, 12 @ 100% | Slowest. Highest quality. All intermediate resolutions. |

**How to choose:**
- **Speed** → `three_quarter_then_full` (fastest, decent quality)
- **Quality** → `quarter_half_3q_full` (most stages, slowest)
- **Default** → `half_then_full` (proven sweet spot)

## Repository structure

```
H3-SPEED/
├── minimax_h3_speed/
│   ├── config.py              — presets, transition steps, SpeedConfig
│   ├── h3_runtime.py          — multi-stage diffusion loop
│   ├── spectral.py            — resolution expansion math
│   ├── flow.py                — sigma alignment, audio handling
│   └── tests/                 — 22 passing tests
├── sampler_node.py            — ComfyUI node definition
└── workflows/
    └── video_minimax_h3_t2v_speed.json
```

## Test suite

```bash
PYTHONPATH=minimax_h3_speed python -m pytest minimax_h3_speed/tests/ -q
```

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md).

Canonical SPEED: [howardhx/speed](https://github.com/howardhx/speed).
