# ComfyUI MiniMax-H3 SPEED Sampler

A single ComfyUI node that runs [SPEED](https://github.com/howardhx/speed) (Spectral Progressive Diffusion) on MiniMax-H3's packed video+audio latent. Replaces KSAMPLER + SamplerCustomAdvanced.

## What it does

Runs multi-stage progressive-resolution diffusion: coarse pass first, then DCT-expand to full resolution and continue denoising. Each stage issues its own `guider.sample()` so the model always sees the right buffer size.

```
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider
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

Requires MiniMax-H3 plugin.

## Usage

Load `workflows/video_minimax_h3_t2v_speed.json`. The default `half_then_full` preset works out of the box.

Options:
- `preset` — see table below
- `transition_mode` — `explicit` (hardcoded for MVP)

### Presets (with default 20-step sigma schedule)

Each preset defines how many denoising steps run at each resolution. The sigma schedule controls total steps; the preset controls where the resolution switches happen.

| Preset | Coarse steps | Resolution path | Full-res steps | Total steps |
|--------|-------------|----------------|----------------|-------------|
| `half_then_full` | 5 @ 50% | 50% → 100% | 15 | 20 |
| `three_quarter_then_full` | 10 @ 75% | 75% → 100% | 10 | 20 |
| `quarter_half_full` | 3 @ 25%, 2 @ 50% | 25% → 50% → 100% | 15 | 20 |
| `aggressive` | 3 @ 25%, 5 @ 75% | 25% → 75% → 100% | 12 | 20 |
| `quarter_half_3q_full` | 3 @ 25%, 2 @ 50%, 3 @ 75% | 25% → 50% → 75% → 100% | 12 | 20 |

**How to choose:**
- **Speed** → `three_quarter_then_full` (fewest coarse steps, most full-res work)
- **Quality** → `quarter_half_3q_full` (most resolution transitions, slowest)
- **Default** → `half_then_full` (good balance, proven calibrated)
- **Fast & decent** → `half_then_full` is already the sweet spot for most prompts

The coarse stages are cheap — fewer pixels to denoise. The full-res stage is where the detail lives. More stages = more time spent cheap, but each transition risks losing mid-frequency detail if the DCT expand doesn't seed it well.

## Repository structure

```
H3-SPEED/
├── minimax_h3_speed/
│   ├── config.py              — presets, transition steps, SpeedConfig
│   ├── h3_runtime.py          — multi-stage diffusion loop
│   ├── spectral.py            — DCT expand (2D orthonormal DCT-II)
│   ├── flow.py                — sigma alignment, audio transition math
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
