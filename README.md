# ComfyUI MiniMax-H3 SPEED Sampler

A single ComfyUI node that runs [SPEED](https://github.com/howardhx/speed) (Spectral Progressive Diffusion) on MiniMax-H3's packed video+audio latent. Replaces KSAMPLER + SamplerCustomAdvanced.

## What it does

Runs multi-stage progressive-resolution diffusion: coarse pass first, then DCT-expand to full resolution and continue denoising. 
Each stage issues its own `guider.sample()` so the model always sees the right buffer size.

```
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider
  sigmas       ← BasicScheduler
  latent_image ← MiniMaxH3ImageToVideo
  preset       ← "2_stage_half" (default)
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
- `preset` — scale ladder (see below)
- `transition_mode` — `explicit` (hardcoded for MVP)

### Presets

Each preset defines how many stages the denoising runs through, and at what resolution fractions. More stages = more time at low res = faster but coarser. Fewer stages = less coarse work = slower but higher quality.

| Preset | Resolution path | When to use |
|--------|----------------|-------------|
| `half_then_full` | 50% → 100% | Default. |
| `three_quarter_then_full` | 75% → 100% | Fastest option.  |
| `quarter_half_full` | 25% → 50% → 100% | Slower, higher quality? |
| `aggressive` | 25% → 75% → 100% | Skips the 50% stage. Fast but probably loses mid-freq detail. |
| `quarter_half_3q_full` | 25% → 50% → 75% → 100% | Untested. |

## Structure

```
H3-SPEED/
├── minimax_h3_speed/
│   ├── config.py              — presets, transition steps
│   ├── h3_runtime.py          — multi-stage diffusion loop
│   ├── spectral.py            — DCT expand
│   ├── flow.py                — transition math
│   └── tests/                 — 22 passing tests
├── sampler_node.py            — ComfyUI node
└── workflows/
    └── video_minimax_h3_t2v_speed.json
```

## Test suite

```bash
PYTHONPATH=minimax_h3_speed python -m pytest minimax_h3_speed/tests/ -q
```

## TODO

Currently this is a proof of concept that naive implements SPEED.
Full paper implementation is still a work in progress.

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md).
