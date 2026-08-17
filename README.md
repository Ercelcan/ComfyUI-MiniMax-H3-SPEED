# MiniMax H3 SPEED Sampler

A ComfyUI custom node running [SPEED / Spectral Progressive Diffusion](https://github.com/howardhx/speed) on MiniMax-H3's packed video+audio latent.

## What it is

One node — **MiniMax H3 SPEED — Sampler** — that replaces KSAMPLER + SamplerCustomAdvanced. Accepts `noise`, `guider`, `sigmas`, `latent_image` and runs multi-stage progressive-resolution diffusion internally.

```text
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider (UNETLoader + MiniMaxH3ImageToVideo)
  sigmas       ← BasicScheduler
  latent_image ← MiniMaxH3ImageToVideo
  preset, transition_mode
                ↓
  output (LATENT) → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo
```

## The algorithm

SPEED replaces one full-res denoise pass with several progressive passes:

1. **Coarse pass** — denoise at reduced resolution (e.g. 50%)
2. **Spectral expansion** — DCT upsample, inject high-frequency noise at boundary sigma
3. **Fine pass** — denoise at full resolution, refining the seeded detail

Each stage issues a separate `guider.sample()` so H3's shape-dependent conditioning rebuilds from current buffer geometry.

```
                     ┌─────────────────────────────────────┐
                     │      latent (full-res)             │
                     │   video [1,24,T,H,W] + audio       │
                     └─────────────────────────────────────┘
                                       ▲
                                       │ stage 2: full-res denoise
                                       │ (kappa-aligned boundary)
                                       │
                     ┌─────────────────────────────────────┐
                     │      spectral expand (DCT upsample) │
                     │   low-band:  from coarse output     │
                     │   high-band: noise × boundary sigma │
                     └─────────────────────────────────────┘
                                       ▲
                                       │ stage 1: half-res denoise
                                       │
                     ┌─────────────────────────────────────┐
                     │      latent (half-res)              │
                     │   video [1,24,T,H/2,W/2] + audio   │
                     └─────────────────────────────────────┘
```

## Configuration

| Parameter | Values | Default |
|-----------|--------|---------|
| `preset` | `2_stage_half`, `3_stage_quarter`, `4_stage_quarter`, `3_stage_aggressive`, `2_stage_3quarter` | `2_stage_half` |
| `transition_mode` | `explicit` | `explicit` |

Presets define the scale ladder and calibrated transition steps:

| Preset | Scales | Transition Steps | Full-res steps |
|--------|--------|-----------------|----------------|
| `2_stage_half` | 0.5 → 1.0 | (5,) | 15 of 20 |
| `3_stage_quarter` | 0.25 → 0.5 → 1.0 | (3, 5) | 15 of 20 |
| `4_stage_quarter` | 0.25 → 0.5 → 0.75 → 1.0 | (3, 5, 8) | 12 of 20 |
| `3_stage_aggressive` | 0.25 → 0.75 → 1.0 | (3, 8) | 12 of 20 |
| `2_stage_3quarter` | 0.75 → 1.0 | (10,) | 10 of 20 |

## Concepts

- **`latent_shapes`** — H3 builds cross-attention from buffer geometry. Each stage re-bakes shapes via fresh `guider.sample()`.
- **`packed latent`** — NestedTensor with video `[B,C,T,H,W]` + audio `[B,C,2,T]` (stereo axis = 2).
- **`kappa alignment`** — Rescales boundary sigma so seeded detail looks natural: `kappa = r / (1 + (r-1)*q)`.
- **`sigma_shift_video/audio`** — H3 model parameters (defaults 12.0, 3.0) controlling denoise regime separation.

## Install

```bash
git clone https://github.com/StanLukuvka/H3-SPEED.git ComfyUI/custom_nodes/H3-SPEED
# restart ComfyUI
```

Requires MiniMax-H3 plugin (or ComfyUI 0.32.0+).

## Usage

Load `workflows/video_minimax_h3_t2v_speed.json` in ComfyUI.

## Repository structure

```
H3-SPEED/
├── minimax_h3_speed/
│   ├── config.py              — SpeedConfig, presets, DEFAULT_TRANSITION_STEPS
│   ├── flow.py                — Sigma alignment, audio transition math
│   ├── h3_runtime.py          — run_progressive_stages (main loop)
│   ├── spectral.py            — DCT primitives
│   └── tests/
│       ├── test_dct.py        — DCT math (8 tests)
│       └── test_sampler.py    — Node contract (14 tests)
├── sampler_node.py            — ComfyUI node definition
├── workflows/
│   └── video_minimax_h3_t2v_speed.json
└── README.md
```

## Modules

### `config.py`
- `SCALE_PRESETS` — name → scale ladder
- `DEFAULT_TRANSITION_STEPS` — name → transition indices
- `SpeedConfig` — frozen dataclass, validates at construction

### `flow.py`
- `aligned_speed_sigma(q, r)` — kappa + aligned boundary sigma
- `time_shift_sigma(sigma, from_shift, to_shift)` — audio sigma shift
- `recover_internal_state(public, sigma, audio_scale)` — denoised → carry
- `carry_preserving_audio_state(...)`, `clock_reindex_audio_state(...)` — audio policies
- `reentry_noise(internal, start_sigma)` — noise seed for next stage

### `h3_runtime.py`
`run_progressive_stages(noise, guider, sigmas, latent, config, ...)`:
1. Validate H3 nested latent (5D video, 4D stereo audio, batch=1)
2. Read live sigma shifts from model
3. Initialize coarse latent + noise
4. Loop stages: denoise → recover → align → DCT-expand → audio transition → build next latent
5. Final full-res stage → return `(output, denoised)`

### `spectral.py`
- `dct2(value)` / `idct2(coeffs)` — 2D orthonormal DCT-II
- `lowpass_dct(value, target_hw)` — DCT-truncate then inverse
- `spectral_expand_dct(value, target_hw, sigma, seed)` — expand with seeded noise
- `spectral_expand_dct_coupled(value, full_noise, sigma)` — coupled expansion

## Test suite

```bash
PYTHONPATH=minimax_h3_speed python -m pytest minimax_h3_speed/tests/ -q
# 22 passed
```

## Not implemented

- `delta_custom` transition mode (deferred to Phase 5 — needs H3 power spectrum calibration)
- Optional latent input for live geometry detection
- `coupled_full_grid` noise policy (architecture supports it; not tested end-to-end)
- SAMPLER-type node variant (would need ComfyUI-level changes)

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md).

Canonical SPEED: [howardhx/speed](https://github.com/howardhx/speed).
