# ComfyUI MiniMax-H3 SPEED Sampler

A minimal, self-contained ComfyUI custom node that runs
[SPEED / Spectral Progressive Diffusion](https://github.com/howardhx/speed) on
MiniMax-H3's packed video+audio latent. Exposed as a `LATENT`-returning node
that drives its own `guider.sample()` per SPEED stage.

## What it is

One ComfyUI node — **`MiniMax H3 SPEED — Sampler`** — that accepts the same
inputs as ComfyUI's stock KSAMPLER step (`noise`, `guider`, `sigmas`,
`latent_image`) and returns a denoised `LATENT` ready for `VAEDecode`. It
occupies the same slot in a workflow as the KSAMPLER + SamplerCustomAdvanced
pair — but does the multi-stage progressive-resolution diffusion internally.

```text
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider (UNETLoader + MiniMaxH3ImageToVideo)
  sigmas       ← BasicScheduler
  latent_image ← MiniMaxH3ImageToVideo
  preset, transition_mode, delta, power_A, power_beta, seed_offset
                ↓
  output (LATENT) → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo
```

The node is named `MiniMaxH3SPEEDSampler` because it occupies the same slot
as KSAMPLER in a workflow — same inputs, same output. The "SAMPLER" name
isn't about ComfyUI's `SAMPLER` data type; it's about what the node *does*
for the user (replaces the KSAMPLER step).

## What SPEED does (the algorithm)

SPEED (Spectral Progressive Diffusion) replaces a single full-resolution
denoising pass with several progressive-resolution passes:

1. **Coarse pass.** Denoise the latent at a reduced spatial resolution (e.g.
   half size). The model does less work per step because the buffer is smaller.
2. **Spectral expansion.** At the boundary between passes, take the coarse
   output, transform to frequency space (DCT), and synthesize the missing
   high-frequency bands by injecting noise scaled to the boundary's noise level.
3. **Fine pass.** Continue denoising at the full resolution. The synthesized
   high-frequency band gives the model something to refine rather than
   hallucinate.

For MiniMax-H3's packed video+audio latent, each stage issues a separate
`guider.sample()` call so the model rebuilds its shape-dependent conditioning
from the current buffer geometry.

```
                     ┌─────────────────────────────────────────────┐
                     │              latent (full-res)             │
                     │   shape = [1, 24, 37, 30, 54]  +  audio    │
                     └─────────────────────────────────────────────┘
                                       ▲
                                       │ stage 2: full-res denoise
                                       │ (kappa-aligned boundary sigma)
                                       │
                     ┌─────────────────────────────────────────────┐
                     │      spectral expand (DCT upsample)        │
                     │   low-band:  from coarse output             │
                     │   high-band: noise * boundary sigma         │
                     └─────────────────────────────────────────────┘
                                       ▲
                                       │ stage 1: half-res denoise
                                       │
                     ┌─────────────────────────────────────────────┐
                     │              latent (half-res)              │
                     │   shape = [1, 24, 19, 15, 27]  +  audio    │
                     └─────────────────────────────────────────────┘
```

## Concepts you'll see

- **`latent_shapes`** — Each H3 buffer has a specific shape (video T/H/W +
  audio length). The model uses this shape to construct cross-attention
  matrices. The shapes are baked into the guider's condition dict at the
  start of `guider.sample()` and reused for every model call. If the buffer
  changes shape mid-pass, the model crashes trying to reshape the attention
  tensors. SPEED works around this by issuing a fresh `guider.sample()` per
  stage — each call re-bakes the shapes from the current buffer.

- **`packed latent`** — MiniMax-H3 stores video and audio in a single
  `NestedTensor` with two streams: video `[1, 24, T, H, W]` and audio
  `[1, C, 2, T]`. Together they hold the joint video+audio content.
  VAEs decode each stream separately after denoising.

- **`kappa alignment`** — When SPEED transitions from coarse to fine, it
  computes a `kappa` rescale factor for the boundary sigma: the boundary
  noise level must be high enough that the seeded high-frequency band looks
  like natural detail to the model, not already-denised content. The formula
  is `kappa = r / (1 + (r-1)*q)` where `r` is the resolution ratio and `q` is
  the sigma at the boundary. Without this rescale, the next pass would either
  ignore the seeded band or smear it.

- **`sigma_shift_video` / `sigma_shift_audio`** — MiniMax-H3 has two
  separate sigma-shift parameters on its diffusion model (defaults 12.0 and
  3.0). They control how aggressively the model's time-shift curve bends
  sigma around its midpoint. Audio and video are denoised in different
  regimes, so they get different shifts. The ratio
  `sigma_shift_video / sigma_shift_audio` is used to rescale the audio
  sigma when transitioning between stages.

- **`DCT-expand`** — A discrete cosine transform on the spatial axes of the
  video latent. Coarse content fills the low-frequency band; seeded noise
  fills the high-frequency band. The inverse DCT gives a full-resolution
  latent ready for the next denoising pass.

## Why a direct node, not a SAMPLER

ComfyUI's `SamplerCustomAdvanced` calls a KSAMPLER's `sample_function` once
per denoising pass. H3's `latent_shapes` are baked once at the start of
`guider.sample()`, then reused across every model call in that pass. If a
sampler tries to shrink the buffer mid-pass (which SPEED's coarse→full
expansion requires), the model crashes at `utils.unpack_latents` because the
buffer is the wrong size.

The Lab's `MiniMaxH3SPEEDConfigurableStageCalls` works around this by issuing
one `guider.sample()` per stage — each stage re-bakes `latent_shapes` from the
current buffer. This MVP node (`MiniMaxH3SPEEDSampler`) does the same thing,
just with a flat package layout instead of the Lab's `speed_lab` namespace.
namespace.

## Install

```bash
git clone <this-repo> ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SPEED-Sampler
# restart ComfyUI
```

No additional dependencies beyond torch + ComfyUI.

## Usage

Load `workflows/video_minimax_h3_t2v_speed.json` in ComfyUI (or wire the chain
above manually). The defaults work for typical prompts.

> **Requires the MiniMax-H3 plugin** (or H3 base ComfyUI 0.32.0) to produce the
> packed AV latent. The bundled workflow uses node schemas verified against
> ComfyUI 0.32.0.

The node has six configuration knobs:

- `preset` — scale ladder shape; default `"2 stages, 0.5 → 1"`
- `transition_mode` — `explicit` (default) or `delta_custom`
- `delta` — boundary tolerance; default `0.01`
- `power_A`, `power_beta` — spectral model constants; defaults from SPEED paper (219.48, 2.42). Only used with `delta_custom`
- `seed_offset` — noise offset for the high-frequency band; default `10000`

Hover any knob in the ComfyUI UI for a tooltip. They are not the interesting
part of SPEED — `preset` is the only one worth thinking about normally;
everything else has a sensible default.

## Running on Modal

The sibling `modal/` config bakes this directory (not the full SPEED Lab) into
ComfyUI's `custom_nodes/` as a standalone V1 node, plus the workflow into the
default user dir:

```bash
cd modal && uv sync && modal serve comfyui.py   # auto-reloads on file change
```

Then in the ComfyUI UI: Workflow → Open → `video_minimax_h3_t2v_speed.json`.
Edit the node model filenames to match whatever a given ComfyUI install has
under `models/` (the bundled workflow targets the filenames at
`Comfy-Org/MiniMax-H3`).

After any change to this repo's Python files, force a Modal rebuild:

```bash
touch modal/comfyui.py
modal serve comfyui.py
```

## Repository structure

```
ComfyUI-MiniMaxH3-SPEED-Sampler/
├── __init__.py                — ComfyUI package entry; re-exports NODE_CLASS_MAPPINGS
├── speed_lab/                 — Main package
│   ├── __init__.py
│   ├── config.py              — SpeedConfig, SCALE_PRESETS, DEFAULT_TRANSITION_STEPS
│   ├── flow.py                — Transition math: sigma alignment, step resolution
│   ├── h3_runtime.py          — run_progressive_stages (THE CORE LOOP)
│   ├── spectral.py            — DCT primitives: dct2, idct2, spectral_expand_dct
│   ├── nodes/                 — ComfyUI node definitions
│   │   ├── __init__.py        — Re-exports NODE_CLASS_MAPPINGS
│   │   └── speed_sampler.py   — MiniMaxH3SPEEDSampler (the one ComfyUI node)
│   └── tests/                 — Test suite
│       ├── test_dct.py        — Pure DCT math (4 tests)
│       └── test_sampler.py    — Node contract + workflow JSON validation (7 tests)
├── workflows/
│   └── video_minimax_h3_t2v_speed.json  — Ready-to-load UI workflow
├── LICENSE.md
└── README.md
```

Standard ComfyUI custom node layout: `__init__.py` is the package entry,
`speed_lab/nodes/` contains the actual node class definitions.

## Modules

### `speed_sampler.py` — the ComfyUI node

The single user-facing class: `MiniMaxH3SPEEDSampler`. Takes
`noise`, `guider`, `sigmas`, `latent_image` as standard ComfyUI inputs plus
the SPEED widgets. Returns `(output, denoised_output)` LATENT tuple.

No DCT or transition math lives here — that's in `spectral.py` and `flow.py`
respectively.

### `config.py` — speed configuration

- `SCALE_PRESETS` — Dict of `name → tuple[float, ...]` scale ladders
- `DEFAULT_TRANSITION_STEPS` — Dict of `name → tuple[int, ...]` transition step indices per preset
- `SpeedConfig` — Frozen dataclass; validates config consistency at construction

### `flow.py` — transition math

Pure functions for sigma alignment:
- `aligned_speed_sigma(q, r)` — Compute kappa and aligned boundary sigma
- `time_shift_sigma(sigma, from_shift, to_shift)` — Time-shift sigma for audio
- `resolve_transition_steps(config, sigmas, H_full, W_full)` — Compute transition steps

The MVP also carries placeholders for `recover_internal_state`, `reentry_noise`,
`carry_preserving_audio_state`, `clock_reindex_audio_state` — they are
identity functions now, kept so the calling code structure (which branches on
`config.audio_policy` / `config.noise_policy`) doesn't need to change when
the full policies are implemented.

### `h3_runtime.py` — the SPEED loop

The meat of the implementation: `run_progressive_stages(noise, guider, sigmas, latent_image, config, ...)`.

For each SPEED stage:
1. Validate H3 nested latent (5D video + 4D stereo audio)
2. Read live `sigma_shift_video` / `sigma_shift_audio` from the model
3. Compute delta-optimal transition steps (when `transition_mode=delta_custom`)
4. Build coarse latent + coarse noise at `scales[stage_idx]`
5. Call `guider.sample()` for current stage's sigma slice
6. At boundary: recover internal state, apply kappa alignment, DCT-expand video, splice next-stage sigmas

### `spectral.py` — DCT primitives

Pure-torch orthonormal DCT-II, used by `h3_runtime.py` for video expansion:
- `dct2(value)` / `idct2(coeffs)` — 2D DCT-II and inverse
- `lowpass_dct(value, target_hw)` — DCT-truncate then inverse (smoothing)
- `spectral_expand_dct(value, target_hw, sigma, seed)` — Expand with seeded high-frequency fill
- `spectral_expand_dct_coupled(value, full_noise_video, sigma)` — Coupled expansion using precomputed full-res noise

## T2V SPEED workflow

`workflows/video_minimax_h3_t2v_speed.json` adapts ComfyUI's stock
`video_minimax_h3_t2v.json` T2V workflow. Same loader nodes
(UNETLoader, CLIPLoader, VAELoader×2), same `MiniMaxH3ImageToVideo`, but the
base `KSamplerSelect` + `SamplerCustomAdvanced` chain is **replaced** by
`MiniMaxH3SPEEDSampler` (no separate sampler + runner — this node drives
the denoising internally). The full chain:

```text
UNETLoader ──▶ BasicGuider ──┐
CLIPLoader ──▶ MiniMaxH3ImageToVideo ─▶ BasicScheduler
VAELoader  ──┘        │                    │
RandomNoise ──────────▶ MiniMaxH3SPEEDSampler
                        │
                        ├──▶ VAEDecode ──▶ CreateVideo(24fps)
                        └──▶ VAEDecodeAudio ─┘         │
                                              SaveVideo
```

`CreateVideo` is set to 24 fps (H3's native frame rate).

## Not in this MVP (deliberately)

- No sigma harvesting or calibration tooling
- No helper/diagnostic nodes (the Lab has 10 of these)
- No parity test against the Lab
- No sampler-type node variant (`MiniMaxH3SPEEDSampler` returns `("SAMPLER",)` and is fundamentally broken for H3 — see the Lab's `sampler_speed.py` for the same pattern if needed)
- No Euler alternatives — multi-sampler support is deferred research (calibrated only for Euler per the SPEED paper)
- No test against real H3 model weights (requires a ~40 GiB GPU host)

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md). This is a minimal
proof-of-concept; use it non-commercially at your own risk.
