"""Self-contained MiniMax-H3 SPEED Sampler node.

Combines DCT spectral expand, flow alignment, and adaptive stage execution
into a single file with no external sub-package requirements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import comfy.nested_tensor
import comfy.samplers
import comfy.utils

# ---------------------------------------------------------------------------
# 1. DCT Spectral Primitives
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _cached_basis(size: int, device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    sample = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequency = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((math.pi / size) * frequency * sample)
    basis[0] *= math.sqrt(1.0 / size)
    if size > 1:
        basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _basis(size: int, device: torch.device) -> torch.Tensor:
    return _cached_basis(size, device.type, device.index)


def dct2(value: torch.Tensor) -> torch.Tensor:
    work = value.float()
    height_basis = _basis(work.shape[-2], work.device)
    width_basis = _basis(work.shape[-1], work.device)
    transformed = torch.matmul(height_basis, work)
    return torch.matmul(transformed, width_basis.transpose(0, 1))


def idct2(coefficients: torch.Tensor) -> torch.Tensor:
    work = coefficients.float()
    height_basis = _basis(work.shape[-2], work.device)
    width_basis = _basis(work.shape[-1], work.device)
    restored = torch.matmul(height_basis.transpose(0, 1), work)
    return torch.matmul(restored, width_basis)


def lowpass_dct(value: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    return idct2(dct2(value)[..., :target_h, :target_w]).to(dtype=value.dtype)


def spectral_expand_dct_coupled(value: torch.Tensor, full_resolution_noise: torch.Tensor, sigma: float) -> torch.Tensor:
    source_h, source_w = value.shape[-2:]
    expanded = dct2(full_resolution_noise).float() * float(sigma)
    expanded[..., :source_h, :source_w] = dct2(value).float()
    return idct2(expanded).to(dtype=value.dtype)


def spectral_expand_dct(value: torch.Tensor, target_hw: tuple[int, int], sigma: float, seed: int) -> torch.Tensor:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    source_h, source_w = value.shape[-2:]
    source_coefficients = dct2(value).float()
    generator = torch.Generator(device=value.device).manual_seed(int(seed))
    expanded = torch.randn(value.shape[:-2] + (target_h, target_w), generator=generator, device=value.device, dtype=torch.float32)
    expanded.mul_(float(sigma))
    expanded[..., :source_h, :source_w] = source_coefficients
    return idct2(expanded).to(dtype=value.dtype)


# ---------------------------------------------------------------------------
# 2. Flow & Audio-Video Alignment Math
# ---------------------------------------------------------------------------

def aligned_speed_sigma(sigma: float, resolution_ratio: float) -> tuple[float, float]:
    q = float(sigma)
    ratio = float(resolution_ratio)
    kappa = ratio / (1.0 + (ratio - 1.0) * q)
    return kappa, q * kappa


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def recover_internal_state(video_public, audio_public, sigma: float, audio_scale: float):
    clean_weight = 1.0 - sigma
    return video_public * clean_weight, audio_public * audio_scale * clean_weight


def clock_reindex_audio_state(carried_audio, clean_carried_audio, old_video_sigma: float, new_video_sigma: float, old_audio_sigma: float, new_audio_sigma: float, audio_scale: float):
    current_native = carried_audio * old_audio_sigma / old_video_sigma
    clean_native = clean_carried_audio / audio_scale
    noise_native = (current_native - (1.0 - old_audio_sigma) * clean_native) / old_audio_sigma
    new_native = ((1.0 - new_audio_sigma) * clean_native + new_audio_sigma * noise_native)
    return new_native * new_video_sigma / new_audio_sigma


def reentry_noise(internal_state, start_sigma: float):
    return internal_state / max(start_sigma, 1e-12)


# ---------------------------------------------------------------------------
# 3. MiniMax H3 Runtime Execution
# ---------------------------------------------------------------------------

def _unpack_tensor(samples):
    streams = list(samples.unbind())
    return streams[0], streams[1]


def _pack_tensor(video, audio):
    return comfy.nested_tensor.NestedTensor([video, audio])


def _active_av_shifts(guider):
    patcher = getattr(guider, "model_patcher", None)
    model = getattr(patcher, "model", None)
    video_shift = getattr(model, "sigma_shift_video", 12.0)
    audio_shift = getattr(model, "sigma_shift_audio", 3.0)
    return float(video_shift), float(audio_shift), float(video_shift) / float(audio_shift)


def _capture():
    state = {}
    def callback(step, x0, x, total_steps):
        state["x0"] = x0
        state["x"] = x
    return state, callback


@dataclass(frozen=True)
class SpeedConfig:
    scales: tuple[float, ...] = (0.5, 1.0)
    transition_steps: tuple[int, ...] = (5,)
    noise_policy: str = "direct_coarse"
    transition_seed_offset: int = 10_000
    full_latent_h: int = 45
    full_latent_w: int = 80


def run_progressive_stages(noise, guider, sigmas: torch.Tensor, latent: dict, config: SpeedConfig, *, sampler, nested_type, disable_pbar: bool = True, output_device=None):
    samples = latent.get("samples")
    full_video, full_audio = _unpack_tensor(samples)
    video_shift, audio_shift, audio_scale = _active_av_shifts(guider)

    scales = config.scales
    n_stages = len(scales)
    full_h, full_w = full_video.shape[-2:]
    stage_hw = [(max(1, round(full_h * s)), max(1, round(full_w * s))) for s in scales]
    transition_steps = config.transition_steps

    # Stage 1: Coarse latent initialization
    s0_h, s0_w = stage_hw[0]
    coarse_samples = _pack_tensor(full_video.new_zeros(full_video.shape[:-2] + (s0_h, s0_w)), torch.zeros_like(full_audio))
    cur_latent = latent.copy()
    cur_latent["samples"] = coarse_samples

    full_noise = None
    if config.noise_policy == "coupled_full_grid":
        full_noise = noise.generate_noise(latent) if hasattr(noise, "generate_noise") else noise
        full_noise_video, full_noise_audio = _unpack_tensor(full_noise)
        coarse_noise = _pack_tensor(lowpass_dct(full_noise_video, (s0_h, s0_w)), full_noise_audio)
    else:
        coarse_noise = noise.generate_noise(cur_latent) if hasattr(noise, "generate_noise") else cur_latent["samples"]

    current_sigmas = sigmas
    stage_start_pub = coarse_noise
    stage_start_latent = cur_latent["samples"]
    last_public = None
    last_capture = None

    seed = getattr(noise, "seed", None)

    for stage_idx in range(n_stages - 1):
        boundary = min(int(transition_steps[stage_idx]), len(current_sigmas) - 2)
        capture, callback = _capture()
        stage_sigmas = current_sigmas[: boundary + 1]

        public = guider.sample(
            stage_start_pub, stage_start_latent, sampler, stage_sigmas,
            callback=callback, disable_pbar=disable_pbar, seed=seed
        )
        last_public, last_capture = public, capture
        public_video, public_audio = _unpack_tensor(public)
        q = float(current_sigmas[boundary])

        internal_video, internal_audio = recover_internal_state(public_video, public_audio, q, audio_scale)
        ratio = scales[stage_idx + 1] / scales[stage_idx]
        kappa, new_q = aligned_speed_sigma(q, ratio)

        next_hw = stage_hw[stage_idx + 1]
        if config.noise_policy == "coupled_full_grid":
            full_noise_video, _ = _unpack_tensor(full_noise)
            expanded_video = spectral_expand_dct_coupled(internal_video, full_noise_video.to(device=internal_video.device, dtype=internal_video.dtype), q)
        else:
            seed_val = (int(seed) if seed is not None else 0) + int(config.transition_seed_offset) + stage_idx
            expanded_video = spectral_expand_dct(internal_video, next_hw, q, seed_val)

        transitioned_video = expanded_video * kappa

        old_audio_sigma = time_shift_sigma(q, video_shift, audio_shift)
        new_audio_sigma = time_shift_sigma(new_q, video_shift, audio_shift)

        if "x0" in capture:
            _, clean_audio = _unpack_tensor(capture["x0"])
            transitioned_audio = clock_reindex_audio_state(internal_audio, clean_audio, q, new_q, old_audio_sigma, new_audio_sigma, audio_scale)
        else:
            transitioned_audio = internal_audio

        next_sigmas = torch.cat([current_sigmas.new_tensor([new_q]), current_sigmas[boundary + 1:]], dim=0)
        stage_start_pub = _pack_tensor(reentry_noise(transitioned_video, new_q), reentry_noise(transitioned_audio, new_q))
        stage_start_latent = _pack_tensor(torch.zeros_like(transitioned_video), torch.zeros_like(transitioned_audio))
        current_sigmas = next_sigmas

    # Final Stage
    final_capture, final_callback = _capture()
    final_public = guider.sample(
        stage_start_pub, stage_start_latent, sampler, current_sigmas,
        callback=final_callback, disable_pbar=disable_pbar, seed=seed
    )
    last_public, last_capture = final_public, final_capture

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = last_public

    denoised = out
    if last_capture is not None and "x0" in last_capture:
        x0 = last_capture["x0"]
        if getattr(x0, "is_nested", False):
            x0_streams = list(x0.unbind())
            x0_video = next((s for s in x0_streams if s.ndim == 5), None)
            if x0_video is not None:
                x0 = x0_video
        denoised = latent.copy()
        denoised["samples"] = guider.model_patcher.model.process_latent_out(x0.cpu() if hasattr(x0, "cpu") else x0)

    return out, denoised


# ---------------------------------------------------------------------------
# 4. Adaptive Sampler Node Interface
# ---------------------------------------------------------------------------

PRESET_MAPPING = {
    "Half -> Full (0.5x -> 1.0x) [Balanced / Recommended]": {"scales": (0.5, 1.0), "ratios": (0.25,)},
    "Three-Quarter -> Full (0.75x -> 1.0x) [Fastest]": {"scales": (0.75, 1.0), "ratios": (0.50,)},
    "Quarter -> Half -> Full (3-Stage) [High Detail]": {"scales": (0.25, 0.5, 1.0), "ratios": (0.15, 0.25)},
    "Quarter -> 3/4 -> Full (Aggressive)": {"scales": (0.25, 0.75, 1.0), "ratios": (0.15, 0.40)},
    "Quarter -> Half -> 3/4 -> Full (4-Stage) [Slow / Quality]": {"scales": (0.25, 0.5, 0.75, 1.0), "ratios": (0.15, 0.25, 0.40)},
}


def calculate_adaptive_steps(preset_name: str, total_steps: int, coarse_override: int = 0) -> tuple[tuple[float, ...], tuple[int, ...]]:
    preset_info = PRESET_MAPPING.get(preset_name, PRESET_MAPPING["Half -> Full (0.5x -> 1.0x) [Balanced / Recommended]"])
    scales, ratios = preset_info["scales"], preset_info["ratios"]
    n_transitions = len(scales) - 1

    if total_steps <= 1:
        return (1.0,), ()

    if coarse_override > 0 and n_transitions == 1:
        step = max(1, min(total_steps - 1, int(coarse_override)))
        return scales, (step,)

    steps = []
    last_step = 0
    for i, ratio in enumerate(ratios):
        target = round(total_steps * ratio)
        min_allowed = last_step + 1
        max_allowed = total_steps - (n_transitions - i)
        step = max(min_allowed, min(max_allowed, target))
        steps.append(step)
        last_step = step

    if len(steps) != len(set(steps)) or steps[-1] >= total_steps:
        scales = (0.5, 1.0)
        steps = [max(1, min(total_steps - 1, round(total_steps * 0.25)))]

    return scales, tuple(steps)


class MiniMaxH3SPEEDSampler:
    """All-in-one SPEED progressive-resolution sampler for MiniMax-H3."""

    DESCRIPTION = (
        "SPEED progressive-resolution sampler for MiniMax-H3. Denoises initial layout "
        "at lower resolution, then DCT-expands to full resolution for crisp details. "
        "Supports Turbo LoRAs (4, 6, 8 steps) and standard schedules (20+ steps)."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "preset": (list(PRESET_MAPPING.keys()), {
                    "default": "Half -> Full (0.5x -> 1.0x) [Balanced / Recommended]"
                }),
                "coarse_steps_override": ("INT", {
                    "default": 0, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Manual coarse steps for Turbo LoRAs (e.g. 1 or 2). Set 0 for auto."
                }),
                "noise_policy": (["direct_coarse", "coupled_full_grid"], {"default": "direct_coarse"}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, preset, coarse_steps_override=0, noise_policy="direct_coarse", seed_offset=10000):
        total_steps = len(sigmas) - 1
        if total_steps < 1:
            raise ValueError("Sigmas schedule must contain at least 1 step.")

        scales, transition_steps = calculate_adaptive_steps(
            preset_name=preset, total_steps=total_steps, coarse_override=coarse_steps_override
        )

        full_video, _ = _unpack_tensor(latent_image.get("samples"))

        config = SpeedConfig(
            scales=scales,
            transition_steps=transition_steps,
            noise_policy=noise_policy,
            transition_seed_offset=int(seed_offset),
            full_latent_h=int(full_video.shape[-2]),
            full_latent_w=int(full_video.shape[-1]),
        )

        return run_progressive_stages(
            noise, guider, sigmas, latent_image, config,
            sampler=comfy.samplers.sampler_object("euler"),
            nested_type=comfy.nested_tensor.NestedTensor,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            output_device=None,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"}