"""MiniMax-H3 SPEED stage runner — self-contained correctness oracle.

Wraps each SPEED stage in a separate `guider.sample()` call so the H3 model
always sees a buffer matching its latent_shapes. Ported from the Lab's
`h3_runtime.py`.
"""

from __future__ import annotations

import math

import torch

from comfy import nested_tensor as default_comfy_nested_tensor

from minimax_h3_speed.config import SpeedConfig
from minimax_h3_speed.flow import (
    aligned_speed_sigma,
    carry_preserving_audio_state,
    clock_reindex_audio_state,
    recover_internal_state,
    reentry_noise,
    time_shift_sigma,
)
from minimax_h3_speed.spectral import (
    dct2,
    lowpass_dct,
    spectral_expand_dct,
    spectral_expand_dct_coupled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_tensor(samples):
    """Unpack a NestedTensor into (video, audio) with H3 geometry validation."""
    if not getattr(samples, "is_nested", False):
        raise ValueError("MiniMax-H3 SPEED requires a NestedTensor video/audio latent")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError("MiniMax-H3 SPEED requires exactly video and audio streams")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("expected H3 video [B,C,T,H,W] and audio [B,C,2,T]")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax-H3 supports batch size one")
    if audio.shape[2] != 2:
        raise ValueError("MiniMax-H3 audio latent requires stereo axis size two")
    return video, audio


def _pack_tensor(video, audio):
    return default_comfy_nested_tensor.NestedTensor([video, audio])



def _active_av_shifts(guider):
    """Return (video_shift, audio_shift, audio_scale) from the guider's model.

    Reads `sigma_shift_video` and `sigma_shift_audio` from the live H3 model
    (defaults: 12.0 and 3.0 respectively), then derives `audio_scale` as the
    ratio `video_shift / audio_shift`. This ratio is used by the audio time-
    shift rescale during stage transitions.

    Note: previously this function read `audio_scale` from the model — that
    attribute does not exist on MiniMax-H3, so it silently defaulted to 1.0,
    which is the wrong-by-4x bug that produced beige-wall artifacts on live
    weights. Fixed in this commit; see `tests/test_active_av_shifts_returns_
    audio_scale_from_ratio` for the regression test.
    """
    model = getattr(getattr(guider, "model_patcher", None), "model", None)
    if model is None:
        raise ValueError("guider has no model_patcher.model — cannot read H3 sigma shifts")
    shifts = getattr(model, "diffusion_model", model)
    video_shift = getattr(shifts, "sigma_shift_video", None)
    audio_shift = getattr(shifts, "sigma_shift_audio", None)
    if not (isinstance(video_shift, (int, float)) and isinstance(audio_shift, (int, float))):
        raise ValueError(
            f"active MiniMax-H3 sigma shifts are unavailable "
            f"(video={video_shift!r}, audio={audio_shift!r})"
        )
    video_shift = float(video_shift)
    audio_shift = float(audio_shift)
    if video_shift <= 0.0 or audio_shift <= 0.0:
        raise ValueError(
            f"active MiniMax-H3 shifts must be positive "
            f"(video={video_shift}, audio={audio_shift})"
        )
    return video_shift, audio_shift, video_shift / audio_shift


class _StageCapture:
    """Mutable container for stage-callback x0 output."""
    __slots__ = ("x0", "step")

    def __init__(self):
        self.x0 = None
        self.step = None

    def populated(self):
        return self.x0 is not None


def _audio_x0_continuation():
    """Create a callback that records x0 from a sampler stage run.

    Used by `clock_reindex` audio mode: the sampler callback gives us x0
    (the denoised output at this sigma), which is the "clean" audio we
    need to clock-reindex across stage boundaries. Without this callback
    we'd lose the denoised state between stages and audio would drift.
    """
    record = _StageCapture()

    def callback(step, x0, x, total_steps):
        record.x0 = x0
        record.step = step

    return record, callback


# ---------------------------------------------------------------------------
# Power spectrum helpers (for delta_custom mode — deferred in MVP)
# ---------------------------------------------------------------------------

def power_spectrum(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum P(omega) = A * |omega|^(-beta). Paper Eq. 8."""
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """Activation time for one radial frequency. Paper Eq. 9."""
    if delta >= 1.0:
        raise ValueError("delta must be < 1.0")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


def _find_first_step_below(sigmas, threshold: float) -> int:
    """First index whose sigma <= threshold; len-1 if none."""
    vals = [float(s) for s in sigmas]
    n = len(vals) - 1
    for i in range(n):
        if vals[i] <= threshold:
            return i
    return n


def resolve_transition_steps(config, sigmas, H_full, W_full):
    """Resolve per-stage transition sigma-step indices.

    When ``transition_mode == "delta_custom"``, uses the paper's proper
    spectral-energy math (compare sigma against the activation time of each
    radial frequency). When ``explicit``, uses the user-supplied step list.
    """
    if not isinstance(config, SpeedConfig):
        return list(config.transition_steps)

    scales = list(config.scales)
    if config.transition_mode == "delta_custom":
        tolerance = config.delta
        A, beta = config.power_A, config.power_beta
        # omega_max derives from min(H, W) at full resolution.
        omega_max = min(H_full, W_full) / 2.0
        steps = []
        for i in range(len(scales) - 1):
            omega_i = scales[i] * omega_max
            p = power_spectrum(omega_i, A, beta)
            thr = activation_time(p, tolerance)
            steps.append(_find_first_step_below(sigmas, thr))
        return steps
    return list(config.transition_steps)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_progressive_stages(
    noise,
    guider,
    sigmas: torch.Tensor,
    latent: dict,
    config: SpeedConfig,
    *,
    sampler,
    nested_type,
    disable_pbar: bool = True,
    output_device=None,
):
    """Run N-stage progressive-resolution Euler chain for MiniMax-H3.

    Each stage gets its own `guider.sample()` call, so `latent_shapes` is
    re-baked from the current buffer geometry and the model never sees a
    size mismatch.
    """
    if "noise_mask" in latent:
        raise ValueError("T2V oracle does not support noise masks")

    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent has no 'samples' key")

    # Unpack into video + audio streams.
    if getattr(samples, "is_nested", False):
        full_video, full_audio = _unpack_tensor(samples)
    else:
        raise ValueError("MiniMax-H3 requires a nested video/audio latent")

    video_shift, audio_shift, audio_scale = _active_av_shifts(guider)

    if torch.count_nonzero(full_video) or torch.count_nonzero(full_audio):
        raise ValueError("T2V oracle currently requires an empty H3 latent")

    if sigmas.ndim != 1 or len(sigmas) < 3:
        raise ValueError("sigmas must be a one-dimensional schedule")

    scales = config.scales
    n_stages = len(scales)
    if n_stages < 2:
        raise ValueError("need at least two stages (scales ending at 1.0)")

    full_h, full_w = full_video.shape[-2:]
    scale_hw = [
        (max(1, round(full_h * s)), max(1, round(full_w * s)))
        for s in scales
    ]

    # Resolve transition steps (delta-optimal or explicit).
    transition_steps = resolve_transition_steps(config, sigmas, full_h, full_w)
    if len(transition_steps) != n_stages - 1:
        raise ValueError("transition steps count must be n_scales - 1")
    for ts in transition_steps:
        if not 0 < int(ts) < len(sigmas) - 1:
            raise ValueError("transition step must be inside the sigma schedule")

    # Stage 1: initialize coarse latent.
    coarse_h, coarse_w = scale_hw[0]
    coarse_samples = _pack_tensor(
        full_video.new_zeros(full_video.shape[:-2] + (coarse_h, coarse_w)),
        torch.zeros_like(full_audio),
    )
    cur_latent = latent.copy()
    cur_latent["samples"] = coarse_samples

    # Generate noise (coarse noise for coupled policy, fresh for direct).
    full_noise = None
    if config.noise_policy == "coupled_full_grid":
        full_noise = noise.generate_noise(latent)
        _, full_noise_audio = _unpack_tensor(full_noise)
        coarse_noise = _pack_tensor(
            lowpass_dct(full_noise[0], (coarse_h, coarse_w)),
            full_noise_audio,
        )
    else:
        coarse_noise = noise.generate_noise(cur_latent)

    # Run each stage.
    current_sigmas = sigmas
    scale_pub = coarse_noise
    scale_latent = cur_latent["samples"]
    latest_public = None
    latest_capture = None

    for scale_idx in range(n_stages - 1):
        boundary = int(transition_steps[scale_idx])
        if boundary < 1 or boundary >= len(current_sigmas) - 1:
            raise ValueError(
                f"transition step {boundary} out of bounds "
                f"(schedule has {len(current_sigmas)} entries, valid: 1..{len(current_sigmas) - 2})"
            )

        capture, callback = _audio_x0_continuation()
        scale_sigmas = current_sigmas[: boundary + 1]
        public = guider.sample(
            scale_pub,
            scale_latent,
            sampler,
            scale_sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        latest_public = public
        latest_capture = capture

        public_video, public_audio = _unpack_tensor(public)
        boundary_sigma = float(current_sigmas[boundary])

        # Recover internal state.
        internal_video, internal_audio = recover_internal_state(
            public_video, public_audio, boundary_sigma, audio_scale
        )

        # Align (scale_ratio) for this transition.
        ratio = scales[scale_idx + 1] / scales[scale_idx]
        if config.sigma_policy == "canonical":
            scale_ratio, next_q = aligned_speed_sigma(boundary_sigma, ratio)
        else:
            scale_ratio, next_q = 1.0, boundary_sigma

        # DCT-expand the video (coupled or fresh band) and rescale by kappa.
        next_hw = scale_hw[scale_idx + 1]
        if config.noise_policy == "coupled_full_grid":
            assert full_noise is not None
            full_noise_video, _ = _unpack_tensor(full_noise)
            expanded_video = spectral_expand_dct_coupled(
                internal_video,
                full_noise_video.to(device=internal_video.device, dtype=internal_video.dtype),
                boundary_sigma,
            )
        else:
            expanded_video = spectral_expand_dct(
                internal_video,
                next_hw,
                boundary_sigma,
                int(noise.seed) + int(config.transition_seed_offset) + scale_idx,
            )
        transitioned_video = expanded_video * scale_ratio

        # Audio handling.
        old_audio_sigma = time_shift_sigma(boundary_sigma, video_shift, audio_shift)
        new_audio_sigma = time_shift_sigma(next_q, video_shift, audio_shift)
        if config.audio_policy == "carry_preserve":
            transitioned_audio = carry_preserving_audio_state(
                internal_audio, boundary_sigma, next_q, old_audio_sigma, new_audio_sigma
            )
        elif config.audio_policy == "clock_reindex":
            if not capture.populated():
                raise RuntimeError("clock_reindex requires an x0 callback")
            _, clean_audio = _unpack_tensor(capture.x0)
            transitioned_audio = clock_reindex_audio_state(
                internal_audio, clean_audio,
                boundary_sigma, next_q,
                old_audio_sigma, new_audio_sigma,
                audio_scale,
            )
        else:  # untouched
            transitioned_audio = internal_audio

        # Build next-scale schedule and latent.
        next_sigmas = torch.cat(
            [current_sigmas.new_tensor([next_q]), current_sigmas[boundary + 1:]],
            dim=0,
        )

        next_noise = _pack_tensor(
            reentry_noise(transitioned_video, next_q),
            reentry_noise(transitioned_audio, next_q),
        )
        next_zero = _pack_tensor(
            torch.zeros_like(transitioned_video),
            torch.zeros_like(transitioned_audio),
        )

        scale_pub = next_noise
        scale_latent = next_zero
        current_sigmas = next_sigmas

    # Final full-res stage.
    final_capture, final_callback = _audio_x0_continuation()
    final_public = guider.sample(
        scale_pub,
        scale_latent,
        sampler,
        current_sigmas,
        callback=final_callback,
        disable_pbar=disable_pbar,
        seed=noise.seed,
    )
    latest_public = final_public
    latest_capture = final_capture

    if output_device is not None and latest_public is not None:
        latest_public = latest_public.to(output_device)

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = latest_public

    denoised = out
    if latest_capture is not None and latest_capture.populated():
        denoised = latent.copy()
        x0 = latest_capture.x0
        if hasattr(x0, "cpu"):
            x0 = x0.cpu()
        denoised["samples"] = guider.model_patcher.model.process_latent_out(x0)
    return out, denoised
