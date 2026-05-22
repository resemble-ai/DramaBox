#!/usr/bin/env python3
"""
Warm TTS server — loads models once, accepts requests via stdin or function call.

The key insight: inference.py spends 11s on Gemma + 8s on model load every call.
This server loads everything once and keeps it warm.

We import and call the same code paths as inference.py but cache the heavy objects.
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio

# Setup paths
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR / "ltx2"))
sys.path.insert(0, str(APP_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from audio_conditioning import AudioConditionByReferenceLatent
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, PromptEncoder
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop
from safetensors import safe_open


DEFAULT_NEG = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"


def estimate_duration(prompt, multiplier=1.1):
    """Defer to the shared sentence-aware + non-verbal action budget estimator
    so warm-server outputs match the lengths of the per-call CLI runs."""
    from duration_estimator import estimate_speech_duration
    base = estimate_speech_duration(prompt)
    return max(3.0, round(base * multiplier, 1))


def _equal_power_crossfade(prev: torch.Tensor, nxt: torch.Tensor,
                           sample_rate: int, fade_ms: float = 50.0) -> torch.Tensor:
    """Equal-power crossfade concat: ``[prev | nxt]`` with a smooth boundary.

    Both tensors are (C, T). Returns (C, T_prev + T_nxt - T_fade).

    Equal-power (cos/sin envelopes) keeps perceived loudness constant through
    the join — unlike a linear fade, which dips by ~3 dB in the middle when
    the two sources are uncorrelated. Default 50 ms is short enough to be
    inaudible on speech while still masking any waveform-level discontinuity
    between independently-generated chunks.
    """
    fade_samples = int(round(fade_ms * 1e-3 * sample_rate))
    fade_samples = max(1, min(fade_samples, prev.shape[-1], nxt.shape[-1]))
    if fade_samples <= 1:
        return torch.cat([prev, nxt], dim=-1)

    t = torch.linspace(0.0, 1.0, fade_samples, device=prev.device, dtype=prev.dtype)
    fade_out = torch.cos(t * torch.pi / 2)   # 1.0 -> 0.0
    fade_in = torch.sin(t * torch.pi / 2)    # 0.0 -> 1.0

    prev_tail = prev[..., -fade_samples:] * fade_out
    nxt_head = nxt[..., :fade_samples] * fade_in
    mixed = prev_tail + nxt_head
    return torch.cat([prev[..., :-fade_samples], mixed, nxt[..., fade_samples:]], dim=-1)


def auto_rescale_for_cfg(cfg: float) -> float:
    """CFG-aware std-rescale schedule that prevents output clipping at high cfg.

    The CFG formula `pred = cond + (cfg-1)*(cond - uncond)` makes pred.std()
    grow roughly linearly with cfg, which the audio VAE+vocoder render as
    progressively louder waveforms. By cfg≈3 the output starts hard-clipping
    at 0 dBFS — and clipped information is unrecoverable in post.

    Empirical sweep on the blues prompt with the back-porch-boogie ref
    (rescale_scale needed for ≥1 dB peak headroom):
        cfg=2.5 → 0.2 ;  cfg=3 → 0.6 ; cfg=4 → 0.8 ; cfg=5–8 → 0.8 ; cfg=10 → 1.0

    Piecewise-linear fit through those points; returns 0 below cfg=2 (no CFG
    even applied at cfg=1), plateaus at 0.8 between cfg=4 and cfg=8 to
    preserve the "extra punch" of high-CFG generations, and ramps to 1.0 by
    cfg=10.
    """
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)               # 0 → 0.6
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)         # 0.6 → 0.8
    if cfg <= 8.0:
        return 0.8                              # plateau
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))   # 0.8 → 1.0 at cfg=10


class TTSServer:
    def __init__(self, checkpoint=None, full_checkpoint=None, gemma_root=None,
                 device="cuda", dtype="bf16", compile_model=True, bnb_4bit=True):
        MODELS = APP_DIR / "models"
        self.checkpoint = checkpoint or str(MODELS / "ltx-2.3-22b-dev-audio-only-v13-merged.safetensors")
        self.full_checkpoint = full_checkpoint or os.environ.get(
            "LTX_FULL_CHECKPOINT", "/mnt/persistent0/manmay/models/ltx23/ltx-2.3-22b-dev.safetensors")
        if gemma_root is None and not os.environ.get("GEMMA_DIR"):
            from model_downloader import get_gemma_path
            gemma_root = get_gemma_path()
        self.gemma_root = gemma_root or os.environ["GEMMA_DIR"]
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        self.compile_model = compile_model
        self.bnb_4bit = bnb_4bit
        self.patchifier = AudioPatchifier(patch_size=1)

        # Cached models
        self._prompt_encoder = None
        self._velocity_model = None
        self._audio_conditioner = None
        self._audio_decoder = None
        # RE-USE denoiser for the voice reference (input-side denoise).
        # Lazy-loaded on first use; the cleaned-waveform cache below keeps
        # chunked generations from re-denoising the same 10 s clip per chunk.
        self._ref_denoiser = None
        self._ref_denoise_cache: dict[tuple, "torch.Tensor"] = {}

        logging.info(f"TTSServer loading on {device}...")
        t0 = time.time()
        self._load_all()
        logging.info(f"All models loaded in {time.time()-t0:.1f}s — ready for requests")

    def _load_all(self):
        # 1. Prompt encoder (Gemma + embeddings processor kept warm)
        t0 = time.time()
        self._prompt_encoder = PromptEncoder(
            checkpoint_path=self.full_checkpoint,
            gemma_root=self.gemma_root,
            dtype=self.dtype, device=self.device,
            warm=True,
            use_bnb_4bit=self.bnb_4bit,
            audio_only=True,
        )
        logging.info(f"  PromptEncoder (warm): {time.time()-t0:.1f}s")

        # 2. Audio conditioner (VAE encoder kept warm)
        t0 = time.time()
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device,
            warm=True,
        )
        logging.info(f"  AudioConditioner (warm): {time.time()-t0:.1f}s")

        # 3. Transformer
        t0 = time.time()
        with safe_open(self.checkpoint, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        t = config.get("transformer", {})

        class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, cfg):
                t = cfg.get("transformer", {})
                cp = None
                if not t.get("caption_proj_before_connector", False):
                    with torch.device("meta"):
                        cp = create_caption_projection(t, audio=True)
                return LTXModel(
                    model_type=LTXModelType.AudioOnly,
                    audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                    audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                    audio_in_channels=t.get("audio_in_channels", 128),
                    audio_out_channels=t.get("audio_out_channels", 128),
                    num_layers=t.get("num_layers", 48),
                    audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                    norm_eps=t.get("norm_eps", 1e-6),
                    attention_type=AttentionFunction(t.get("attention_type", "default")),
                    positional_embedding_theta=10000.0,
                    audio_positional_embedding_max_pos=[20.0],
                    timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                    use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                    rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                    double_precision_rope=t.get("frequencies_precision", False) == "float64",
                    apply_gated_attention=t.get("apply_gated_attention", False),
                    audio_caption_projection=cp,
                    cross_attention_adaln=t.get("cross_attention_adaln", False),
                )

        audio_sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
            "model.diffusion_model.", "")
        builder = Builder(
            model_path=self.checkpoint,
            model_class_configurator=AudioOnlyConfigurator,
            model_sd_ops=audio_sd_ops,
            registry=DummyRegistry(),
        )
        self._velocity_model = builder.build(device=self.device, dtype=self.dtype).to(self.device).eval()
        n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
        vram_gb = sum(p.numel() * p.element_size() for p in self._velocity_model.parameters()) / 1e9
        logging.info(f"  Transformer: {time.time()-t0:.1f}s ({n_params:.1f}B params, {vram_gb:.1f}GB VRAM, {self.dtype})")

        # torch.compile for faster denoising
        if self.compile_model:
            t0 = time.time()
            logging.info("  Compiling transformer with torch.compile (default mode)...")
            self._velocity_model = torch.compile(self._velocity_model, mode="default", dynamic=True)
            logging.info(f"  Compiled: {time.time()-t0:.1f}s (first call triggers actual compilation)")

        # 4. Audio decoder (VAE decoder + vocoder kept warm)
        t0 = time.time()
        self._audio_decoder = AudioDecoder(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device,
            warm=True,
        )
        logging.info(f"  AudioDecoder (warm): {time.time()-t0:.1f}s")

    def _denoise_voice_ref(self, voice, voice_ref_path: str, ref_duration: float):
        """Run RE-USE on the loaded voice reference and replace its waveform
        with a cleaned mono signal.

        Why pre-condition rather than post-generate: applying RE-USE to the
        *output* suppresses paralinguistic events the model generates (laughs,
        gasps, breaths, sighs) because they're broadband, non-tonal — exactly
        what universal speech enhancement targets as "noise". Running it on
        the *reference* instead gives the model a clean speaker / style
        anchor, which it generalises from at inference time, while leaving
        the generated paralinguistic content untouched.

        Cached by ``(path, ref_duration, sampling_rate)`` so chunked
        generations don't re-denoise the same 10 s clip per chunk.
        """
        cache_key = (voice_ref_path, float(ref_duration), int(voice.sampling_rate))
        if cache_key in self._ref_denoise_cache:
            return Audio(
                waveform=self._ref_denoise_cache[cache_key],
                sampling_rate=voice.sampling_rate,
            )

        # Lazy-load the denoiser. target_sr = input sr → no librosa resample
        # round-trip; RE-USE does pure denoise. (The 48 kHz BWE that
        # REUSEUpsampler can do is irrelevant here — the VAE conditioner
        # resamples internally to whatever the audio branch expects.)
        if self._ref_denoiser is None:
            from super_resolution import REUSEUpsampler
            try:
                self._ref_denoiser = REUSEUpsampler(
                    target_sr=int(voice.sampling_rate),
                    device=self.device,
                    chunk_size_s=1.0,
                )
            except Exception as e:
                # Mamba kernels / weights missing → silently skip the denoise
                # rather than blocking generation. Surfaces once per session.
                logging.warning(f"Voice-ref denoise disabled (RE-USE unavailable: {e})")
                self._ref_denoiser = False  # sentinel: don't retry this session
                return voice

        if self._ref_denoiser is False:
            return voice

        w = voice.waveform
        # Collapse to mono — voice cloning is speaker-as-mono-source; we'll
        # re-broadcast back to stereo after the conditioner.
        if w.dim() == 3:
            mono = w[0].mean(dim=0)
        elif w.dim() == 2:
            mono = w.mean(dim=0)
        else:
            mono = w
        mono = mono.contiguous()

        t0 = time.time()
        cleaned, _ = self._ref_denoiser(mono, in_sr=int(voice.sampling_rate))
        if cleaned.dim() == 2 and cleaned.shape[0] == 1:
            cleaned = cleaned[0]
        # Restore the (1, C=1, T) shape that the rest of the pipeline expects
        # to consume — downstream code re-expands channels via repeat().
        cleaned = cleaned.unsqueeze(0).unsqueeze(0).to(self.device, dtype=w.dtype)
        logging.info(f"Voice-ref denoise (RE-USE): {time.time() - t0:.2f}s")

        self._ref_denoise_cache[cache_key] = cleaned
        return Audio(waveform=cleaned, sampling_rate=voice.sampling_rate)

    @torch.inference_mode()
    def generate(self, prompt, voice_ref=None, cfg_scale=2.5, stg_scale=1.5,
                 duration_multiplier=1.1, seed=42, ref_duration=10.0,
                 rescale_scale="auto", gen_duration: float = 0.0,
                 denoise_ref: bool = True):
        """Generate audio. Returns (waveform_path, duration_seconds).

        rescale_scale: latent-side CFG std-rescale that prevents clipping at
            high cfg. Set to "auto" (default) for the cfg-aware schedule, a
            float in [0, 1] for a fixed override, or 0 to disable.
        gen_duration: explicit target duration in seconds. 0 (default) → auto
            from prompt + duration_multiplier; >0 overrides everything else.
        denoise_ref: when True (default) and a voice reference is provided,
            RE-USE is applied to the *reference* before VAE encoding so the
            model conditions on a clean speaker / style anchor. Generated
            output (24→48 kHz) always goes through the LTX BigVGAN BWE.
        """
        t_total = time.time()

        # Duration + target shape — explicit gen_duration wins over the estimator.
        if gen_duration and gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            gen_dur = estimate_duration(prompt, duration_multiplier)
        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(patchifier=self.patchifier, target_shape=target_shape)

        # Initial state
        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)

        # Voice ref conditioning
        if voice_ref and os.path.exists(voice_ref):
            t0 = time.time()
            voice = decode_audio_from_file(voice_ref, self.device, 0.0, ref_duration)
            if denoise_ref:
                voice = self._denoise_voice_ref(voice, voice_ref, ref_duration)
            w = voice.waveform
            if w.dim() == 2:
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                w = w.unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            target_samples = int(ref_duration * voice.sampling_rate)
            if w.shape[-1] < target_samples:
                w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
            w = w[..., :target_samples]
            peak = w.abs().max()
            if peak > 0:
                w = w * (10 ** (-4.0 / 20) / peak)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
            ref_latent = self._audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
            cond = AudioConditionByReferenceLatent(latent=ref_latent.to(self.device, self.dtype), strength=1.0)
            state = cond.apply_to(state, audio_tools)
            logging.info(f"Voice ref: {time.time()-t0:.2f}s")

        # Noise
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)

        # Prompt encode
        t0 = time.time()
        prompts = [prompt, DEFAULT_NEG] if cfg_scale > 1.0 else [prompt]
        ctx = self._prompt_encoder(prompts, streaming_prefetch_count=None)
        a_ctx = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if cfg_scale > 1.0 else None
        logging.info(f"Prompt: {time.time()-t0:.2f}s")

        # Denoiser
        resc = auto_rescale_for_cfg(cfg_scale) if rescale_scale == "auto" else float(rescale_scale)
        if rescale_scale == "auto":
            logging.info(f"Auto rescale_scale = {resc:.2f} for cfg={cfg_scale}")
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                stg_blocks=[29], rescale_scale=resc, modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None, a_context=a_ctx,
            video_guider=None, audio_guider=guider,
        )

        # Sigmas
        sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(self.device)

        # Denoise
        t0 = time.time()
        x0 = X0Model(self._velocity_model)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas, video_state=None, audio_state=state,
            stepper=EulerDiffusionStep(), transformer=x0, denoiser=denoiser,
        )
        logging.info(f"Denoise (30 steps): {time.time()-t0:.2f}s")

        # Strip + unpatchify + decode
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # End-of-clip silence-prior fix.
        # The base LTX-2.3 22B DiT was trained on audio clips ≤ ~20 s and
        # learned a strong "clip-end silence" prior that lands on the next
        # patchifier-aligned latent frame after 20 s — index 513 = 8*64+1.
        # When inference produces longer audio, this prior leaks through as a
        # high-norm latent burst at frame 513 (and adjacent 512), which the
        # audio VAE + vocoder render as a ~30 ms hard silence dip near 20.4 s.
        # Linear interpolation across the two affected frames removes the dip
        # cleanly without any retraining. Only runs when the latent is long
        # enough to actually contain the boundary.
        latent = audio_state.latent
        if latent.shape[2] > 513:
            f0, f1 = 511, 514          # neighbours used for interpolation
            n = f1 - f0                # = 3
            patched = latent.clone()
            for f in (512, 513):
                t = (f - f0) / n
                patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
            latent = patched

        t0 = time.time()
        decoded = self._audio_decoder(latent)
        out_waveform, out_sr = decoded.waveform, decoded.sampling_rate
        logging.info(f"Decode (LTX BWE): {time.time()-t0:.2f}s")

        total = time.time() - t_total
        dur = out_waveform.shape[-1] / out_sr
        logging.info(f"Total: {total:.2f}s for {dur:.1f}s audio")
        return out_waveform, out_sr

    @torch.inference_mode()
    def generate_long(self, prompt, max_chunk_duration: float = 45.0,
                      target_chunk_duration: float = 37.0,
                      crossfade_ms: float = 50.0,
                      progress_callback=None,
                      **kwargs):
        """Chunk-and-stitch generation for prompts whose estimated duration
        exceeds ``max_chunk_duration``.

        Splits ``prompt`` into <= ``max_chunk_duration`` chunks via
        :func:`text_chunker.chunk_prompt_for_duration`, generates each one
        through :meth:`generate` (same voice reference + seed for every
        chunk, so speaker identity stays coherent across joins), and
        concatenates the waveforms with an equal-power crossfade.

        Returns ``(waveform, sample_rate)`` matching :meth:`generate`.
        """
        from text_chunker import chunk_prompt_for_duration

        # gen_duration / duration_multiplier are per-chunk; pop them out so we
        # control sizing here and forward only the per-chunk values.
        per_chunk_mul = float(kwargs.pop("duration_multiplier", 1.1))
        # gen_duration coming in as a global target only makes sense for the
        # single-shot path; chunked generation derives durations per chunk.
        kwargs.pop("gen_duration", None)

        chunks = chunk_prompt_for_duration(
            prompt,
            max_duration_s=max_chunk_duration,
            target_duration_s=target_chunk_duration,
            duration_multiplier=per_chunk_mul,
        )
        logging.info(f"Long-form: {len(chunks)} chunks (target {target_chunk_duration:.0f}s, "
                     f"max {max_chunk_duration:.0f}s)")

        out_waveform: Optional[torch.Tensor] = None
        out_sr: Optional[int] = None
        t_total = time.time()
        for idx, chunk in enumerate(chunks):
            logging.info(f"  Chunk {idx + 1}/{len(chunks)}: est {chunk.est_duration_s:.1f}s, "
                         f"{len(chunk.text)} chars")
            if progress_callback is not None:
                try:
                    progress_callback(idx, len(chunks), chunk.est_duration_s)
                except Exception as e:
                    logging.warning(f"progress_callback raised, ignoring: {e}")
            wav, sr = self.generate(
                chunk.text,
                duration_multiplier=per_chunk_mul,
                **kwargs,
            )
            wav = wav.cpu().float()
            if out_waveform is None:
                out_waveform, out_sr = wav, sr
            else:
                if sr != out_sr:
                    raise RuntimeError(f"Sample-rate mismatch between chunks: {out_sr} vs {sr}")
                # Align channel counts: stereo crossfade with a mono buddy
                # broadcasts cleanly via torch.cat after equalising dim 0.
                if wav.shape[0] != out_waveform.shape[0]:
                    if wav.shape[0] == 1:
                        wav = wav.repeat(out_waveform.shape[0], 1)
                    elif out_waveform.shape[0] == 1:
                        out_waveform = out_waveform.repeat(wav.shape[0], 1)
                out_waveform = _equal_power_crossfade(out_waveform, wav, out_sr,
                                                      fade_ms=crossfade_ms)

        total_dur = out_waveform.shape[-1] / out_sr
        logging.info(f"Long-form total: {time.time() - t_total:.2f}s wall, {total_dur:.1f}s audio")
        return out_waveform, out_sr

    def generate_to_file(self, prompt, output, watermark: bool = True,
                         max_chunk_duration: float = 45.0,
                         target_chunk_duration: float = 37.0,
                         crossfade_ms: float = 50.0,
                         progress_callback=None,
                         **kwargs):
        # Auto-route to generate_long when the requested duration (explicit
        # gen_duration if set, otherwise prompt-estimated) exceeds the chunk
        # cap. Single-shot path otherwise — same as before, no regression for
        # short prompts.
        explicit_dur = float(kwargs.get("gen_duration") or 0.0)
        est_dur = explicit_dur if explicit_dur > 0 else estimate_duration(
            prompt, kwargs.get("duration_multiplier", 1.1))

        if est_dur > max_chunk_duration:
            waveform, sr = self.generate_long(
                prompt,
                max_chunk_duration=max_chunk_duration,
                target_chunk_duration=target_chunk_duration,
                crossfade_ms=crossfade_ms,
                progress_callback=progress_callback,
                **kwargs,
            )
        else:
            if progress_callback is not None:
                try:
                    progress_callback(0, 1, est_dur)
                except Exception:
                    pass
            waveform, sr = self.generate(prompt, **kwargs)
        wav_cpu = waveform.cpu().float()
        if watermark:
            try:
                import numpy as np, perth
                if not hasattr(self, "_perth"):
                    self._perth = perth.PerthImplicitWatermarker()
                mono = wav_cpu.mean(dim=0).numpy() if wav_cpu.shape[0] > 1 else wav_cpu[0].numpy()
                mono_wm = self._perth.apply_watermark(mono, sample_rate=sr)
                mono_wm_t = torch.from_numpy(np.asarray(mono_wm, dtype=np.float32)).unsqueeze(0)
                wav_cpu = mono_wm_t if wav_cpu.shape[0] == 1 else mono_wm_t.repeat(wav_cpu.shape[0], 1)
            except Exception as e:
                logging.warning(f"Perth watermark skipped ({e})")
        torchaudio.save(output, wav_cpu, sr)
        logging.info(f"Saved: {output}")
        return output


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bnb-4bit", action="store_true",
                   help="Disable bitsandbytes 4-bit path (default: on, since the default "
                        "unsloth Gemma checkpoint is pre-quantized).")
    args = p.parse_args()

    server = TTSServer(device=args.device, dtype=args.dtype, compile_model=not args.no_compile,
                       bnb_4bit=not args.no_bnb_4bit)

    # First call - includes any warmup
    logging.info("=== First request ===")
    server.generate_to_file(
        prompt='A woman speaks clearly, "The weather today will be sunny."',
        output="/tmp/warm_test1.wav",
        voice_ref="/mnt/persistent0/manmay/expressive/female_radio_nikole/female_radio_nikole.wav",
    )

    # Second call - should be much faster (models already warm)
    logging.info("\n=== Second request (warm) ===")
    server.generate_to_file(
        prompt='A man speaks excitedly, "This is amazing, I cannot believe it!"',
        output="/tmp/warm_test2.wav",
        voice_ref="/mnt/persistent0/manmay/expressive/male_arnie/male_arnie.mp3",
    )
