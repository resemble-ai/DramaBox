"""Optional external super-resolution / bandwidth-extension stages.

Replaces DramaBox's built-in 24->48 kHz BWE (the second BigVGAN inside
``VocoderWithBWE``) with one of:

  - ``ap_bwe``  : yxlu-0102/AP-BWE - feed-forward ConvNeXt amp+phase predictor.
  - ``audiosr`` : haoheliu/versatile_audio_super_resolution - latent diffusion.

Both expose the same surface:

    up = make_upsampler("ap_bwe", device="cuda")
    wav_48k, sr = up(wav_24k, in_sr=24000)        # wav_24k: (C, T) float

The wrappers lazy-load weights on first call so importing this module is cheap.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO = Path(__file__).resolve().parent.parent
AP_BWE_DIR = REPO / "third_party" / "AP-BWE"
AUDIOSR_DIR = REPO / "third_party" / "audiosr"
# REUSE_DIR is resolved lazily via model_downloader.get_reuse_code_path on
# first use of REUSEUpsampler — it returns the vendored third_party/RE-USE/
# tree if present, otherwise snapshot-downloads just the code from HF.
_REUSE_DIR: Optional[Path] = None


def _resolve_reuse_dir() -> Path:
    global _REUSE_DIR
    if _REUSE_DIR is None:
        from model_downloader import get_reuse_code_path
        _REUSE_DIR = Path(get_reuse_code_path())
    return _REUSE_DIR


# ---------------------------------------------------------------------------
# Bypass helper - pulls the 24 kHz waveform out of the LTX audio chain without
# running the second BigVGAN (the built-in BWE residual).
# ---------------------------------------------------------------------------


def decode_base_24k(audio_decoder_block, latent: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Run VAE decoder + base vocoder only (skip the LTX BWE residual).

    ``audio_decoder_block`` is a ``ltx_pipelines.utils.blocks.AudioDecoder``
    instance (warm-loaded). Its ``_warm_vocoder`` is a ``VocoderWithBWE`` whose
    ``.vocoder`` attribute is the base BigVGAN that natively outputs 24 kHz.
    """
    decoder = audio_decoder_block._warm_decoder
    bwe_wrapper = audio_decoder_block._warm_vocoder
    if decoder is None or bwe_wrapper is None:
        raise RuntimeError("AudioDecoder must be warm-loaded to use base 24 kHz bypass")

    base_vocoder = bwe_wrapper.vocoder
    base_sr = bwe_wrapper.input_sampling_rate  # 24000

    features = decoder(latent)
    # Match VocoderWithBWE's autocast-to-fp32 - bf16 accumulation through the
    # 60-layer BigVGAN degrades spectral metrics by 40-90%.
    with torch.autocast(device_type=features.device.type, dtype=torch.float32):
        waveform = base_vocoder(features.float())
    waveform = waveform.squeeze(0).float()  # (C, T)
    return waveform.clamp(-1.0, 1.0), int(base_sr)


# ---------------------------------------------------------------------------
# AP-BWE
# ---------------------------------------------------------------------------


class APBWEUpsampler:
    """Feed-forward bandwidth extension via AP-BWE (yxlu-0102/AP-BWE).

    Loads ``models/model.py:APNet_BWE_Model`` from the vendored repo at
    ``third_party/AP-BWE``. The variant (8/12/16/24 -> 48 kHz) is auto-picked
    from the input sample rate on first ``__call__``, so this works with any
    LTX vocoder rate. Override with ``checkpoint_path`` / ``config_path`` to
    pin a specific variant.

    Checkpoints come from the official Google Drive folder:
    https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF
    and should land at ``third_party/AP-BWE/checkpoints/<N>kto48k/g_<N>kto48k``.
    Override that location via the ``AP_BWE_CHECKPOINT`` env var.
    """

    SUPPORTED_LR_KHZ = (8, 12, 16, 24)  # all variants that target 48 kHz output

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self._checkpoint_override = (
            Path(checkpoint_path) if checkpoint_path
            else (Path(os.environ["AP_BWE_CHECKPOINT"]) if os.environ.get("AP_BWE_CHECKPOINT")
                  else None)
        )
        self._config_override = Path(config_path) if config_path else None
        self.config_path: Optional[Path] = None     # filled in by _resolve_paths
        self.checkpoint_path: Optional[Path] = None
        self._model = None
        self._h = None

    def _resolve_paths(self, in_sr: int) -> None:
        """Pick the AP-BWE variant whose ``lr_sampling_rate`` matches ``in_sr``."""
        lr_khz = in_sr // 1000
        if self._config_override:
            self.config_path = self._config_override
        else:
            if lr_khz not in self.SUPPORTED_LR_KHZ:
                raise ValueError(
                    f"AP-BWE has no <{lr_khz}>kto48k variant. "
                    f"Supported low-rate inputs: {self.SUPPORTED_LR_KHZ} kHz."
                )
            self.config_path = AP_BWE_DIR / "configs" / f"config_{lr_khz}kto48k.json"
        if self._checkpoint_override:
            self.checkpoint_path = self._checkpoint_override
        else:
            self.checkpoint_path = (
                AP_BWE_DIR / "checkpoints" / f"{lr_khz}kto48k" / f"g_{lr_khz}kto48k"
            )

    def _lazy_load(self, in_sr: int) -> None:
        if self._model is not None:
            return
        self._resolve_paths(in_sr)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"AP-BWE checkpoint not found at {self.checkpoint_path}. "
                f"Download `g_{in_sr // 1000}kto48k` from "
                "https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF "
                "(or set AP_BWE_CHECKPOINT to its path)."
            )

        # AP-BWE's code does `from utils import init_weights, get_padding` and
        # `from env import AttrDict` against the repo root. Add it to path and
        # remove after import so we don't shadow other modules.
        sys.path.insert(0, str(AP_BWE_DIR))
        try:
            from env import AttrDict  # type: ignore
            from models.model import APNet_BWE_Model  # type: ignore
        finally:
            sys.path.remove(str(AP_BWE_DIR))

        with open(self.config_path) as f:
            self._h = AttrDict(json.load(f))

        model = APNet_BWE_Model(self._h).to(self.device)
        model.train(False)  # inference mode
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        # Official ckpts store the generator under "generator"; some forks use bare state_dict.
        state = ckpt.get("generator", ckpt)
        model.load_state_dict(state)
        self._model = model
        logging.info(
            f"AP-BWE loaded: {self._h.lr_sampling_rate} -> {self._h.hr_sampling_rate} Hz "
            f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)"
        )

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, in_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """Upsample ``waveform`` (shape (C, T) or (T,)) from ``in_sr`` to 48 kHz.

        ``in_sr`` selects the matching AP-BWE variant (16k/24k/etc. -> 48k)
        on first call. Subsequent calls must keep the same input rate.
        """
        self._lazy_load(in_sr)
        h = self._h
        if in_sr != h.lr_sampling_rate:
            raise ValueError(
                f"AP-BWE was loaded for {h.lr_sampling_rate} Hz input; "
                f"got {in_sr}. Construct a new APBWEUpsampler for a different rate."
            )

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        x = waveform.to(self.device, dtype=torch.float32)

        # Sinc-upsample 24 kHz -> 48 kHz so STFT bins match the wide-band grid.
        import torchaudio.functional as taF
        x_up = taF.resample(x, orig_freq=h.lr_sampling_rate, new_freq=h.hr_sampling_rate)

        # AP-BWE's amp_pha_stft / istft live in datasets/dataset.py.
        sys.path.insert(0, str(AP_BWE_DIR))
        try:
            from datasets.dataset import amp_pha_stft, amp_pha_istft  # type: ignore
        finally:
            sys.path.remove(str(AP_BWE_DIR))

        # Per-channel: model is mono-trained; STFT/iSTFT roundtrip preserves length.
        out_channels = []
        for c in range(x_up.shape[0]):
            amp_nb, pha_nb, _ = amp_pha_stft(x_up[c : c + 1], h.n_fft, h.hop_size, h.win_size)
            amp_wb, pha_wb, _ = self._model(amp_nb, pha_nb)
            y = amp_pha_istft(amp_wb, pha_wb, h.n_fft, h.hop_size, h.win_size)
            out_channels.append(y.squeeze(0))
        y = torch.stack(out_channels, dim=0)  # (C, T)
        return y.clamp(-1.0, 1.0), int(h.hr_sampling_rate)


# ---------------------------------------------------------------------------
# AudioSR
# ---------------------------------------------------------------------------


class AudioSRUpsampler:
    """Versatile Audio Super-Resolution (haoheliu/versatile_audio_super_resolution).

    Latent diffusion. 200-step DDIM by default - slow but high quality.
    ``model_name``: "basic" (general audio) or "speech" (recommended for TTS).
    """

    def __init__(
        self,
        model_name: str = "speech",
        device: str | torch.device = "cuda",
        ddim_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self._latent_diffusion = None
        self._sr = 48000  # AudioSR is hard-wired to 48 kHz output

    def _lazy_load(self) -> None:
        if self._latent_diffusion is not None:
            return
        if str(AUDIOSR_DIR) not in sys.path:
            sys.path.insert(0, str(AUDIOSR_DIR))
        from audiosr import build_model  # type: ignore

        self._latent_diffusion = build_model(model_name=self.model_name, device=self.device)
        logging.info(f"AudioSR loaded: {self.model_name} (ddim={self.ddim_steps})")

    # AudioSR's super_resolution_long_audio uses 15 s chunks with 2 s overlap and
    # crashes when the trailing chunk is shorter than overlap_samples. We pad
    # the input so the last chunk is always >= overlap_samples + a safety bin.
    CHUNK_S = 15.0
    OVERLAP_S = 2.0

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, in_sr: int = 24000) -> Tuple[torch.Tensor, int]:
        self._lazy_load()
        from audiosr.pipeline import super_resolution_long_audio  # type: ignore

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        import tempfile
        import soundfile as sf

        # super_resolution_long_audio crashes when the trailing chunk is shorter
        # than overlap_samples. Pad only when needed: target the smallest length
        # of the form (step_s * N + CHUNK_S) that >= dur_s, so the final chunk
        # is always full-size. Adds at most (step_s + CHUNK_S) of dead audio,
        # ~28 s of extra work in the worst case (~2 DDIM passes).
        dur_s = waveform.shape[-1] / in_sr
        step_s = self.CHUNK_S - self.OVERLAP_S
        # smallest n s.t. step_s * n + CHUNK_S >= dur_s
        n_extra_steps = max(0, int(((dur_s - self.CHUNK_S) // step_s) + 1))
        pad_to_s = step_s * n_extra_steps + self.CHUNK_S
        pad_samples_in = max(0, int(round(pad_to_s * in_sr)) - waveform.shape[-1])
        keep_samples_out = int(round(dur_s * self._sr))

        out_channels = []
        with tempfile.TemporaryDirectory() as tmp:
            for c in range(waveform.shape[0]):
                ch = waveform[c].cpu().float()
                if pad_samples_in > 0:
                    ch = torch.cat([ch, torch.zeros(pad_samples_in, dtype=ch.dtype)], dim=0)
                wav_path = os.path.join(tmp, f"ch{c}.wav")
                sf.write(wav_path, ch.numpy(), in_sr)
                y = super_resolution_long_audio(
                    self._latent_diffusion,
                    wav_path,
                    seed=self.seed,
                    ddim_steps=self.ddim_steps,
                    guidance_scale=self.guidance_scale,
                    chunk_duration_s=self.CHUNK_S,
                    overlap_duration_s=self.OVERLAP_S,
                )
                # Returns torch.Tensor (1, 1, T) at 48 kHz; trim back to original dur.
                y = y.squeeze().detach().cpu().float()[:keep_samples_out]
                out_channels.append(y)

        min_len = min(c.shape[-1] for c in out_channels)
        y = torch.stack([c[:min_len] for c in out_channels], dim=0)
        return y.clamp(-1.0, 1.0), self._sr


# ---------------------------------------------------------------------------
# RE-USE (NVIDIA / SEMamba)
# ---------------------------------------------------------------------------


class REUSEUpsampler:
    """Universal speech enhancement with optional bandwidth extension.

    nvidia/RE-USE is a 9.6 M-param bidirectional-Mamba model that operates on
    STFT amplitude+phase. With ``target_sr`` set it both denoises *and* extends
    the bandwidth to that rate via librosa kaiser-best resample + restoration.

    License: NSCLv1 (noncommercial). The base ``SEMamba`` class lives in the
    HF repo under ``models/generator_SEMamba_time_d4.py`` and pulls in the
    ``mamba_ssm`` / ``causal-conv1d`` CUDA kernels.
    """

    def __init__(
        self,
        target_sr: int = 48000,
        config_path: Optional[str] = None,
        chunk_size_s: float = 1.0,
        hop_portion: float = 0.5,
        device: str | torch.device = "cuda",
    ) -> None:
        # chunk_size_s: peak VRAM scales linearly with chunk length.
        #   5.0s -> 2.95 GB | 2.5s -> 1.52 GB | 1.0s -> 0.67 GB (default).
        # 1.0s is chosen as default so RE-USE fits comfortably on top of the
        # rest of the DramaBox pipeline on any 24 GB-class GPU. ~10% slower
        # than 2.5s due to more per-chunk launch overhead, but still ~3x
        # real-time on a B300 (5-6s wall time per 14s clip after warmup).
        self.device = torch.device(device)
        self.target_sr = int(target_sr)
        self.chunk_size_s = float(chunk_size_s)
        self.hop_portion = float(hop_portion)
        # Default config matches inference.sh / inference.py defaults in the repo.
        # Config path is resolved lazily on first use (alongside the code tree)
        # so importing this module never triggers a download.
        self._config_path_override = Path(config_path) if config_path else None
        self.config_path: Optional[Path] = None
        self._model = None
        self._cfg = None
        self._stft_fns = None  # (mag_phase_stft, mag_phase_istft, compress_factor, pad_or_trim)

    @staticmethod
    def _ensure_mamba_ssm_importable() -> None:
        """Import ``mamba_ssm`` cleanly, with a kernel-free fallback if needed.

        Normal path (kernels present): just import — fast path uses
        ``selective_scan_cuda`` natively. ~real-time inference on B300.

        Fallback (kernels missing): the official package does an unconditional
        ``import selective_scan_cuda`` at module load. We stub it into
        ``sys.modules`` before importing, then redirect ``selective_scan_fn``
        to the pure-PyTorch ``selective_scan_ref`` so the model still runs (~5-
        10x slower; only useful if compiling the kernel for sm_103 isn't viable).
        """
        try:
            import selective_scan_cuda  # noqa: F401
            import mamba_ssm  # noqa: F401
            return  # Fast path: kernel present.
        except ImportError:
            pass

        import types
        if "selective_scan_cuda" not in sys.modules:
            stub = types.ModuleType("selective_scan_cuda")
            def _missing(*a, **kw):  # pragma: no cover - safety net only
                raise NotImplementedError(
                    "selective_scan_cuda kernel missing; the call should have "
                    "been routed to selective_scan_ref via the runtime patch."
                )
            stub.fwd = _missing
            stub.bwd = _missing
            sys.modules["selective_scan_cuda"] = stub

        from mamba_ssm.ops import selective_scan_interface as ssi
        from mamba_ssm.modules import mamba_simple
        if getattr(ssi, "_dramabox_kernel_free_patch_applied", False):
            return
        ssi.selective_scan_fn = ssi.selective_scan_ref
        ssi.mamba_inner_fn = ssi.mamba_inner_ref
        # mamba_simple imported these names by reference at module load -
        # rebind there too, otherwise Mamba.forward keeps the original handles.
        mamba_simple.selective_scan_fn = ssi.selective_scan_ref
        mamba_simple.mamba_inner_fn = ssi.mamba_inner_ref
        ssi._dramabox_kernel_free_patch_applied = True
        logging.info(
            "mamba_ssm kernel missing - using kernel-free fallback "
            "(selective_scan_fn -> selective_scan_ref). Expect ~5-10x slowdown."
        )

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        # Prefer real CUDA kernels; gracefully fall back to pure-PyTorch impl.
        self._ensure_mamba_ssm_importable()

        # The RE-USE module imports `from models...` and `from utils...` —
        # both relative to the repo root. Add to path during load.
        reuse_dir = _resolve_reuse_dir()
        if str(reuse_dir) not in sys.path:
            sys.path.insert(0, str(reuse_dir))

        if self.config_path is None:
            self.config_path = self._config_path_override or (
                reuse_dir / "recipes" /
                "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
            )

        from models.generator_SEMamba_time_d4 import SEMamba  # type: ignore
        from models.stfts import mag_phase_stft, mag_phase_istft  # type: ignore
        from utils.util import load_config, pad_or_trim_to_match  # type: ignore

        self._cfg = load_config(str(self.config_path))
        compress_factor = self._cfg["model_cfg"]["compress_factor"]
        self._stft_fns = (mag_phase_stft, mag_phase_istft, compress_factor, pad_or_trim_to_match)

        # SEMamba is a PyTorchModelHubMixin; from_pretrained pulls weights from HF.
        model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=self._cfg).to(self.device)
        model.train(False)
        self._model = model
        n_params = sum(p.numel() for p in model.parameters())
        logging.info(f"RE-USE loaded: SEMamba ({n_params / 1e6:.1f}M params) -> {self.target_sr} Hz")

    @staticmethod
    def _make_even(v: float) -> int:
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, in_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """Chunked overlap-add upsample (ports nvidia/RE-USE inference_chunk.py).

        Peak VRAM is bounded by ``chunk_size_s * target_sr`` rather than the
        whole clip, so a 60 s clip costs the same as a 5 s one. Crossfade is
        a Hann-window normalized overlap-add with default 50% hop.
        """
        import math
        self._lazy_load()
        import librosa
        mag_phase_stft, mag_phase_istft, compress_factor, pad_or_trim_to_match = self._stft_fns

        # STFT params are scaled relative to the config's training rate (8000).
        base_n_fft = self._cfg["stft_cfg"]["n_fft"]
        base_hop = self._cfg["stft_cfg"]["hop_size"]
        base_win = self._cfg["stft_cfg"]["win_size"]
        base_sr = self._cfg["stft_cfg"]["sampling_rate"]

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 1. Resample to target rate first (the --BWE flow).
        if self.target_sr != in_sr:
            wav_np = waveform.cpu().float().numpy()
            wav_np = librosa.resample(
                wav_np, orig_sr=in_sr, target_sr=self.target_sr, res_type="kaiser_best"
            )
            wav = torch.from_numpy(wav_np).to(self.device, dtype=torch.float32)
        else:
            wav = waveform.to(self.device, dtype=torch.float32)

        op_sr = self.target_sr
        n_fft = self._make_even(base_n_fft * op_sr // base_sr)
        hop = self._make_even(base_hop * op_sr // base_sr)
        win = self._make_even(base_win * op_sr // base_sr)

        # 2. Chunked OLA with Hann analysis window. Mirrors inference_chunk.py.
        chunk_size = int(self.chunk_size_s * op_sr)
        hop_length = int(self.hop_portion * chunk_size)
        window = torch.hann_window(chunk_size, device=self.device)

        n_ch, total = wav.shape
        enhanced = torch.zeros_like(wav)
        window_sum = torch.zeros_like(wav)
        n_chunks = max(1, math.ceil((total - chunk_size) / hop_length) + 1) if total > chunk_size else 1

        for c in range(n_ch):
            ch_in = wav[c : c + 1]                              # (1, T)
            for i in range(n_chunks):
                start = i * hop_length
                end = min(start + chunk_size, total)
                chunk = ch_in[:, start:end]
                if chunk.shape[-1] < 2:                          # skip degenerate tail
                    continue
                noisy_mag, noisy_pha, _ = mag_phase_stft(
                    chunk, n_fft=n_fft, hop_size=hop, win_size=win,
                    compress_factor=compress_factor, center=True, addeps=False,
                )
                amp_g, pha_g, _ = self._model(noisy_mag, noisy_pha)
                # "Sweep artifact" filter — match the official inference.
                mag = torch.expm1(torch.relu(amp_g))
                zero_portion = (mag == 0).sum(dim=1) / mag.shape[1]
                amp_g[:, :, (zero_portion > 0.5)[0]] = 0

                audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress_factor)
                audio_g = pad_or_trim_to_match(chunk.detach(), audio_g, pad_value=1e-8)

                w_slice = window[: audio_g.shape[-1]]
                enhanced[c : c + 1, start : start + audio_g.shape[-1]] += audio_g * w_slice
                window_sum[c : c + 1, start : start + audio_g.shape[-1]] += w_slice

        # 3. Normalize where windows overlap. Avoid divide-by-zero at clip tails.
        mask = window_sum > 1e-8
        enhanced[mask] = enhanced[mask] / window_sum[mask]
        return enhanced.clamp(-1.0, 1.0).cpu().float(), op_sr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_upsampler(name: str, **kwargs):
    """Return an upsampler callable. ``name`` in {'ap_bwe', 'audiosr', 'reuse'}."""
    name = name.lower().replace("-", "_")
    if name == "ap_bwe":
        return APBWEUpsampler(**kwargs)
    if name == "audiosr":
        return AudioSRUpsampler(**kwargs)
    if name == "reuse":
        return REUSEUpsampler(**kwargs)
    raise ValueError(f"Unknown upsampler: {name!r}. Choose 'ap_bwe', 'audiosr', or 'reuse'.")
