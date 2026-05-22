"""RE-USE (nvidia/RE-USE) speech-enhancement wrapper.

Used by ``TTSServer._denoise_voice_ref`` to denoise the input voice reference
before VAE conditioning. Lazy-loads weights + code on first call so importing
this module is cheap.

    up = REUSEUpsampler(target_sr=48000, device="cuda")
    clean, sr = up(wav, in_sr=24000)        # wav: (C, T) or (T,) float
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


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
        # rest of the DramaBox pipeline on any 24 GB-class GPU.
        self.device = torch.device(device)
        self.target_sr = int(target_sr)
        self.chunk_size_s = float(chunk_size_s)
        self.hop_portion = float(hop_portion)
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
        ``selective_scan_cuda`` natively.

        Fallback (kernels missing): the official package does an unconditional
        ``import selective_scan_cuda`` at module load. We stub it into
        ``sys.modules`` before importing, then redirect ``selective_scan_fn``
        to the pure-PyTorch ``selective_scan_ref`` so the model still runs
        (~5-10x slower).
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
        """Chunked overlap-add denoise / BWE (ports nvidia/RE-USE inference_chunk.py).

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

        # 1. Resample to target rate first (skips if target_sr == in_sr).
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
