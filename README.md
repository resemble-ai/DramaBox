<p align="center">
  <a href="https://www.resemble.ai/learn/models/dramabox">
    <img src="assets/Dramabox.png" alt="DramaBox" width="720"/>
  </a>
</p>

# DramaBox — Expressive TTS with Voice Cloning

[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Dramabox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

> **Built on [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks.**
> DramaBox is **Resemble AI's** expressive TTS, trained on top of the LTX-2.3 audio branch under the LTX-2 Community License. Huge thanks to the Lightricks team for open-sourcing the base.

*Made with ♥️ by* <a href="https://www.resemble.ai/learn/models/dramabox" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

Prompt-driven TTS with voice cloning. The prompt itself controls speaker identity, emotion, delivery style, laughs, sighs, pauses and transitions; an optional 10-second voice reference clones the target timbre. DramaBox is an IC-LoRA fine-tune of the **LTX-2.3 3.3B audio-only** model.

| | |
|---|---|
| 🤗 **Model** | [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox) |
| 🎭 **Demo Space** | [`ResembleAI/Dramabox`](https://huggingface.co/spaces/ResembleAI/Dramabox) (ZeroGPU) |
| 🏗️ **Base model** | [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) |
| 📜 **License** | LTX-2 Community License — see [`LICENSE`](LICENSE) |

## Models

Auto-downloaded from the HF model repo on first run.

| File | Size | Description |
|---|---|---|
| `dramabox-dit-v1.safetensors` | 6.6 GB | DiT transformer (LoRA already merged into base) |
| `dramabox-audio-components.safetensors` | 1.9 GB | Audio embeddings connector + audio text projection + audio VAE + vocoder |
| [`unsloth/gemma-3-12b-it-bnb-4bit`](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) | ~8 GB | Text encoder |

**VRAM**: ~24 GB peak · **Speed**: ~2.5 s / generation (warm server, H100)

## Quick Start

### Warm server (recommended)

```python
from src.inference_server import TTSServer

server = TTSServer(device="cuda")

server.generate_to_file(
    prompt='A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"',
    output="output.wav",
    voice_ref="reference.wav",   # optional, 10+ seconds
)
```

### CLI

```bash
python src/inference.py \
  --voice-sample reference.wav \
  --prompt 'A woman speaks warmly, "Hello, how are you today?"' \
  --output output.wav \
  --cfg-scale 2.5 --stg-scale 1.5
```

### Gradio app

```bash
CUDA_VISIBLE_DEVICES=4 python app.py
```

## Inference Settings

| Parameter | Default | Notes |
|---|---|---|
| `cfg-scale` | 2.5 | Lower = more natural, higher = more text-faithful |
| `stg-scale` | 1.5 | Skip-token guidance |
| `rescale` | 0 | No rescaling |
| `modality` | 1 | No modality guidance |
| `duration-multiplier` | 1.1 | 10% breathing room on auto-estimated length |
| `steps` | 30 | Euler flow matching |

## Prompt Writing Guide

**Structure:** `<speaker description>, "<dialogue>" <action direction> "<more dialogue>"`

**Inside quotes** (model produces actual sounds):
- Laughs: `"Hahaha"` `"Hehehe"` (always one word, never separated)
- Sounds: `"Mmmmm"` `"Ugh"` `"Argh"` `"Ahhh"` `"Hmm"`

**Outside quotes** (stage directions):
- `She sighs deeply.` · `He gulps nervously.` · `A long pause.`
- `Her voice cracks.` · `He clears his throat.` · `She scoffs.`

**Avoid inside quotes** (model speaks them literally): `Ahem`, `Pfft`, `Sigh`, `Gasp`, `Cough`.

**Tips**
- Match gender/age in the speaker description to the voice reference
- Break long dialogue into segments with action directions in between
- End the prompt at the last closing quote mark (no trailing description)

## Watermarking

Every audio output from `inference.py` and `inference_server.TTSServer.generate_to_file` is automatically watermarked with [Resemble Perth](https://github.com/resemble-ai/Perth) — an imperceptible neural watermark that survives MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

```python
import perth, librosa
wav, sr = librosa.load("output.wav", sr=None, mono=True)
detector = perth.PerthImplicitWatermarker()
print(detector.get_watermark(wav, sample_rate=sr))   # confidence ≈ 1.0
```

Pass `--no-watermark` to `inference.py` (or `watermark=False` to `generate_to_file`) to disable for debugging.

## Voice reference denoising (RE-USE)

The voice reference is denoised with [`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE) before VAE conditioning. On (defaults to `True`):

```python
server.generate_to_file(
    prompt='A woman speaks warmly, "Hello."',
    output="out.wav",
    voice_ref="ref.wav",
    denoise_ref=True,
)
```

Setup is automatic — code (`.py` / `.yaml`, ~150 KB) is snapshot-downloaded into `~/.cache/dramabox/` on first call; weights (`~38 MB`) come from `SEMamba.from_pretrained` through the standard HF cache. Pass `$REUSE_DIR` or vendor at `third_party/RE-USE/` to skip the download.

Optional kernels: `mamba-ssm` + `causal-conv1d`. If their wheels fail to build, the wrapper falls back to a pure-PyTorch path (5-10× slower, same output). RE-USE is [NSCLv1](https://github.com/NVlabs/HMAR/blob/main/LICENSE) (non-commercial) — set `denoise_ref=False` to skip.

## Long-form generation (text chunking)

The base LTX-2.3 audio DiT was trained on clips ≤ ~20 s. Generations up to ~45 s remain usable thanks to the silence-prior patch in `inference_server.py`, but quality drifts past that. For arbitrarily long prompts, `TTSServer` now chunks automatically:

```python
server.generate_to_file(
    prompt=very_long_scene,        # 2 minutes worth of dialogue is fine
    output="long.wav",
    voice_ref="ref.wav",           # reused across every chunk → speaker stays coherent
)
```

The chunker (`src/text_chunker.py:chunk_prompt_for_duration`) splits at sentence / quote-group boundaries, preserves the speaker-description prefix on every chunk, and targets ~37 s per chunk (≤45 s hard cap). `TTSServer.generate_long` then concatenates the per-chunk waveforms with a 50 ms equal-power crossfade so joins are inaudible. Set `max_chunk_duration=` / `target_chunk_duration=` on `generate_to_file` to tune.

## Training a LoRA on top of DramaBox

You can fine-tune your own LoRA using DramaBox itself as the base — no need to start from raw LTX-2.3. Useful for adding a specific speaker, language flavour, or style on top of the existing expressive prior.

### 1. Prepare your index file

The preprocessor accepts four formats. The `text` field is the **target transcript**; if you want to attach a scene-style prompt (the part the model conditions on at inference time), prepend it to the transcript in the same format the model was trained on:

> `A woman speaks warmly, "<your transcript here>"`

Both forms are supported — with or without the prompt wrapper. Without the wrapper the model treats the entry as plain text-to-speech.

**Format A — `manifest` (JSONL)** — recommended for new datasets:

```jsonl
{"audio_filepath": "wavs/spk01_001.wav", "text": "A woman speaks warmly, \"Hello, how are you today?\""}
{"audio_filepath": "wavs/spk01_002.wav", "text": "Hello, how are you today?"}
{"audio_filepath": "wavs/spk02_001.flac", "text": "An exhausted father sighs, \"Sweetie, daddy is asking very nicely.\"", "duration": 4.7}
```

Fields: `audio_filepath` (or `audio_path`) is required, `text` (or `transcript`) is required, `duration` is optional.

**Format B — `tsv`** — simplest, one line per sample:

```
wavs/spk01_001.wav	A woman speaks warmly, "Hello, how are you today?"
wavs/spk01_002.wav	Hello, how are you today?
```

**Format C — `gemini_synthetic`** — `~`-separated, used for prompted synthetic data:

```
id~speaker~lang~sr~samples~dur~phonemes~text
spk01_001~spk01~en~24000~93000~3.875~_~A woman speaks warmly, "Hello, how are you today?"
```

**Format D — `libriheavy`** — `~`-separated, for unprompted text-only data:

```
id~speaker~lang~samples~dur_ms~phonemes~text
spk01_001~spk01~en~93000~3875~_~Hello, how are you today?
```

### 2. Preprocess

```bash
python src/preprocess.py \
  --dataset-type manifest \
  --index your_data.jsonl \
  --audio-dir /path/to/wavs \
  --output-dir /path/to/preprocessed/ \
  --checkpoint /path/to/dramabox-audio-components.safetensors \
  --gemma-root /path/to/gemma-3-12b-it-bnb-4bit/ \
  --max-duration 20.0 --min-duration 2.0
```

Output layout (training-ready `.pt` files):

```
preprocessed/
├── audio_latents/sample_*.pt     # Audio VAE-encoded latents
├── conditions/sample_*.pt        # Gemma text embeddings
└── latents/sample_*.pt           # Dummy video latents (placeholder)
```

### 3. Train

Copy `configs/training_args.example.yaml`, point `data_dir` / `speaker_index` at your preprocessed output, set `checkpoint` + `full_checkpoint` to the DramaBox files, then launch with HuggingFace `accelerate`. Any flag passed on the CLI overrides the YAML.

```bash
accelerate launch src/train.py \
  --config configs/training_args.example.yaml
```

The trainer attaches a fresh LoRA to the audio branch on top of the DramaBox checkpoint. LoRA targets: `audio_attn1.{to_q,to_k,to_v,to_out.0}` + `audio_ff.{net.0.proj,net.2}` × 48 transformer blocks (288 LoRA pairs total). Default rank 128 / alpha 128 / dropout 0.1, cosine LR schedule from 1e-4 with 500-step warmup over 10k steps.

To monitor training, set `val_config: configs/val_config.example.yaml` in your training YAML — `src/validate.py` is then spawned at every save step to generate one wav per speaker entry, so you can A/B listen during the run.

### Inference with your trained LoRA

```bash
python src/inference.py \
  --lora /path/to/your/lora_step_5000.safetensors \
  --voice-sample reference.wav \
  --prompt 'A woman speaks warmly, "..."' \
  --output output.wav
```

Always load the LoRA at inference rather than pre-merging it — pre-merged checkpoints have produced degraded output in our runs.

## Language

English.

## License & acknowledgement

DramaBox is a Resemble AI fine-tune of [LTX-2](https://github.com/Lightricks/LTX-2). Distributed under the LTX-2 Community License Agreement — see [`LICENSE`](LICENSE). Thanks again to Lightricks for releasing the base model.
