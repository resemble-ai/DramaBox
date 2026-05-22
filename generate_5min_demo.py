#!/usr/bin/env python3
"""Smoke-test the updated DramaBox pipeline by generating ~5 min of audio.

Exercises the full long-form path:
  - duration_estimator + text_chunker plan ~14 chunks
  - RE-USE pre-denoises the voice reference once, cached across chunks
  - LTX BigVGAN BWE handles 24 -> 48 kHz on the output
  - generate_long stitches chunks with a 50 ms equal-power crossfade
"""
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference_server import TTSServer  # noqa: E402
from model_downloader import get_all_paths  # noqa: E402

PROMPT_PATH = ROOT / "prompts" / "five_minute_demo.txt"
VOICE_REF = ROOT / "assets" / "voices" / "male_old_movie.wav"
OUTPUT = ROOT / "output" / "five_minute_demo.wav"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

logging.info("Fetching model paths (cached after first run)...")
PATHS = get_all_paths()

logging.info("Loading warm TTSServer...")
t0 = time.time()
tts = TTSServer(
    checkpoint=PATHS["transformer"],
    full_checkpoint=PATHS["audio_components"],
    gemma_root=PATHS["gemma_root"],
    device="cuda",
    dtype="bf16",
    compile_model=False,                 # avoid first-call recompile cost
    bnb_4bit=True,
)
logging.info(f"Warm load: {time.time()-t0:.1f}s")

prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
logging.info(f"Prompt: {len(prompt)} chars")

t0 = time.time()
tts.generate_to_file(
    prompt=prompt,
    output=str(OUTPUT),
    voice_ref=str(VOICE_REF),
    cfg_scale=2.5,
    stg_scale=1.5,
    duration_multiplier=1.1,
    seed=42,
    ref_duration=10.0,
    denoise_ref=True,                    # RE-USE on the reference
    max_chunk_duration=45.0,
    target_chunk_duration=37.0,
    crossfade_ms=50.0,
)
wall = time.time() - t0

import soundfile as sf
info = sf.info(str(OUTPUT))
logging.info(f"DONE in {wall:.1f}s -> {OUTPUT}")
logging.info(f"Output: {info.duration:.1f}s @ {info.samplerate} Hz, {info.channels} ch, "
             f"{os.path.getsize(OUTPUT)/1e6:.1f} MB")
print(f"\nOUTPUT_PATH={OUTPUT}")
