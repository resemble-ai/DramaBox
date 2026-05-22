"""Pure-Python speech-duration estimator for DramaBox prompts.

Originally lived in ``inference.py`` but pulled out so chunkers / tooling /
unit tests can import it without dragging torch + the LTX pipeline through
sys.path. ``inference.py`` and ``inference_server.py`` continue to import
``estimate_speech_duration`` from here.
"""
from __future__ import annotations

import re


_LAUGH_VERBS = {
    # base seconds per occurrence; gets scaled by the modifier found nearby.
    # Verb regex covers inflections: laugh/laughs/laughed/laughing.
    r"\blaugh(?:s|ed|ing)?\b": 1.5,
    r"\bcackl(?:e|es|ed|ing)\b": 1.5,
    r"\bchuckl(?:e|es|ed|ing)\b": 1.0,
    r"\bgiggl(?:e|es|ed|ing)\b": 1.0,
    r"\bsnicker(?:s|ed|ing)?\b": 0.8,
    r"\bcru?el laugh\b": 1.5,
}


def _contextual_laugh_duration(text: str) -> float:
    """Context-aware laugh budget.

    For each laugh verb in the prompt, look at the adjective/adverb that
    modifies it and scale the base duration:
      - short modifiers  (briefly, softly, once)     -> 0.4x base
      - long modifiers   (maniacally, heartily, ...) -> 1.2x base
      - default (no mod / neutral)                   -> 1.0x base
    Also reward phonetic repetition inside quotes -- 'Hahahahahaha' buys more
    time than 'Haha' -- at ~0.2s per extra repeated syllable.
    """
    short_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:briefly|shortly|once|quickly)",
        re.IGNORECASE)
    long_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:maniacally|heartily|uproariously|uncontrollably|"
        r"hysterically|darkly|wickedly|evilly|loudly|long)"
        r"|^\s*between phrases", re.IGNORECASE)

    total = 0.0
    for pat, base_dur in _LAUGH_VERBS.items():
        for m in re.finditer(pat, text, re.IGNORECASE):
            ctx = text[m.end(): m.end() + 40]
            if short_mod.match(ctx):
                total += base_dur * 0.4
            elif long_mod.match(ctx):
                total += base_dur * 1.2
            else:
                total += base_dur

    # Phonetic laugh repetition inside quotes.
    for q in re.findall(r'"([^"]+)"', text) + re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text):
        for run in re.findall(r"(?:h[ae]){3,}|(?:h[ae][ \-]?){3,}", q, re.IGNORECASE):
            syls = len(re.findall(r"h[ae]", run, re.IGNORECASE))
            total += 0.2 * max(syls - 2, 0)
    return total


def _estimate_nonverbal_duration(text: str) -> float:
    """Estimate extra duration for non-verbal sounds and actions in the prompt.

    Laugh-verb handling lives in ``_contextual_laugh_duration`` so cackle /
    chuckle / laugh budgets scale with the adjective ("maniacally" vs
    "briefly") and with the repetition length of 'Ha'/'He' tokens inside
    quotes.
    """
    PATTERNS = {
        r'\bsighs?\b': 0.8, r'\bshaky breath\b': 1.0, r'\bbreathing deeply\b': 1.0,
        r'\bgasps?\b': 0.5, r'\bburps?\b': 0.5, r'\byawns?\b': 1.0,
        r'\bpants?\b': 0.8, r'\bwheezes?\b': 0.8, r'\bcoughs?\b': 0.8,
        r'\bsniffles?\b': 0.5, r'\bsnorts?\b': 0.3, r'\bgroans?\b': 0.8,
        r'\blong pause\b': 1.0, r'\bpauses? briefly\b': 0.3,
        r'\bpauses?\b': 0.5, r'\bsilence\b': 1.0,
        r'\blets? the .{1,20} hang\b': 1.0, r'\blets? .{1,20} sink in\b': 1.0,
        r'\bslams?\b': 0.5, r'\bclaps?\b': 0.3,
        r'\bdraws? (?:his|her|a) sword\b': 0.5,
        r'\btakes? a (?:drag|swig|sip|drink)\b': 0.5,
        r'\bwhistles?\b': 1.0, r'\bhums?\b': 0.8,
        r'\bmutters?\b': 1.5, r'\bmumbles?\b': 1.0, r'\bwhispers?\b': 0.0,
        r'\bclears? (?:his|her) throat\b': 0.5, r'\bgulps?\b': 0.5,
        r'\bswallows?\b': 0.5,
        r'\bvoice (?:breaks?|cracks?|trembles?|drops?|rises?)\b': 0.5,
        r'\bsteadies? (?:him|her)self\b': 1.0,
        r'\bcatches? (?:his|her) breath\b': 1.0,
        r'\bcomposes? (?:him|her)self\b': 0.8,
        r'\bdemeanor shifts?\b': 0.5, r'\bsettles? in\b': 0.5,
        r'\bleans? in\b': 0.3, r'\bwipes? (?:his|her) eyes\b': 0.5,
    }
    extra = 0.0
    for pattern, dur in PATTERNS.items():
        extra += dur * len(re.findall(pattern, text, re.IGNORECASE))
    extra += _contextual_laugh_duration(text)
    return extra


def estimate_speech_duration(text: str, speed: float = 1.0) -> float:
    """Estimate speech duration from spoken content + non-verbal actions.

    Extracts spoken text by priority:
    1. Quoted text ('...' or "...") -- official prompt guide format
    2. Text after colon -- simple "Speaker: dialogue" format
    3. Full text -- fallback

    Also scans the full prompt for non-verbal cues (laughs, pauses, sighs,
    gasps, etc.) and adds estimated duration for each.
    """
    quotes = re.findall(r'"([^"]+)"', text)
    if not quotes:
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text)
        quotes = [q for q in quotes if len(q.split()) > 3]
    if quotes:
        spoken = " ".join(quotes)
    elif ":" in text:
        spoken = text.split(":", 1)[1].strip()
    else:
        spoken = text

    CHARS_PER_SEC = 14.0
    text_len = len(spoken)

    if text_len < 40:
        chars_per_sec = CHARS_PER_SEC * 0.6
    elif text_len < 80:
        chars_per_sec = CHARS_PER_SEC * 0.8
    else:
        chars_per_sec = CHARS_PER_SEC

    chars_per_sec *= speed
    duration = text_len / chars_per_sec

    sentence_count = spoken.count(".") + spoken.count("!") + spoken.count("?")
    duration += sentence_count * 0.3

    duration += _estimate_nonverbal_duration(text)

    return max(3.0, round(duration + 2.0, 1))
