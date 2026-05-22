"""Prompt chunking for long-form DramaBox generation.

The base LTX-2.3 audio DiT was trained on clips <= ~20 s. The silence-prior
patch in ``inference_server.py`` keeps generations sane up to ~45 s, but the
prior re-emerges past that boundary. For arbitrary-length prompts we split the
text into < 45 s chunks, generate each conditioned on the same voice reference,
and crossfade them back together.

Chunking is quote-aware (sentence terminators inside ``"..."`` don't count)
and preserves the speaker-description prefix on every chunk so the model keeps
the same persona / delivery style across joins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# Matches the leading speaker description, ending at the first comma that's
# directly followed by a space + opening quote. Anything before that is treated
# as persona/style metadata and re-attached to every chunk.
#   "A shadowy villain speaks with cold menace, \"You have entered...\""
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_PREFIX_RE = re.compile(r'^([^"\']{3,}?)(,\s*)(?=["\'])', re.DOTALL)


@dataclass
class PromptChunk:
    text: str
    est_duration_s: float


def extract_speaker_prefix(prompt: str) -> tuple[Optional[str], str]:
    """Return ``(prefix, body)`` where ``prefix`` is the speaker description.

    If the prompt has the canonical ``"<persona>, "<dialogue>"..."`` form, the
    persona (without the trailing comma) is returned as the prefix and the rest
    of the prompt as the body. Otherwise ``(None, prompt)`` — no prefix to
    propagate, the whole prompt is treated as a single body.
    """
    m = _PREFIX_RE.match(prompt)
    if not m:
        return None, prompt
    return m.group(1).strip(), prompt[m.end():]


def split_sentences_outside_quotes(text: str) -> List[str]:
    """Split ``text`` into sentences, ignoring terminators inside quotes.

    A "sentence" here is a span ending in ``.``/``!``/``?`` (optionally followed
    by a closing quote) at the top level — i.e. not inside an open ``"..."`` or
    ``'...'`` pair. Empty / whitespace-only fragments are dropped.

    Examples:
        >>> split_sentences_outside_quotes('He says, "Hi, how are you?" Then leaves.')
        ['He says, "Hi, how are you?"', 'Then leaves.']
    """
    sentences: List[str] = []
    buf: List[str] = []
    in_double = False
    in_single = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)

        if ch == '"' and not in_single:
            was_inside = in_double
            in_double = not in_double
            # Treat the *closing* quote as a sentence boundary if the last
            # meaningful char inside it was a terminator: ``...how are you?"``.
            if was_inside and len(buf) >= 2 and buf[-2] in ".!?":
                # Boundary requires whitespace / end-of-string after.
                if i + 1 >= n or text[i + 1].isspace():
                    sentence = "".join(buf).strip()
                    if sentence:
                        sentences.append(sentence)
                    buf = []
                    i += 1
                    continue

        elif ch == "'" and not in_double:
            # Apostrophes inside a word (don't, it's) are not quote toggles.
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < n else " "
            if not (prev.isalpha() and nxt.isalpha()):
                in_single = not in_single

        elif ch in ".!?" and not in_double and not in_single:
            # Greedily eat trailing closing quotes / punctuation.
            j = i + 1
            while j < n and text[j] in '."\')]':
                buf.append(text[j])
                if text[j] == '"':
                    in_double = not in_double  # closing quote toggle
                j += 1
            if j >= n or text[j].isspace():
                sentence = "".join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                buf = []
                i = j
                continue
        i += 1

    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _assemble(prefix: Optional[str], sentences: List[str]) -> str:
    body = " ".join(s.strip() for s in sentences if s.strip())
    if not prefix:
        return body
    # Re-attach prefix in the canonical "persona, body" form. If the first
    # sentence already starts with a stage direction (no opening quote), drop
    # the comma + use a period so the syntax reads naturally.
    if body.lstrip().startswith(("'", '"')):
        return f"{prefix}, {body}"
    return f"{prefix}. {body}"


def chunk_prompt_for_duration(
    prompt: str,
    max_duration_s: float = 45.0,
    target_duration_s: float = 37.0,
    duration_multiplier: float = 1.1,
) -> List[PromptChunk]:
    """Split ``prompt`` into <= ``max_duration_s`` chunks.

    Args:
        prompt: Full scene prompt (DramaBox format or plain text).
        max_duration_s: Hard cap per chunk; we never emit a chunk whose
            estimator output (after ``duration_multiplier``) exceeds this.
        target_duration_s: Soft cap; we close the current chunk when adding
            the next sentence would push it past this. Leaving 5-10 s of
            headroom below ``max_duration_s`` keeps us safe against the
            estimator under-shooting by ~10-15% on action-heavy prompts.
        duration_multiplier: Same breathing-room multiplier the inference
            server applies in ``estimate_duration``; matches the per-chunk
            target the model is actually asked to generate.

    Returns:
        List of :class:`PromptChunk`. Single-chunk prompts return a 1-element
        list with the original prompt unchanged.
    """
    from duration_estimator import estimate_speech_duration

    def _est(t: str) -> float:
        return estimate_speech_duration(t) * duration_multiplier

    total = _est(prompt)
    if total <= max_duration_s:
        return [PromptChunk(text=prompt, est_duration_s=total)]

    prefix, body = extract_speaker_prefix(prompt)
    sentences = split_sentences_outside_quotes(body)
    if not sentences:
        # Degenerate: no sentence boundaries. Fall back to whitespace-token
        # chunking so we still produce SOMETHING under the cap.
        sentences = body.split()

    chunks: List[PromptChunk] = []
    current: List[str] = []
    current_dur = 0.0

    for sent in sentences:
        candidate = _assemble(prefix, current + [sent])
        cand_dur = _est(candidate)

        if current and cand_dur > target_duration_s:
            # Close the current chunk before adding this sentence.
            assembled = _assemble(prefix, current)
            chunks.append(PromptChunk(text=assembled, est_duration_s=_est(assembled)))
            current = [sent]
            current_dur = _est(_assemble(prefix, current))
        else:
            current.append(sent)
            current_dur = cand_dur

        # Pathological case: a single sentence whose estimator output is
        # already past max_duration_s. Emit it on its own and let downstream
        # generate() truncate the request at the model's hard limit; the user
        # gets a degraded but non-crashing result instead of an exception.
        if len(current) == 1 and current_dur > max_duration_s:
            solo = _assemble(prefix, current)
            chunks.append(PromptChunk(text=solo, est_duration_s=current_dur))
            current = []
            current_dur = 0.0

    if current:
        assembled = _assemble(prefix, current)
        chunks.append(PromptChunk(text=assembled, est_duration_s=_est(assembled)))

    return chunks
