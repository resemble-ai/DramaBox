# DramaBox Prompting Guide

A practical reference for getting consistent, expressive output from DramaBox.
Compiled from empirical prompt-engineering work — every pattern below was
validated by listening to generated audio.

---

## Quick start

Minimal prompt structure:

```
<speaker description> <action verb>, "<spoken dialogue>"
```

Example:
```
A woman speaks tenderly, "It has been a long day, my love."
```

Key facts to remember:
1. **Quoted text is spoken literally** — laughs (`Hahaha`), hums (`Mmmmm`),
   and cheers (`Woooo`) work *inside* quotes. Plain words like `Sigh` or
   `Gasp` inside quotes are also spoken literally (avoid — use stage
   directions instead).
2. **Unquoted text is stage direction** — emotion/action verbs (`speaks
   tenderly`, `chuckles darkly`, `gasps with shock`) shape delivery but
   are NOT spoken.
3. **Multi-segment prompts are stronger than single-segment.** A calm
   setup span followed by an emotional contrast span fires emotions much
   more reliably than a single emotional verb on its own.
4. **End the prompt at the last closing quote.** Trailing description
   after the last quote gets ignored or read.

---

## Universal speaker rule

Stick to generic noun phrases (+ optional one-word adjective):

| Generic | With adjective |
|---------|----------------|
| `A man`, `A woman` | `A weary man`, `A weary woman` |
| `A young man`, `A young woman` | `A nervous young woman` |
| `An elderly man`, `An elderly woman` | `A grieving woman` |
| `A child`, `A boy`, `A girl` | `A scared child`, `A terrified woman`, `A frightened woman`, `A confident woman`, `A flirtatious woman` |

**Why this matters**: role-based speakers (`A drill sergeant`, `A spy`,
`A teacher`, `A detective`, `A nurse`, `A scholar`) tend to bleed the role
name into the spoken text. The model treats them as out-of-distribution
and falls back to reading them.

### Adjective leakage caution
Some adjectives leak as spoken words even when used in the speaker desc.
Confirmed leaker — avoid:
- `tired` (the word "tired" comes out audibly)

When unsure, drop the adjective and use bare `A man` / `A woman`.

---

## Generation parameters

| Param | Default | When to override |
|-------|---------|------------------|
| `cfg_scale` | 2.5 | Lower = more natural prosody; higher (4-7) = more text-faithful but flatter dynamic range (auto-rescale kicks in at high cfg) |
| `stg_scale` | 1.5 | Skip-token guidance; rarely needs tuning |
| `duration_multiplier` | 1.1 | Audio-length budget = estimator × multiplier. Drop to **0.88-0.95** to speed up pacing; below 0.5 produces rushed/garbled output |
| `seed` | 42 | Same seed + same prompt = bit-identical output |
| `ref_duration` | 10.0 | First N seconds of voice ref used for cloning; longer ref doesn't help |

### Upsampler choice (24 → 48 kHz)

| Upsampler | When to use |
|-----------|-------------|
| `ltx` (built-in BWE) | **Sharp transients**: `[lip-smack]`, `[tongue-click]`, claps, gulps. Also faster (~0.1s vs ~3s for ReUSE). |
| `reuse` (NVIDIA SEMamba) | **Sustained sounds**: laughs, sighs, hums, dialogue. Denoises + bandwidth-extends but attenuates transients. |
| `ap_bwe` | AP-BWE feed-forward — alternative to ReUSE, less aggressive denoising |
| `audiosr` | Diffusion-based super-resolver — highest quality, slowest |

Default `ltx` is the safest choice for general dialogue. Use `reuse` when
the output sounds noisy and you want a cleaner result.

### Voice ref preprocessing

Voice refs with background noise, reverb, or low bandwidth produce worse
clones. Preprocessing options (provided as helper scripts in the wider
TTS repo):

1. **ElevenLabs audio-isolation** — cleanest dialogue separation, removes
   background music/noise. API-based (paid).
2. **NVIDIA RE-USE** — local model, denoises + bandwidth-extends.
   Free, runs on the same GPU as DramaBox.
3. **Sucial/Dereverb-Echo MBR (v2 SDR 13.48)** — dereverberation specifically.
   Use when the only problem is room reverb. Available via the ZFTurbo
   Music-Source-Separation-Training repo.
4. **Dolby Media API enhance** — full speech-isolation pipeline with
   loudness normalization, dynamics, noise reduction, sibilance/plosive
   reduction. API-based (paid).

Rule of thumb: voice ref ≥ 10 s, 24+ kHz, single speaker, no music.

---

## Paratag patterns (inline `[tag]` events)

Each section gives the prompt patterns that consistently produce the named
sound. `<speaker>` is from the universal rule. Default upsampler ReUSE
unless otherwise noted.

### `[gasp]` — sharp inhale of shock
```
A <speaker> lets out a sharp gasp, "<line>"
A <speaker> draws a sharp breath, "<line>"
A <speaker> gasps audibly, "<line>"
A <speaker> says <calm-modifier>, "<setup>" <pronoun> gasps, "<reveal>"
A <speaker> speaks calmly, "<setup>" <pronoun> gasps with shock, "<reaction>"
```
Example: `A nurse lets out a sharp gasp, "Doctor, his pulse is dropping!"`

### `[laugh]` — full audible laugh
```
A <speaker> bursts into laughter, "Hahaha! <line>"
A <speaker> laughs heartily, "Hahaha! <line>"
A <speaker> says, "<setup>" <pronoun> bursts into laughter, "Hahaha! <reaction>"
```
The phonetic `Hahaha` inside quotes is what produces the actual laugh
sound. Without it, the laugh action is much weaker.

### `[chuckle]` — short low laugh
```
A <speaker> chuckles darkly, "<line>"
A <speaker> lets out a quiet chuckle, "<line>"
A <speaker> speaks coldly, "<setup>" <pronoun> chuckles, "<reaction>"
```

### `[giggle]` — short high laugh
```
A <speaker> giggles uncontrollably, "Hehehe, <line>"
A <speaker> lets out a giggle, "Hehehe, <line>"
A <speaker> says, "<setup>" <pronoun> giggles, "Hehehe, <reaction>"
```
Validated primarily on female voices. Male giggles work less reliably —
fall back to `[laugh]` with shorter `Hehe` interjection if needed.

### `[sigh]` — exhale of resignation
```
A <speaker> sighs heavily, "<line>"
A <speaker> lets out a long sigh, "<line>"
A <speaker> says, "<setup>" <pronoun> sighs deeply, "<reaction>"
```

### `[hum-tune]` — humming a melody
```
A <speaker> hums a soft tune, "Mmmm-mmm, <line>"
A <speaker> hums quietly, "Mmmm-mmm, <line>"
A <speaker> says, "<setup>" <pronoun> hums quietly, "Mmmm-mmm, <line>"
```
Phonetic `Mmmm-mmm` inside quotes drives the actual hum.

### `[clears-throat]` — audible throat clear
```
A <speaker> clears his throat, "<line>"
A <speaker> gives a short throat clear, "<line>"
A <speaker> says, "<setup>" <pronoun> clears his throat, "<continuation>"
```

### `[cough]` — single cough
```
A <speaker> coughs once, "<line>"
A <speaker> lets out a dry cough, "<line>"
A <speaker> says, "<setup>" <pronoun> coughs, "<continuation>"
```

### `[yawn]` — audible yawn
```
A <speaker> yawns deeply, "Ugh, <line>"
A <speaker> lets out a big yawn, "<line>"
A <speaker> says, "<setup>" <pronoun> yawns deeply, "<continuation>"
```

### `[wheeze]` — breathless wheeze (often paired with laughter)
```
A <speaker> wheezes with laughter, "<line>"
A <speaker> lets out a wheeze, "<line>"
A <speaker> laughs, "Hahaha, <line>" <pronoun> wheezes, "<continuation>"
```

### `[exhale]` — slow audible outbreath
```
A <speaker> blows out a long exhale, "<line>"
A <speaker> releases a heavy exhale, "<line>"
A <speaker> breathes out audibly, "<line>"
```
Bare verbs (`exhales`, `lets out an exhale`) fail. Use stronger physical
verbs (`blows out`, `releases`) + the adverb `audibly`.

### `[inhale]` — sharp audible inbreath
```
A <speaker> sucks in a startled inhale, "<line>"
A <speaker> draws in a sharp inhale, "<line>"
A <speaker> says, "<setup>" <pronoun> audibly inhales, "<reaction>"
```
Same lesson as `[exhale]`: stronger physical verbs (`sucks in`, `draws in`)
+ sensory adjectives (`startled`, `sharp`) reliably fire.

### `[breath]` — neutral breathing event
```
A <speaker> breathes deeply, "<line>"
A <speaker> takes a shaky breath, "<line>"
A <speaker> takes a measured breath, "<line>"
A <speaker> pauses to take a breath, "<line>"
A <speaker> takes a deep breath, "<line>"
```
Use a speaker adjective (`A weary man`, `A nervous young woman`) — bare
`A man` produces inaudible breath.

### `[tsk]` — disapproval tongue tut
```
A <speaker> makes a tsk-tsk sound, "<line>"
```
The doubled `tsk-tsk` phonetic is required. `tsks disapprovingly` fails.
Use disapproval/judgment context for the dialogue.

### `[sniff]` — nasal sniff
```
A <speaker> lets out a wet sniffle, "<line>"
```
The word `wet` is required — without it the sniff is silent. Pair with
grief/emotional dialogue context.

### `[woo]` — cheer/whoop
```
A <speaker> cheers loudly, "Woooo! <line>"
A <speaker> lets out a cheer, "Woo hoo! <line>"
A <speaker> cheers, "Woooooo! <line>"
A <speaker> whoops loudly, "Wooo-hoo! <line>"
A <speaker> lets out a long whoop, "Wooooo!"
```
The phonetic content inside quotes drives the sound. Vary the number of
`o`s for cheer length.

### `[lip-smack]` — UPSAMPLER: `ltx` (NOT reuse)
```
A <speaker> says, "<food/drink setup>" <pronoun> makes a smack-smack sound, "<food reaction>"
```
**All required:**
1. Mid-dialogue placement (setup → action → reaction)
2. Doubled phonetic `smack-smack sound` (not `smacks his lips`)
3. Food/drink/eating context in both setup AND reaction
4. LTX BWE upsampler — ReUSE attenuates the wet transient

### `[tongue-click]` — UPSAMPLER: `ltx` (NOT reuse)
```
A <speaker> says, "<setup>" <pronoun> makes a click-click sound, "<reaction>"
A <speaker> says, "<setup>" <pronoun> clucks his tongue, "<reaction>"
```
Mid-dialogue placement + doubled phonetic `click-click sound` OR the verb
`clucks his tongue`. Standalone action fails. Use disapproval/judgment/
impatience context (complaints about late friends, broken promises).

---

## Style/emotion patterns (sustained delivery tone)

All styles work best as **multi-segment** prompts — a calm setup span
followed by an emotional contrast span. Single-segment style prompts
(e.g. `A man speaks angrily, "..."` alone) are unreliable.

### `<angry>` — restrained frustration → fury
```
A man speaks with fraying patience, "<setup>" He sighs deeply, "<follow-up>"
A man speaks evenly, "<setup>" His voice rises with fury, "<angry-line>"
A woman pauses, "<setup>" Her voice rises with fury, "<angry-line>"
A man speaks coldly, "<setup>" His voice rises with fury, "<angry-line>"
```

### `<tender>` — soft loving delivery
```
A woman speaks tenderly, "<line>"
A woman speaks tenderly, "<setup>" She hums quietly, "Mmmm-mmm, <follow-up>"
A weary woman speaks tenderly, "<setup>" She whispers, "<follow-up>"
A woman speaks tenderly, "<setup>" She laughs softly, "<follow-up>"
```

### `<menacing>` — cold threatening delivery
```
A man speaks with cold menace, "<line>"
A man speaks with cold menace, "<setup>" He chuckles darkly, "<follow-up>"
A man speaks with cold menace, "<setup>" His voice rises with fury, "<escalation>"
A woman speaks with cold menace, "<setup>" She chuckles darkly, "<follow-up>"
```

### `<loud>` — shouting/yelling
```
A man speaks with cold menace, "<setup>" His voice rises with fury, "<loud-line>"
A man speaks evenly, "<setup>" His voice rises with fury, "<loud-line>"
A man speaks with fraying patience, "<setup>" His voice rises with fury, "<LOUD-LINE-CAPS>"
A man pauses, "<setup>" His voice rises with fury, "<LOUD-LINE-CAPS>"
A woman pauses, "<setup>" Her voice rises with fury, "<LOUD-LINE-CAPS>"
```
**Always** use escalation: calm verb (`speaks evenly` / `speaks coldly` /
`pauses`) → `His/Her voice rises with fury`. Pure shouting verbs in
isolation are out-of-distribution. ALL-CAPS dialogue gives maximum volume.

### `<sad>` — grief, sorrow
```
A grieving woman weeps softly, "<line>"
A weary woman speaks heavily, "<setup>" She sighs with despair, "<follow-up>"
An elderly man speaks with sorrow, "<setup>" His voice trembles, "<follow-up>"
A woman sighs with despair, "<setup>" She speaks quietly, "<follow-up>"
A man speaks heavily, "<setup>" He weeps softly, "<follow-up>"
```

### `<joyful>` — exuberant happiness
```
A man bursts into uncontrollable laughter, "Hahaha! <line>"
A woman bursts into uncontrollable laughter, "Hahaha! <line>"
A young woman cries out in joy, "<exclamation>" She laughs through tears, "<follow-up>"
A man pauses, "<setup>" He cries out in joy, "<exclamation>"
A young woman speaks excitedly, "<setup>" She laughs joyously, "<follow-up>"
A girl already mid-giggle, "Hehehe, <line>" She gasps with joy, "<follow-up>"
```

### `<fearful>` — fear, terror
```
A terrified woman whispers in fear, "<setup>" She speaks shakily, "<follow-up>"
A scared child stammers, "<setup>" He whispers in fear, "<follow-up>"
A frightened woman screams in terror, "<line>"
A man pauses, "<setup>" He speaks in fear, "<follow-up>"
A woman speaks shakily, "<setup>" She begins to cry, "<follow-up>"
```

### `<nervous>` — uncertain, stammering
```
A nervous young woman stammers nervously, "<line>"
A young man speaks haltingly, "<setup>" He swallows nervously, "<follow-up>"
A young man clears his throat, "<setup>" He stammers nervously, "<follow-up>"
A nervous woman speaks shakily, "<setup>" She takes a deep breath, "<follow-up>"
A man speaks haltingly, "<setup>" He shifts uncomfortably, "<follow-up>"
```

### `<awe>` — wonder, astonishment
```
A woman gasps with shock, "<exclamation>" She speaks in disbelief, "<follow-up>"
A man speaks with quiet awe, "<setup>" He breathes out slowly, "<follow-up>"
An elderly woman gasps with shock, "<exclamation>" She speaks softly, "<follow-up>"
A man speaks with quiet awe, "<setup>" He laughs in wonder, "<follow-up>"
A young woman gasps audibly, "<exclamation>" She speaks in stunned disbelief, "<follow-up>"
```

### `<smug>` — confident self-satisfaction
```
A man speaks with smug pride, "<setup>" He chuckles confidently, "<follow-up>"
A confident woman speaks with smug pride, "<setup>" She laughs lightly, "<follow-up>"
A man speaks coolly, "<setup>" He speaks with smug pride, "<follow-up>"
```

### `<flirty>` — playful seduction
```
A flirtatious woman purrs flirtatiously, "<setup>" She laughs softly, "<follow-up>"
A woman speaks playfully, "<setup>" She purrs flirtatiously, "<follow-up>"
A flirtatious woman speaks teasingly, "<setup>" She whispers softly, "<follow-up>"
A woman speaks with playful charm, "<setup>" She laughs flirtatiously, "<follow-up>"
A flirtatious woman speaks softly, "<setup>" She giggles, "<follow-up>"
```

---

## Patterns that DON'T work in DramaBox

These are documented failure modes — listed so you know not to retry
them. Source these effects from a separate tool/dataset instead.

| Want | DramaBox does | Source elsewhere |
|------|---------------|------------------|
| `[cry]` (audible sobbing) | Produces low-amplitude soft sob even with phonetic primers (`Boohoo`, `Aaahahahaha`, `wails loudly`). Tried 4 rounds of prompt iteration — could not get loud crying reliably. | ElevenLabs TTS (has explicit `[cry]` paratag), recorded actor samples |
| `[clap]` | Sharp transient too weak even on LTX BWE. Clap sounds barely audible. | Sound-effect library |
| `[snort]` | Only works in a very narrow recipe (young speaker + anime voice ref + bare mouth-sound verb + broken-promise dialogue topic + LTX BWE). Too restrictive for scale. | ElevenLabs TTS |
| `<whisper>` as a style | DramaBox can't sustain a true whisper across multi-word dialogue. Best you get is "speaks softly". | Post-process audio gain + low-pass filter, OR source whisper-tagged dataset clips |

### Other failure modes to avoid

- **Role-based speakers** (`A drill sergeant`, `A spy`, `A teacher`,
  `A detective`, `A nurse`, `A scholar`, `A coach`): model speaks the role
  noun out loud. Always use generic speaker + adjective per the universal
  rule.

- **Single-verb style prompts** (`A man speaks angrily, "..."` alone):
  unreliable. Use multi-segment escalation instead.

- **Phonetic interjections inside quotes for paratags that don't have a
  matching onomatopoeia**: e.g. `"Ah!"` or `"Hah!"` for gasp doesn't help
  — the model reads them literally. The only paratags where phonetic
  content WORKS are `[laugh]` (`Hahaha`), `[chuckle]`/`[giggle]` (`Hehehe`),
  `[hum-tune]` (`Mmmm-mmm`), `[woo]` (`Woooo`), `[tsk]` (in stage direction:
  `tsk-tsk sound`), `[lip-smack]` (`smack-smack sound`), `[tongue-click]`
  (`click-click sound`).

- **Long stage directions with adjective-stuffed verbs** (`speaks with
  growing rage from his chest in barely-contained fury at being
  betrayed`): the parser breaks down on long stage directions and starts
  reading them as dialogue. Keep stage directions short (3-6 words).

---

## Worked examples

### Example 1: single-paratag prompt
```
A woman lets out a wet sniffle, "I miss him every single day."
```
Produces: a wet nasal sniff, then a quiet grieving delivery of the line.

### Example 2: multi-paratag chain
```
A man clears his throat, "Excuse me, pardon that." He settles into a
warm professional tone, "Good evening everyone, and welcome back to
the show."
```
Produces: throat-clear, beat, then a warm announcer-style delivery.

### Example 3: multi-segment style + paratag
```
A man speaks evenly, "I gave you one job to do." His voice rises with
fury, "AND YOU MANAGED TO MESS THAT UP TOO!"
```
Produces: calm setup, then a shouted angry response. ALL-CAPS reinforces
volume.

### Example 4: tender + hum combo (sample 3 from DramaBox's own demos)
```
A woman speaks tenderly, "It has been a long day, my love." She
whispers, "Close your eyes. I am right here." She hums quietly,
"Mmmm-mmm. Sleep now."
```
Produces: gentle delivery → quieter whisper → soft hum.

### Example 5: villain monologue
```
A man speaks with cold menace, "You have entered my domain, mortal."
He chuckles darkly, "Such arrogance will be your undoing." His voice
rises with fury, "Kneel, or be destroyed where you stand!"
```
Produces: menacing setup → dark chuckle → escalation to shout.

### Example 6: joyful laughter
```
A man bursts into uncontrollable laughter, "Hahaha! Oh my god, that
is the best news I have heard all year!"
```
Produces: actual laugh burst tied to the `Hahaha` phonetic, then a
joyful delivery of the rest.

---

## Reference: DramaBox demo samples

DramaBox ships a working demo set in `scripts/generate_samples.py`. The
patterns below are extracted from those demos and used as the basis for
this guide.

| Demo | Pattern |
|------|---------|
| Villain monologue | `speaks with cold menace` → `chuckles darkly` → `His voice rises with fury` |
| Talk-show wheeze-laugh | `gasps with shock` → `bursts into uncontrollable laughter` → `wheezes` |
| Tender goodnight | `speaks tenderly` → `whispers` → `hums quietly` |
| Radio anchor | `clears his throat` → `settles into a warm professional tone` |
| Catgirl giggling | `already mid-giggle` (giggle baked into speaker state) → `gasps for air between giggles` → `tries to compose herself` |
| Exhausted dad | `speaks with fraying patience` → `sighs deeply` → `puts on an overly cheerful voice` → `laughs helplessly` |

These six patterns are guaranteed to work because DramaBox ships them as
canonical demos. The patterns in this guide are extensions of those
demos applied to a wider tag set.

---

## Changelog

- **v1.0** — Initial guide consolidating 60+ prompt-engineering iterations.
  Coverage: 18 working paratags, 11 working style descriptors,
  4 documented failure modes.
