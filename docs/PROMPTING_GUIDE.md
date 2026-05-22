# DramaBox Prompting Guide

Quick reference for writing prompts that fire reliably. Every pattern below was validated by listening.

---

## The pattern

```
A <speaker> <verb>, "<dialogue>" <pronoun> <verb>, "<dialogue>"
```

- **Quoted text is spoken literally** (including `Hahaha`, `Mmmm-mmm`, `Woooo`).
- **Unquoted text is stage direction** — shapes delivery, not spoken.
- **End at the last closing quote.** Trailing description gets ignored or read.
- **Multi-segment beats single-segment.** Calm setup → emotional contrast fires emotions far more reliably than one expressive verb alone.

```
A man speaks evenly, "I gave you one job."
His voice rises with fury, "AND YOU MESSED IT UP!"
```

---

## Speaker rule

Generic noun phrase + optional **one-word** adjective. Never use roles.

| ✅ Good | ❌ Bad (model speaks the role noun) |
|---|---|
| `A man`, `A woman`, `A young woman`, `An elderly man`, `A child` | `A radio host`, `A drill sergeant`, `A spy`, `A detective`, `A teacher`, `A nurse` |
| `A weary man`, `A grieving woman`, `A nervous young woman`, `A flirtatious woman`, `A scared child` | adjectives stacked >1, or full descriptions like `A late-night radio host with a warm, smoky voice` |

---

## Paratags (validated)

Use the phonetic content **inside the quote** to trigger the actual sound. Stage-direction verbs alone are weak.

| Want | Pattern |
|---|---|
| Laugh | `A <speaker> bursts into uncontrollable laughter, "Hahaha! <line>"` |
| Chuckle | `A <speaker> chuckles darkly, "<line>"` |
| Giggle | `A <speaker> giggles, "Hehehe, <line>"` *(reliable on female voices)* |
| Sigh | `A <speaker> sighs heavily, "<line>"` |
| Gasp | `A <speaker> says, "<setup>" <pronoun> gasps with shock, "<reaction>"` |
| Hum | `A <speaker> hums quietly, "Mmmm-mmm, <line>"` |
| Throat clear | `A <speaker> clears his throat, "<line>"` |
| Cough | `A <speaker> coughs once, "<line>"` |
| Yawn | `A <speaker> yawns deeply, "Ugh, <line>"` |
| Wheeze | `A <speaker> wheezes with laughter, "<line>"` |
| Exhale | `A <speaker> blows out a long exhale, "<line>"` |
| Inhale | `A <speaker> sucks in a startled inhale, "<line>"` |
| Breath | `A weary man takes a shaky breath, "<line>"` *(speaker adjective required)* |
| Tsk | `A <speaker> makes a tsk-tsk sound, "<line>"` *(doubled phonetic required)* |
| Sniffle | `A <speaker> lets out a wet sniffle, "<line>"` *(word "wet" required)* |
| Cheer | `A <speaker> cheers loudly, "Woooo! <line>"` |

---

## Style / emotion

Each style works best as a **two-span prompt**: calm setup → emotional payoff.

| Tone | Pattern |
|---|---|
| Angry / loud | `A man speaks evenly, "<setup>" His voice rises with fury, "<LINE-IN-CAPS>"` |
| Tender | `A woman speaks tenderly, "<setup>" She hums quietly, "Mmmm-mmm, <follow-up>"` |
| Menacing | `A man speaks with cold menace, "<setup>" He chuckles darkly, "<follow-up>"` |
| Sad | `A grieving woman weeps softly, "<setup>" She sighs with despair, "<follow-up>"` |
| Joyful | `A man bursts into uncontrollable laughter, "Hahaha! <line>"` |
| Fearful | `A terrified woman speaks shakily, "<setup>" She begins to cry, "<follow-up>"` |
| Nervous | `A young man clears his throat, "<setup>" He stammers nervously, "<follow-up>"` |
| Awe | `A man speaks with quiet awe, "<setup>" He breathes out slowly, "<follow-up>"` |
| Smug | `A man speaks with smug pride, "<setup>" He chuckles confidently, "<follow-up>"` |
| Flirty | `A flirtatious woman purrs flirtatiously, "<setup>" She laughs softly, "<follow-up>"` |

Use the speaker's gendered pronoun (`He` / `She`) in continuation verbs.

