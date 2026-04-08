# Seinfeld Browser LLM — Design Spec

**Date:** 2026-04-08

## Overview

A client-side Seinfeld scene generator. User inputs a topic or situation; the app generates a speaker-labeled dialogue scene in the style of the show, running entirely in the browser with no server required after initial model download.

---

## Goals

- Generate Seinfeld-style scenes from a user-provided topic
- Run fully client-side (no API calls, no backend)
- Feature the main 4 characters: Jerry, George, Elaine, Kramer
- Scenes include a scene location tag and speaker-labeled dialogue

---

## Architecture

Four sequential phases:

### Phase 1 — Data Pipeline

**Input:** Seinfeld scripts scraped from the web  
**Output:** `data/processed/train.jsonl`

`scripts/download.py` scrapes all episode scripts from `seinfeldscripts.com/seinfeld-scripts.html` — parses the episode index page for individual script links, fetches each one, strips HTML, and saves as a raw `.txt` file to `data/raw/`. Handles rate-limits politely.

`scripts/preprocess.py` processes raw scripts into training examples:

- **Keep:** Scene location tags (`[JERRY'S APARTMENT]`), all JERRY/GEORGE/ELAINE/KRAMER dialogue lines
- **Strip:** Parenthetical cues `(entering)`, character descriptions inside scene tags, non-main-4 characters, line continuation indents
- **Topic derivation:** episode title + top 3 TF-IDF keywords extracted from the episode's dialogue, joined as a short phrase (e.g., `"The Parking Spot — spot, Buick, circling"`)
- Each processed example follows the training format (see below)

### Phase 2 — Training

**Platform:** Google Colab Pro (A100)  
**Model:** `gpt2-medium` (355M params)  
**Method:** Full finetune via HuggingFace `Trainer`  
**Output:** `model/gpt2-seinfeld/` (HF checkpoint)

Training format per example:
```
TOPIC: {topic derived from episode}

[SCENE LOCATION]

JERRY: ...

GEORGE: ...

ELAINE: ...

KRAMER: ...

[END]
```

Each example ends with `[END]` as a stop token so generation knows when to halt.

**Notebook:** `training/finetune.ipynb`

### Phase 3 — Export

**Tool:** HuggingFace `optimum`  
**Output:** `web/model/` — `model.onnx` (int8 quantized, ~450MB) + `tokenizer.json`

`scripts/export.py` converts the HF checkpoint to ONNX and applies int8 quantization. The output is everything Transformers.js needs for browser inference.

### Phase 4 — Web App

**Runtime:** Transformers.js (WASM, no WebGPU required)  
**Files:** `web/index.html`, `web/app.js`

**User flow:**
1. First visit: model downloads (~450MB) with a progress bar — cached in browser after
2. User types a topic/situation
3. App builds prompt: `TOPIC: {input}\n\n[`
4. Model generates until `[END]` or max tokens (~300)
5. Output parsed line-by-line into scene tag + speaker dialogue blocks
6. Rendered with scene location in italic, each speaker name bolded

**UI actions:** Generate, Regenerate, Copy

---

## Data Format Detail

### Inference prompt
```
TOPIC: {user input}

[
```
The model completes the `[` into a scene location tag, then generates the full dialogue.

### Output parsing
Split on newlines. Lines matching `^(JERRY|GEORGE|ELAINE|KRAMER):` are rendered as dialogue entries. Lines matching `^\[` and not `\[END\]` are rendered as scene tags.

---

## Repo Structure

```
gpt-seinfeld/
├── data/
│   ├── raw/              # original scripts
│   └── processed/        # train.jsonl
├── scripts/
│   ├── download.py       # scrape scripts → data/raw/
│   ├── preprocess.py     # raw → training examples
│   └── export.py         # HF checkpoint → ONNX int8
├── training/
│   └── finetune.ipynb    # Colab Pro training notebook
├── model/
│   └── gpt2-seinfeld/    # HF checkpoint (intermediate, not committed)
└── web/
    ├── index.html
    ├── app.js
    └── model/            # model.onnx + tokenizer.json (committed)
```

---

## Key Constraints & Decisions

| Decision | Choice | Reason |
|---|---|---|
| Model | GPT-2 Medium (355M) | Best quality/size ratio; ~450MB quantized runs CPU/WASM |
| Training | Full finetune | Simpler than LoRA; A100 makes it fast enough |
| Browser runtime | Transformers.js | No WebGPU required; broad browser support |
| Characters | Main 4 only | Clean training signal; avoids sparse character data |
| Stop token | `[END]` | Explicit scene boundary for reliable generation halting |

---

## Out of Scope

- Character-specific chatbot mode
- Streaming token output (render full scene at once)
- Mobile-optimized layout
- Streaming token output (render full scene at once)
- Mobile-optimized layout
