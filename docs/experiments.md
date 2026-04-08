# Experiment Log

## v1 — GPT-2 medium LoRA (baseline)

**Config**
- Base model: `gpt2-medium` (345M)
- LoRA: r=8, alpha=16, target=`c_attn+c_proj`
- Epochs: 10, lr=2e-4, warmup=100
- Dataset: 2264 examples (`data/processed/train.jsonl`)

**Results**
- Final loss: **2.09**
- Output quality: **broken** — generated nonsense like `[RADOPTACULAR HOTEL CAB]`, no proper `CHARACTER: text` format, completely off-topic content
- Diagnosis: model never learned the dialogue format; loss too high

**Notes**
- ONNX int8 export: 357 MB (`hermanda/gpt2-seinfeld` on HuggingFace, v1)
- Web app prompt was `TOPIC: {topic}\n\n` (no `[` prefix) — this caused the model to ramble before reaching format

---

## v2 — GPT-2 medium LoRA (improved)

**Config**
- Base model: `gpt2-medium` (345M)
- LoRA: r=16, alpha=32, target=`c_attn+c_proj`
- Epochs: 20, lr=3e-4, warmup=200
- Dataset: 2295 examples (same data, minor additions)
- Saved to: `/content/gpt2-seinfeld-v2` (Colab local, not Drive)

**Results**
- Final loss: **1.83** (down from 2.09)
- Output quality: **recognizable** — correct character names (JERRY, GEORGE, ELAINE, KRAMER), correct locations ([JERRY'S APARTMENT], [MONK'S DINER], [GEORGE'S APARTMENT]), Seinfeld-ish tone
- Remaining issues:
  - Topics are mostly **ignored** — model generates Seinfeld content but doesn't connect to the given topic (training topics were episode titles like "The Abstinence — jackie, doctor", not scene descriptions)
  - Missing `\n\n` between character turns (all jammed on one line in ~80% of outputs)
  - **Rambling** after ~80–100 tokens — loses coherence
  - Occasional wrong character names (JERRORY, ANDELBAUM)

**Key fix discovered**
- Prompt must end with `\n\n[` not `\n\n` — this primes the model into scene-tag mode and produces much cleaner output
- Without `[`: meanders into prose before finding format
- With `[`: immediately outputs `[LOCATION]CHARACTER: text`

**ONNX export**
- fp32: 1420 MB → int8: **357 MB**
- Uploaded to `hermanda/gpt2-seinfeld` (replaces v1, same filename)

---

## Dataset notes

- 2295 training examples, each a scene chunk from a Seinfeld episode
- Format: `TOPIC: Episode Title — keyword1, keyword2\n\n[LOCATION]\n\nCHARACTER: text\n\n...\n\n[END]`
- Problem: TOPIC field is the **episode title + cast keywords**, not a description of the scene content → model cannot map user topics to specific scene content
- Potential fix: replace TOPIC with a 1-sentence description of what happens in the scene

---

---

## v3 — Qwen2.5-7B QLoRA (base model, quality target)

**Config**
- Base model: `Qwen/Qwen2.5-7B` (7B)
- LoRA: r=16, alpha=32, target=`q/k/v/o/gate/up/down_proj`
- Epochs: 3, lr=2e-4, warmup=100, batch=2, grad_accum=8 (eff. batch=16)
- QLoRA 4-bit (nf4, bfloat16), A100 80GB
- Dataset: 2295 examples

**Results**
- Training time: ~36 min on A100 80GB
- Loss trajectory: 2.38 → 1.618 (still decreasing at epoch 3)
- Final avg train loss: **1.881**
- Output quality: **garbage** — model generates code/math (`[TEA sequence response = 43.221000%...`), not dialogue

**Diagnosis**
- Qwen2.5-7B **base** was pretrained heavily on code/math; `TOPIC: parking tickets\n\n[` triggers code-completion mode
- 3 epochs of LoRA (40M trainable / 7.66B total) not enough to override strong base priors
- Loss still decreasing → model hadn't converged; more epochs (6-10) may help, but base priors are the root issue

**Next step**: try `Qwen/Qwen2.5-7B-Instruct` — RLHF'd for instruction following, more format-obedient

---

---

## v4 — Qwen2.5-7B-Instruct QLoRA (instruct model)

**Config**
- Same as v3 but base model: `Qwen/Qwen2.5-7B-Instruct`
- Epochs: 3, A100 80GB, ~37 min

**Results**
- Final avg train loss: **1.903**
- Output quality: **wrong format** — model generates instruction-following responses
  - "losing a parking spot" → math forum post about permutations
  - "a bad haircut" → `[TEACHER]`/`[STUDENT]` Q&A format
  - "waiting for a table" → factual article with suggestions
  - The `[` prompt prefix is interpreted as forum markup, Q&A, or numbered list — not `[LOCATION]`

**Diagnosis**
- RLHF/instruct training is too strong to override with 3 epochs of plain-text LoRA
- Our training format `TOPIC: ...\n\n[LOCATION]\nCHARACTER: text` was designed for a base model
- Instruct models expect the chat template (`<|im_start|>system\n...<|im_end|>`) — plain continuation doesn't work

**Path forward**
To make an instruct model work, training data must be reformatted using the chat template:
```
<|im_start|>system
You are a writer for the TV show Seinfeld. Write a scene in the exact format:
[LOCATION]
CHARACTER: dialogue
...[END]
<|im_end|>
<|im_start|>user
TOPIC: {topic}
<|im_end|>
<|im_start|>assistant
[LOCATION]
CHARACTER: ...
```
This requires reannotating training data. Alternatively, train the BASE model with more epochs.

---

## Ideas for next experiments

### SmolLM2-360M (browser target)
- ~360 MB int8, same deployment size as current GPT-2 medium
- Much better base model (2024, trained on FineWeb-Edu + SMOL data)
- Expected: better format-following, less rambling
- Same LoRA recipe should work

### Llama 3.1 8B (quality target)
- Not browser-viable, but for quality benchmarking
- A100 80GB fits comfortably in fp16 with LoRA r=16
- Expected: should follow topic prompts much better, coherent multi-turn dialogue
- Inference via Colab pipeline or API, not web app

### Data improvements
- Replace episode-title TOPIC with scene-content description (needs reannotation or LLM-generated summaries)
- Filter out thin scenes (< 4 character lines) — currently included, adds noise
- Consider character-turn chunking instead of scene chunking to get more examples
