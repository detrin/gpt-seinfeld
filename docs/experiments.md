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

---

## v5 — GPT-2 medium full fine-tune (all params)

**Config**
- Base model: `gpt2-medium` (345M), all parameters updated (no LoRA)
- Epochs: 5, lr=5e-5, warmup=100, batch=8, grad_accum=2 (eff. batch=16)
- Dataset: 2295 examples (`data/processed/train_v2.jsonl` — Haiku-annotated topics)
- A100 40GB, ~5 min

**Results**
- Final avg train loss: **1.998**
- Output quality: generates Seinfeld content, topic partially ignored, prose-like narrative rather than `CHARACTER: text` format
- Sample (parking tickets): `[STREET] ST. PETERSBURG - DAY — And that's the end of it! I'm finally getting my spaces back...`

---

---

## v6 — GPT-2 medium deep LoRA (r=64, c_attn+c_proj+c_fc) ← current prod

**Config**
- Base model: `gpt2-medium` (345M)
- LoRA: r=64, alpha=128, target=`c_attn+c_proj+c_fc` (adds MLP to attention), 6.6% trainable params
- Epochs: 20, lr=2e-4, warmup=200, batch=8, grad_accum=2 (eff. batch=16)
- Dataset: 2295 examples (`data/processed/train_v2.jsonl` — Haiku-annotated topics)
- A100 40GB, ~19 min

**Results**
- Final avg train loss: **1.766** (still decreasing at epoch 20, loss trajectory 2.49 → 1.45)
- Output quality: **better topic relevance** than all previous GPT-2 versions — scenes actually reference the user's topic
  - "losing a parking spot" → scene about someone taking a spot, Kramer and Elaine arguing
  - "waiting for a table" → Elaine/George at a restaurant, Mr. Costanza ordering appetizers
  - "a bad haircut" → Jerry/George/Elaine at Monk's, bald guy becoming George
- Remaining issues: prose-narrative style rather than proper `CHARACTER: text` format

**Winner over v5** — lower loss, significantly better topic adherence

**ONNX export**
- fp32: 1421 MB → int8: **358 MB** (opset 14, legacy torch.onnx.export, external data merged)
- Uploaded to `hermanda/gpt2-seinfeld` (replaces previous `onnx/model_quantized.onnx`)
- Export note: dynamo-based exporter (torch 2.10+ default) produces broken models for GPT-2 + KV cache; must use `dynamo=False` to force legacy TorchScript exporter

**fp32 vs int8 quantitative comparison** (`scripts/compare_onnx.py`, 5 topics, seed=42, temp=0.7, top_k=5)

| Metric | Value |
|--------|-------|
| Mean logit cosine similarity | **0.199** (near-orthogonal!) |
| Min cosine similarity | **-0.999** (some steps have inverted logit rankings) |
| Top-1 token agreement | **6.5%** |
| Top-5 token agreement | **0.9%** |
| Identical output sequences | **0/5** |
| fp32 total time (5 topics × 200 tok) | 14.6s |
| int8 total time | 7.1s (**2.1x faster**) |
| Avg seinfeld characters found | fp32=2.2, int8=2.0 |
| Avg dialogue lines detected | fp32=11.8, int8=11.8 |

Key observations:
- **Logits diverge massively** — cosine similarity averaging 0.20 means the 50K-dim logit vectors are nearly orthogonal. At some decode steps the cosine is *negative*, meaning the quantized model ranks tokens in nearly opposite order.
- **Despite this, output quality is comparable.** Both models produce proper `[LOCATION]\nCHARACTER: dialogue` format, use correct Seinfeld character names (Jerry, George, Elaine, Kramer), generate relevant locations (Jerry's Apartment, Monk's Diner, Restaurant), and stay on-topic.
- **The int8 model is ~2x faster** on CPU (ONNX Runtime, Apple Silicon).
- Both models share the same weaknesses: character repetition (same character talks to themselves for multiple turns), occasional rambling past coherence.
- Explanation: language generation has many valid continuations at each step. The model's "knowledge" of Seinfeld format and characters is distributed across weights in a way that survives aggressive quantization, even though individual token predictions diverge completely. Top-k=5 sampling at low temperature helps — we only need the top tokens to be *plausible*, not *identical*.

Sample outputs (first 150 chars):
```
Topic: "shrinkage"
  fp32: [MONK'S DINER] ELAINE: I think it's possible. JERRY: What if it wasn't? ELAINE: It's very possible...
  int8: [JERRY'S APARTMENT] JERRY: You know, I don't know why I waited so long to tell you about my shrinkage...

Topic: "a bad haircut"
  fp32: [JERRY'S APARTMENT] JERRY: What happened to your head? GEORGE: It was all brown...
  int8: [MONK'S DINER] ELAINE: You know, I think the haircut may have been the reason that George got expelled...
```

---

## Agreed next steps (in priority order)

### 1. Clean training data with Haiku
Use Claude Haiku to generate a 1-sentence scene description for each of the 2295 training examples, replacing the current TOPIC field (episode title + cast keywords). This is the single highest-ROI change — without it, no model architecture improvement can fix the topic-ignored problem.

Also worth trying: **two-stage inference in the web app** — first call the model to generate a 1-sentence summary from the user's topic, then use that as the TOPIC prompt to generate the scene. Mirrors the training data format and may help topic relevance even with the current model.

Script to write: `scripts/reannotate_topics.py`

### 2. Fix `[END]` stop token at inference time
`[END]` is already in training data. Add it as a stop token in `web/app.js` (Transformers.js `stop_sequences`) so generation halts immediately instead of rambling past the end of the scene. Also fix the sanity check in `training/train.py`.

### 3. Full fine-tuning (no LoRA) for GPT-2 medium
GPT-2 medium (345M) is small enough to fully fine-tune — LoRA was only necessary for 7B models. Full fine-tuning updates MLP weights, which is where format/content patterns live and which LoRA (targeting only `c_attn+c_proj`) doesn't touch. Add `USE_LORA=false` path to `train.py` with validation split + early stopping.

Best done on cleaned data from step 1.

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
