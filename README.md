# GPT-Seinfeld

Fine-tuned language models that generate Seinfeld scenes, running entirely in your browser.

**[Try it live](https://gpt-seinfeld.hermandaniel.com)**

## How it works

Two models were fine-tuned on ~2,300 Seinfeld script excerpts to learn the show's dialogue style, character voices, and scene structure. Both run client-side via WebAssembly — no server, no API keys.

| Model | Method | Params trained | Size served | Runtime |
|-------|--------|---------------|-------------|---------|
| Llama 3.2 3B | QLoRA (4-bit base + LoRA r=32) | 1.6% (51M / 3.2B) | ~1.9 GB GGUF Q4_K_M | wllama (llama.cpp WASM) |
| GPT-2 Medium | LoRA (r=64, attention + MLP) | 6.6% (22M / 345M) | ~358 MB ONNX int8 | Transformers.js (ONNX Runtime WASM) |

## Fine-tuning notebooks

Step-by-step notebooks with detailed explanations of LoRA, QLoRA, quantization, and browser deployment.

| Notebook | Model | GPU | Time | |
|----------|-------|-----|------|-|
| GPT-2 Medium LoRA | 345M params | Free T4 | ~19 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/detrin/gpt-seinfeld/blob/main/notebooks/finetune_gpt2.ipynb) |
| Llama 3.2 3B QLoRA | 3.2B params | A100 | ~25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/detrin/gpt-seinfeld/blob/main/notebooks/finetune_llama.ipynb) |

### What's covered

- **LoRA** — how low-rank adaptation works, rank selection, alpha/r scaling, which layers to target
- **QLoRA** — 4-bit NormalFloat quantization, double quantization, bf16 adapters on 4-bit base
- **ONNX int8 quantization** — dynamic quantization for browser deployment via Transformers.js
- **GGUF Q4_K_M conversion** — llama.cpp format for browser deployment via wllama, sharding for browser memory limits

## Project structure

```
data/
  processed/          Training data (2,295 JSONL examples)
training/
  train.py            Multi-model training script (GPT-2, Llama, Qwen configs)
notebooks/
  finetune_gpt2.ipynb   GPT-2 LoRA fine-tuning + ONNX export
  finetune_llama.ipynb  Llama 3.2 QLoRA fine-tuning + GGUF conversion
scripts/
  preprocess.py       Raw script → training data pipeline
  download.py         Scrape episode transcripts
  export.py           HF model → ONNX int8 export
  reannotate_topics.py  LLM-based topic reannotation
web/                  Astro static site (deployed to Vercel)
  src/pages/          Page templates
  public/scripts/     Client-side JS (model backends, post-processing, rendering)
docs/
  experiments.md      Full experiment log (v1–v6)
```

## Training data format

```
TOPIC: debating who to eat in a plane crash

[OUTSIDE LOCATION]

GEORGE: Say you, me, and Kramer are flying over the Andes.
JERRY: Why are we flyin' over the Andes?
GEORGE: We got a soccer game in Chile. Anyway, the plane crashes.
...
[END]
```

## Models on HuggingFace

- [hermanda/gpt2-seinfeld](https://huggingface.co/hermanda/gpt2-seinfeld) — ONNX int8
- [hermanda/Llama-3.2-3B-Seinfeld-GGUF](https://huggingface.co/hermanda/Llama-3.2-3B-Seinfeld-GGUF) — GGUF Q4_K_M (4 shards)

## Local development

```bash
cd web
npm install
npm run dev    # http://localhost:4321
```

Requires COOP/COEP headers for SharedArrayBuffer (configured in `astro.config.mjs` for dev, `vercel.json` for production).

## Disclaimer

This is a fan project made out of love for the show. Not affiliated with or endorsed by the creators of Seinfeld, Castle Rock Entertainment, or Sony Pictures Television.
