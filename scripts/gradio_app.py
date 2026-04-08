"""Gradio debug app for gpt2-seinfeld — pure onnxruntime + tiktoken (no torch)."""

from __future__ import annotations

import re

import gradio as gr
import numpy as np
import onnxruntime as ort
import tiktoken

MODEL_PATH = "/tmp/gpt2-seinfeld/onnx/model_quantized.onnx"
EOS_ID = 50256  # GPT-2 <|endoftext|>

print("Loading tokenizer...")
enc = tiktoken.get_encoding("gpt2")

print("Loading ONNX session...")
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
print("Model loaded.")

# Inspect model I/O
_in_names = [i.name for i in session.get_inputs()]
_out_names = [o.name for o in session.get_outputs()]
print("Inputs:", _in_names)
print("Outputs:", _out_names)


N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64


def _sample(logits: np.ndarray, seen: set[int], temperature: float, rep_penalty: float) -> int:
    logits = logits.astype(np.float64)
    for tid in seen:
        logits[tid] = logits[tid] * rep_penalty if logits[tid] < 0 else logits[tid] / rep_penalty
    logits /= max(temperature, 1e-8)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_tokens(prompt: str, max_new: int, temperature: float, rep_penalty: float) -> list[int]:
    prompt_ids = enc.encode(prompt)
    seen = set(prompt_ids)
    generated: list[int] = []

    # Initial empty KV cache: (1, n_heads, 0, head_dim)
    empty = np.empty((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
    past = {f"past_key_values.{i}.{kv}": empty for i in range(N_LAYERS) for kv in ("key", "value")}

    # First forward pass: full prompt
    seq = np.array([prompt_ids], dtype=np.int64)
    attn = np.ones((1, len(prompt_ids)), dtype=np.int64)
    pos = np.arange(len(prompt_ids), dtype=np.int64)[None]

    feeds = {"input_ids": seq, "attention_mask": attn, "position_ids": pos, **past}
    outputs = session.run(None, feeds)
    logits_all = outputs[0]  # (1, prompt_len, vocab)
    present = outputs[1:]  # 2*N_LAYERS arrays

    next_id = _sample(logits_all[0, -1], seen, temperature, rep_penalty)
    generated.append(next_id)
    seen.add(next_id)
    past_len = len(prompt_ids)

    # Update past from present
    past = {
        f"past_key_values.{i}.{kv}": present[i * 2 + j]
        for i in range(N_LAYERS)
        for j, kv in enumerate(("key", "value"))
    }

    for step in range(max_new - 1):
        if next_id == EOS_ID:
            break
        past_len += 1
        seq = np.array([[next_id]], dtype=np.int64)
        attn = np.ones((1, past_len + 1), dtype=np.int64)
        pos = np.array([[past_len]], dtype=np.int64)

        feeds = {"input_ids": seq, "attention_mask": attn, "position_ids": pos, **past}
        outputs = session.run(None, feeds)
        logits_all = outputs[0]
        present = outputs[1:]

        next_id = _sample(logits_all[0, -1], seen, temperature, rep_penalty)
        generated.append(next_id)
        seen.add(next_id)

        past = {
            f"past_key_values.{i}.{kv}": present[i * 2 + j]
            for i in range(N_LAYERS)
            for j, kv in enumerate(("key", "value"))
        }

    return generated


def parse_scene(text: str) -> str:
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line == "[END]":
            break
        if line.startswith("[") and line.endswith("]"):
            lines.append(f"*{line}*\n")
            continue
        m = re.match(r"^([A-Z][A-Z\s]{1,20}):\s*(.+)", line)
        if m:
            lines.append(f"**{m.group(1).strip()}**: {m.group(2)}")
    return "\n".join(lines) if lines else "(no parseable dialogue)"


def generate(
    topic: str, max_tokens: int, temperature: float, rep_penalty: float
) -> tuple[str, str]:
    if not topic.strip():
        return "", ""
    prompt = f"TOPIC: {topic}\n\n"
    token_ids = generate_tokens(prompt, int(max_tokens), temperature, rep_penalty)
    raw = enc.decode(token_ids)
    return raw, parse_scene(raw)


with gr.Blocks(title="Seinfeld Scene Generator — Debug") as demo:
    gr.Markdown(
        "## Seinfeld Scene Generator — Debug\n"
        "Loads `model_quantized.onnx` from `/tmp/gpt2-seinfeld/`"
    )

    topic = gr.Textbox(label="Topic / Situation", placeholder="e.g. losing a parking spot")

    with gr.Row():
        max_tokens = gr.Slider(50, 400, value=200, step=10, label="Max tokens")
        temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
        rep_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty")

    btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        raw_out = gr.Textbox(label="Raw output", lines=12)
        parsed_out = gr.Markdown(label="Parsed scene")

    btn.click(
        generate,
        inputs=[topic, max_tokens, temperature, rep_penalty],
        outputs=[raw_out, parsed_out],
    )

if __name__ == "__main__":
    demo.launch()
