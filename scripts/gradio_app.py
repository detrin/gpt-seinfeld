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
print("Inputs:", _in_names[:3], "...")
print("Outputs:", _out_names[:4], "...")


N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64


def _sample(logits: np.ndarray, temperature: float, top_k: int) -> int:
    """Top-k sampling — more stable than full-vocab sampling for int8 quantized models."""
    logits = logits.astype(np.float64)
    top_ids = np.argsort(logits)[-top_k:]
    top_logits = logits[top_ids]
    top_logits /= max(temperature, 1e-8)
    top_logits -= top_logits.max()
    probs = np.exp(top_logits)
    probs /= probs.sum()
    return int(top_ids[int(np.random.choice(top_k, p=probs))])


def _decode_step(token_id: int, past_len: int, past: dict) -> tuple[np.ndarray, dict]:
    """Single decode step. attention_mask shape = (1, past_len+1) as the model expects."""
    feeds = {
        "input_ids": np.array([[token_id]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
        "position_ids": np.array([[past_len]], dtype=np.int64),
        **past,
    }
    outputs = session.run(None, feeds)
    present = outputs[1:]
    new_past = {
        f"past_key_values.{i}.{kv}": present[i * 2 + j]
        for i in range(N_LAYERS)
        for j, kv in enumerate(("key", "value"))
    }
    return outputs[0][0, -1], new_past  # logits (vocab,), new_past


def generate_tokens(prompt: str, max_new: int, temperature: float, top_k: int) -> list[int]:
    prompt_ids = enc.encode(prompt)
    generated: list[int] = []

    # Token-by-token prefill (model expects attention_mask of shape (1, past_len+1))
    empty = np.empty((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
    past = {f"past_key_values.{i}.{kv}": empty for i in range(N_LAYERS) for kv in ("key", "value")}
    for pos, tok in enumerate(prompt_ids):
        logits, past = _decode_step(tok, pos, past)
    past_len = len(prompt_ids)

    # Generate
    next_id = _sample(logits, temperature, top_k)
    generated.append(next_id)

    for _ in range(max_new - 1):
        if next_id == EOS_ID:
            break
        logits, past = _decode_step(next_id, past_len, past)
        past_len += 1
        next_id = _sample(logits, temperature, top_k)
        generated.append(next_id)

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


def generate(topic: str, max_tokens: int, temperature: float, top_k: int) -> tuple[str, str]:
    if not topic.strip():
        return "", ""
    prompt = f"TOPIC: {topic}\n\n["
    token_ids = generate_tokens(prompt, int(max_tokens), temperature, int(top_k))
    raw = enc.decode(token_ids)
    full = "[" + raw  # re-add the '[' we primed with
    return full, parse_scene(full)


with gr.Blocks(title="Seinfeld Scene Generator — Debug") as demo:
    gr.Markdown(
        "## Seinfeld Scene Generator — Debug\n"
        "Loads `model_quantized.onnx` from `/tmp/gpt2-seinfeld/`"
    )

    topic = gr.Textbox(label="Topic / Situation", placeholder="e.g. losing a parking spot")

    with gr.Row():
        max_tokens = gr.Slider(50, 400, value=200, step=10, label="Max tokens")
        temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
        top_k = gr.Slider(1, 50, value=5, step=1, label="Top-k (1 = greedy)")

    btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        raw_out = gr.Textbox(label="Raw output", lines=12)
        parsed_out = gr.Markdown(label="Parsed scene")

    btn.click(
        generate,
        inputs=[topic, max_tokens, temperature, top_k],
        outputs=[raw_out, parsed_out],
    )

if __name__ == "__main__":
    demo.launch()
