"""Evaluate int8 ONNX model quality across diverse topics."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import tiktoken

INT8_PATH = "/tmp/gpt2-seinfeld/onnx/model_quantized.onnx"
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64
EOS_ID = 50256

TOPICS = [
    "losing a parking spot",
    "a bad haircut",
    "waiting for a table at a restaurant",
    "forgetting someone's name",
    "shrinkage",
    "a close talker",
    "double dipping a chip",
    "getting stuck in traffic",
    "a terrible first date",
    "returning a jacket without a receipt",
    "being put on hold forever",
    "a loud neighbor",
    "finding a deal on a couch",
    "jury duty",
    "an annoying laugh",
]

enc = tiktoken.get_encoding("gpt2")

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(INT8_PATH, opts, providers=["CPUExecutionProvider"])


def empty_past() -> dict:
    e = np.empty((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
    return {f"past_key_values.{i}.{kv}": e for i in range(N_LAYERS) for kv in ("key", "value")}


def decode_step(token_id: int, past_len: int, past: dict):
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
    return outputs[0][0, -1], new_past


def sample_topk(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits.astype(np.float64)
    top_ids = np.argsort(logits)[-top_k:]
    top_logits = logits[top_ids]
    top_logits /= max(temperature, 1e-8)
    top_logits -= top_logits.max()
    probs = np.exp(top_logits)
    probs /= probs.sum()
    return int(top_ids[int(np.random.choice(top_k, p=probs))])


def generate(topic: str, seed: int = 42) -> str:
    np.random.seed(seed)
    prompt = f"TOPIC: {topic}\n\n["
    prompt_ids = enc.encode(prompt)

    past = empty_past()
    for pos, tok in enumerate(prompt_ids):
        logits, past = decode_step(tok, pos, past)
    past_len = len(prompt_ids)

    generated = []
    next_id = sample_topk(logits, 0.7, 5)
    generated.append(next_id)

    for _ in range(299):
        if next_id == EOS_ID:
            break
        logits, past = decode_step(next_id, past_len, past)
        past_len += 1
        next_id = sample_topk(logits, 0.7, 5)
        generated.append(next_id)

    return "[" + enc.decode(generated)


if __name__ == "__main__":
    for i, topic in enumerate(TOPICS):
        text = generate(topic, seed=42 + i)
        print(f"{'═' * 70}")
        print(f"TOPIC: {topic}")
        print(f"{'─' * 70}")
        print(text)
        print()
