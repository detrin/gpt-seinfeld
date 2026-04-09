"""Test inference-time improvements: prompt variants, best-of-N, post-processing."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import onnxruntime as ort
import tiktoken

INT8_PATH = "/tmp/gpt2-seinfeld/onnx/model_quantized.onnx"
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64
EOS_ID = 50256

SEINFELD_CHARS = {
    "JERRY",
    "GEORGE",
    "ELAINE",
    "KRAMER",
    "NEWMAN",
    "MORTY",
    "FRANK",
    "ESTELLE",
    "SUSAN",
    "PETERMAN",
    "PUDDY",
}

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


def generate(
    prompt: str,
    max_new: int = 250,
    temperature: float = 0.7,
    top_k: int = 8,
    seed: int = 42,
) -> str:
    np.random.seed(seed)
    prompt_ids = enc.encode(prompt)
    past = empty_past()
    for pos, tok in enumerate(prompt_ids):
        logits, past = decode_step(tok, pos, past)
    past_len = len(prompt_ids)

    generated = []
    next_id = sample_topk(logits, temperature, top_k)
    generated.append(next_id)
    for _ in range(max_new - 1):
        if next_id == EOS_ID:
            break
        logits, past = decode_step(next_id, past_len, past)
        past_len += 1
        next_id = sample_topk(logits, temperature, top_k)
        generated.append(next_id)

    return "[" + enc.decode(generated)


def generate_with_injection(
    prompt: str,
    max_new: int = 250,
    temperature: float = 0.7,
    top_k: int = 8,
    seed: int = 42,
    inject_after: int = 60,
    inject_chars: list[str] | None = None,
) -> str:
    """Generate with forced character injection every N tokens."""
    if inject_chars is None:
        inject_chars = ["JERRY", "GEORGE", "ELAINE", "KRAMER"]
    np.random.seed(seed)
    prompt_ids = enc.encode(prompt)
    past = empty_past()
    for pos, tok in enumerate(prompt_ids):
        logits, past = decode_step(tok, pos, past)
    past_len = len(prompt_ids)

    generated_ids = []
    tokens_since_inject = 0
    char_idx = 0

    next_id = sample_topk(logits, temperature, top_k)
    generated_ids.append(next_id)

    for step in range(max_new - 1):
        if next_id == EOS_ID:
            break
        tokens_since_inject += 1

        # Check if we should inject a character turn
        if tokens_since_inject >= inject_after:
            # Inject "\n\nCHARACTER: "
            char_name = inject_chars[char_idx % len(inject_chars)]
            char_idx += 1
            inject_text = f"\n\n{char_name}: "
            inject_ids = enc.encode(inject_text)
            for inj_tok in inject_ids:
                logits, past = decode_step(inj_tok, past_len, past)
                past_len += 1
                generated_ids.append(inj_tok)
            tokens_since_inject = 0
            # Sample next token after injection
            next_id = sample_topk(logits, temperature, top_k)
            generated_ids.append(next_id)
            continue

        logits, past = decode_step(next_id, past_len, past)
        past_len += 1
        next_id = sample_topk(logits, temperature, top_k)
        generated_ids.append(next_id)

    return "[" + enc.decode(generated_ids)


# ── Post-processing ───────────────────────────────────────────────────────────


def remove_repetitions(text: str, max_repeats: int = 2) -> str:
    """Remove repeated phrases (3+ words repeated 3+ times)."""
    # Split into sentences/phrases
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen: Counter[str] = Counter()
    result = []
    for s in sentences:
        key = s.strip().lower()[:50]
        seen[key] += 1
        if seen[key] <= max_repeats:
            result.append(s)
    text = " ".join(result)

    # Also detect word-level repetition loops (e.g. "I'm a chip. I'm a chip.")
    # Find any substring of 4+ words repeated 3+ times consecutively
    text = re.sub(
        r"((?:\S+\s+){3,10}?)\1{2,}",
        r"\1",
        text,
    )
    return text


def trim_trailing(text: str) -> str:
    """Trim incomplete sentence at the end."""
    # Find the last sentence-ending punctuation
    m = list(re.finditer(r'[.!?"\)]\s', text))
    if m:
        return text[: m[-1].end()].strip()
    return text


def postprocess(text: str) -> str:
    text = text.replace("[END]", "")
    text = remove_repetitions(text)
    text = trim_trailing(text)
    return text.strip()


# ── Quality scoring ───────────────────────────────────────────────────────────


def score_output(text: str) -> dict:
    """Score output quality for best-of-N selection."""
    # Count distinct Seinfeld characters
    found_chars = set()
    for name in SEINFELD_CHARS:
        if re.search(rf"\b{name}\b", text):
            found_chars.add(name)

    # Count dialogue turns (CHARACTER: text patterns)
    turns = re.findall(r"\b[A-Z][A-Z\s]{1,20}:\s", text)

    # Repetition ratio: unique sentences / total sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    unique_ratio = len(set(s.lower() for s in sentences)) / max(len(sentences), 1)

    # Has location tag
    has_location = bool(re.search(r"\[[A-Z][^\]]+\]", text))

    # Composite score
    score = (
        len(found_chars) * 3  # characters are most important
        + len(turns) * 1  # dialogue turns
        + unique_ratio * 5  # penalize repetition
        + (2 if has_location else 0)
    )

    return {
        "score": round(score, 1),
        "n_chars": len(found_chars),
        "chars": sorted(found_chars),
        "n_turns": len(turns),
        "unique_ratio": round(unique_ratio, 2),
        "has_location": has_location,
    }


def best_of_n(topic: str, prompt_template: str, n: int = 3, **gen_kwargs) -> str:
    """Generate N outputs and return the best one."""
    best_text = ""
    best_score = -1
    for i in range(n):
        prompt = prompt_template.format(topic=topic)
        text = generate(prompt, seed=42 + i * 7, **gen_kwargs)
        text = postprocess(text)
        sc = score_output(text)
        if sc["score"] > best_score:
            best_score = sc["score"]
            best_text = text
    return best_text


# ── Main experiment ───────────────────────────────────────────────────────────

TOPICS = [
    "losing a parking spot",
    "a bad haircut",
    "shrinkage",
    "double dipping a chip",
    "an annoying laugh",
    "a terrible first date",
    "jury duty",
    "being put on hold forever",
    "a close talker",
    "returning a jacket without a receipt",
]

# Prompt variants to test
PROMPTS = {
    "baseline": "TOPIC: {topic}\n\n[",
    "with_chars": "TOPIC: {topic}\n\nCHARACTERS: JERRY, GEORGE, ELAINE, KRAMER\n\n[",
    "with_location": "TOPIC: {topic}\n\n[JERRY'S APARTMENT]\n\n",
    "scene_desc": "Write a Seinfeld scene about {topic}.\n\nTOPIC: {topic}\n\n[",
}


def run_experiment(name: str, template: str, topics: list[str]):
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Prompt: {template[:60]}...")
    print(f"{'=' * 60}")

    scores = []
    for topic in topics:
        prompt = template.format(topic=topic)
        raw = generate(prompt)
        processed = postprocess(raw)
        sc = score_output(processed)
        scores.append(sc)

        print(
            f"\n  [{topic}] score={sc['score']} "
            f"chars={sc['n_chars']} turns={sc['n_turns']} "
            f"unique={sc['unique_ratio']}"
        )
        # Show first 150 chars
        preview = processed.replace("\n", " ")[:150]
        print(f"    {preview}...")

    avg_score = np.mean([s["score"] for s in scores])
    avg_chars = np.mean([s["n_chars"] for s in scores])
    avg_turns = np.mean([s["n_turns"] for s in scores])
    avg_unique = np.mean([s["unique_ratio"] for s in scores])
    print(
        f"\n  AVERAGES: score={avg_score:.1f} "
        f"chars={avg_chars:.1f} turns={avg_turns:.1f} "
        f"unique={avg_unique:.2f}"
    )
    return scores


if __name__ == "__main__":
    print("Loading model...")

    # ── A) Test prompt variants ──
    all_results = {}
    for name, template in PROMPTS.items():
        all_results[name] = run_experiment(name, template, TOPICS)

    # ── B) Test best-of-3 with baseline prompt ──
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT: best_of_3 (baseline prompt)")
    print(f"{'=' * 60}")
    bon_scores = []
    for topic in TOPICS:
        text = best_of_n(topic, "TOPIC: {topic}\n\n[", n=3)
        sc = score_output(text)
        bon_scores.append(sc)
        print(f"\n  [{topic}] score={sc['score']} chars={sc['n_chars']} turns={sc['n_turns']}")
        preview = text.replace("\n", " ")[:150]
        print(f"    {preview}...")

    avg = np.mean([s["score"] for s in bon_scores])
    print(f"\n  AVERAGE score: {avg:.1f}")

    # ── D) Test character injection ──
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT: forced character injection (every 60 tokens)")
    print(f"{'=' * 60}")
    inj_scores = []
    for topic in TOPICS:
        prompt = f"TOPIC: {topic}\n\n["
        text = generate_with_injection(prompt, inject_after=60)
        text = postprocess(text)
        sc = score_output(text)
        inj_scores.append(sc)
        print(f"\n  [{topic}] score={sc['score']} chars={sc['n_chars']} turns={sc['n_turns']}")
        preview = text.replace("\n", " ")[:200]
        print(f"    {preview}...")

    avg = np.mean([s["score"] for s in inj_scores])
    print(f"\n  AVERAGE score: {avg:.1f}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, scores in all_results.items():
        avg_s = np.mean([s["score"] for s in scores])
        avg_c = np.mean([s["n_chars"] for s in scores])
        avg_t = np.mean([s["n_turns"] for s in scores])
        print(f"  {name:20s}: score={avg_s:5.1f}  chars={avg_c:.1f}  turns={avg_t:.1f}")
    avg_bon = np.mean([s["score"] for s in bon_scores])
    avg_bon_c = np.mean([s["n_chars"] for s in bon_scores])
    avg_bon_t = np.mean([s["n_turns"] for s in bon_scores])
    print(
        f"  {'best_of_3':20s}: score={avg_bon:5.1f}  chars={avg_bon_c:.1f}  turns={avg_bon_t:.1f}"
    )
    avg_inj = np.mean([s["score"] for s in inj_scores])
    avg_inj_c = np.mean([s["n_chars"] for s in inj_scores])
    avg_inj_t = np.mean([s["n_turns"] for s in inj_scores])
    print(
        f"  {'char_injection':20s}: score={avg_inj:5.1f}  "
        f"chars={avg_inj_c:.1f}  turns={avg_inj_t:.1f}"
    )
