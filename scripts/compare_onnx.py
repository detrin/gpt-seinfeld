"""Compare fp32 vs int8 ONNX models: logit divergence, output quality, speed."""

from __future__ import annotations

import json
import re
import time

import numpy as np
import onnxruntime as ort
import tiktoken

# ── Paths ─────────────────────────────────────────────────────────────────────
FP32_PATH = "/tmp/gpt2-seinfeld/onnx/model.onnx"
INT8_PATH = "/tmp/gpt2-seinfeld/onnx/model_quantized.onnx"

# ── Architecture constants (GPT-2 medium) ────────────────────────────────────
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
]

enc = tiktoken.get_encoding("gpt2")


def load_session(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def empty_past() -> dict:
    e = np.empty((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
    return {f"past_key_values.{i}.{kv}": e for i in range(N_LAYERS) for kv in ("key", "value")}


def decode_step(session, token_id: int, past_len: int, past: dict):
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


def prefill(session, prompt_ids: list[int]):
    past = empty_past()
    for pos, tok in enumerate(prompt_ids):
        logits, past = decode_step(session, tok, pos, past)
    return logits, past, len(prompt_ids)


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
    session,
    prompt: str,
    max_new: int = 200,
    temperature: float = 0.7,
    top_k: int = 5,
    seed: int = 42,
):
    np.random.seed(seed)
    prompt_ids = enc.encode(prompt)
    logits, past, past_len = prefill(session, prompt_ids)

    generated = []
    all_logits = [logits.copy()]
    next_id = sample_topk(logits, temperature, top_k)
    generated.append(next_id)

    for _ in range(max_new - 1):
        if next_id == EOS_ID:
            break
        logits, past = decode_step(session, next_id, past_len, past)
        past_len += 1
        all_logits.append(logits.copy())
        next_id = sample_topk(logits, temperature, top_k)
        generated.append(next_id)

    return generated, all_logits


def logit_stats(logits_list: list[np.ndarray]) -> dict:
    """Summary stats for a sequence of logit vectors."""
    maxes = [v.max() for v in logits_list]
    mins = [v.min() for v in logits_list]
    ranges = [v.max() - v.min() for v in logits_list]
    return {
        "mean_max": float(np.mean(maxes)),
        "mean_min": float(np.mean(mins)),
        "mean_range": float(np.mean(ranges)),
        "std_max": float(np.std(maxes)),
    }


def score_output(text: str) -> dict:
    """Heuristic quality metrics for a generated scene."""
    seinfeld_chars = {"JERRY", "GEORGE", "ELAINE", "KRAMER", "NEWMAN", "MORTY", "FRANK", "ESTELLE"}
    found_chars = set()
    for name in seinfeld_chars:
        if name in text.upper():
            found_chars.add(name)

    has_location = bool(re.search(r"\[[A-Z].*?\]", text))
    has_end = "[END]" in text
    char_lines = len(re.findall(r"\b[A-Z][A-Z\s]{1,20}:\s", text))
    word_count = len(text.split())
    has_newlines = "\n" in text

    return {
        "seinfeld_chars": sorted(found_chars),
        "n_seinfeld_chars": len(found_chars),
        "has_location_tag": has_location,
        "has_end_token": has_end,
        "char_dialogue_lines": char_lines,
        "word_count": word_count,
        "has_newlines": has_newlines,
    }


def compare_logits(fp32_logits: list[np.ndarray], int8_logits: list[np.ndarray]) -> dict:
    """Compare logit distributions between fp32 and int8 at each step."""
    n = min(len(fp32_logits), len(int8_logits))
    diffs = []
    cosines = []
    top1_agree = 0
    top5_agree = 0

    for i in range(n):
        f, q = fp32_logits[i].astype(np.float64), int8_logits[i].astype(np.float64)
        diffs.append(float(np.mean(np.abs(f - q))))
        cos = np.dot(f, q) / (np.linalg.norm(f) * np.linalg.norm(q) + 1e-12)
        cosines.append(float(cos))
        if np.argmax(f) == np.argmax(q):
            top1_agree += 1
        f_top5 = set(np.argsort(f)[-5:])
        q_top5 = set(np.argsort(q)[-5:])
        if f_top5 == q_top5:
            top5_agree += 1

    return {
        "n_steps": n,
        "mean_abs_diff": float(np.mean(diffs)),
        "mean_cosine_sim": float(np.mean(cosines)),
        "min_cosine_sim": float(np.min(cosines)),
        "top1_agreement": top1_agree / n,
        "top5_agreement": top5_agree / n,
    }


def main():
    print("Loading fp32 model...")
    fp32_sess = load_session(FP32_PATH)
    print("Loading int8 model...")
    int8_sess = load_session(INT8_PATH)
    print()

    all_results = []

    for topic in TOPICS:
        prompt = f"TOPIC: {topic}\n\n["
        print(f"═══ Topic: {topic} ═══")

        # Generate with same seed for fair comparison
        t0 = time.time()
        fp32_ids, fp32_logits = generate(fp32_sess, prompt, seed=42)
        fp32_time = time.time() - t0

        t0 = time.time()
        int8_ids, int8_logits = generate(int8_sess, prompt, seed=42)
        int8_time = time.time() - t0

        fp32_text = "[" + enc.decode(fp32_ids)
        int8_text = "[" + enc.decode(int8_ids)

        # Logit comparison
        logit_cmp = compare_logits(fp32_logits, int8_logits)
        fp32_quality = score_output(fp32_text)
        int8_quality = score_output(int8_text)

        result = {
            "topic": topic,
            "fp32": {
                "text": fp32_text,
                "time_s": round(fp32_time, 2),
                "n_tokens": len(fp32_ids),
                "logit_stats": logit_stats(fp32_logits),
                "quality": fp32_quality,
            },
            "int8": {
                "text": int8_text,
                "time_s": round(int8_time, 2),
                "n_tokens": len(int8_ids),
                "logit_stats": logit_stats(int8_logits),
                "quality": int8_quality,
            },
            "logit_comparison": logit_cmp,
            "tokens_identical": fp32_ids == int8_ids,
        }
        all_results.append(result)

        # Print summary
        print(f"  fp32: {fp32_time:.1f}s, {len(fp32_ids)} tokens")
        print(
            f"  int8: {int8_time:.1f}s, {len(int8_ids)} tokens, {int8_time / fp32_time:.2f}x speed"
        )
        print(f"  Tokens identical: {fp32_ids == int8_ids}")
        cos_mean = logit_cmp["mean_cosine_sim"]
        cos_min = logit_cmp["min_cosine_sim"]
        print(f"  Logit cosine sim: {cos_mean:.6f} (min {cos_min:.6f})")
        print(f"  Top-1 agreement:  {logit_cmp['top1_agreement']:.1%}")
        print(f"  Top-5 agreement:  {logit_cmp['top5_agreement']:.1%}")
        print()
        nc_fp32 = fp32_quality["n_seinfeld_chars"]
        dl_fp32 = fp32_quality["char_dialogue_lines"]
        nc_int8 = int8_quality["n_seinfeld_chars"]
        dl_int8 = int8_quality["char_dialogue_lines"]
        print(f"  FP32 output ({nc_fp32} chars, {dl_fp32} lines):")
        print(f"    {fp32_text[:200]}...")
        print(f"  INT8 output ({nc_int8} chars, {dl_int8} lines):")
        print(f"    {int8_text[:200]}...")
        print()

    # Aggregate stats
    print("═══ AGGREGATE ═══")
    avg_cosine = np.mean([r["logit_comparison"]["mean_cosine_sim"] for r in all_results])
    avg_top1 = np.mean([r["logit_comparison"]["top1_agreement"] for r in all_results])
    avg_top5 = np.mean([r["logit_comparison"]["top5_agreement"] for r in all_results])
    identical = sum(1 for r in all_results if r["tokens_identical"])
    fp32_total_time = sum(r["fp32"]["time_s"] for r in all_results)
    int8_total_time = sum(r["int8"]["time_s"] for r in all_results)

    fp32_avg_chars = np.mean([r["fp32"]["quality"]["n_seinfeld_chars"] for r in all_results])
    int8_avg_chars = np.mean([r["int8"]["quality"]["n_seinfeld_chars"] for r in all_results])
    fp32_avg_lines = np.mean([r["fp32"]["quality"]["char_dialogue_lines"] for r in all_results])
    int8_avg_lines = np.mean([r["int8"]["quality"]["char_dialogue_lines"] for r in all_results])

    print(f"  Mean cosine similarity: {avg_cosine:.6f}")
    print(f"  Mean top-1 agreement:   {avg_top1:.1%}")
    print(f"  Mean top-5 agreement:   {avg_top5:.1%}")
    print(f"  Identical outputs:      {identical}/{len(all_results)}")
    print(f"  Total fp32 time:        {fp32_total_time:.1f}s")
    ratio = int8_total_time / fp32_total_time
    print(f"  Total int8 time:        {int8_total_time:.1f}s ({ratio:.2f}x)")
    print(f"  Avg seinfeld chars:     fp32={fp32_avg_chars:.1f}  int8={int8_avg_chars:.1f}")
    print(f"  Avg dialogue lines:     fp32={fp32_avg_lines:.1f}  int8={int8_avg_lines:.1f}")

    # Save raw results
    out_path = "/tmp/gpt2-seinfeld/compare_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
