"""Bulk evaluation: generate 200 scenes on random topics, analyze quality."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tiktoken

INT8_PATH = "/tmp/gpt2-seinfeld/onnx/model_quantized.onnx"
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64
EOS_ID = 50256

MAIN_CHARS = ["JERRY", "GEORGE", "ELAINE", "KRAMER"]
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

CHAR_TYPOS = {
    "JERREY": "JERRY",
    "JERRRY": "JERRY",
    "JERR": "JERRY",
    "JERY": "JERRY",
    "GEROGE": "GEORGE",
    "GOERGE": "GEORGE",
    "GEOGE": "GEORGE",
    "GORGE": "GEORGE",
    "ELIANE": "ELAINE",
    "EALINE": "ELAINE",
    "ELANE": "ELAINE",
    "KARMER": "KRAMER",
    "KRAMR": "KRAMER",
    "KRAMAR": "KRAMER",
    "KRAMRE": "KRAMER",
    "NEWMAM": "NEWMAN",
    "NEWMANN": "NEWMAN",
}

# 200 diverse Seinfeld-appropriate topics
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
    "forgetting someone's name",
    "waiting for a table at a restaurant",
    "getting stuck in traffic",
    "a loud neighbor",
    "finding a deal on a couch",
    "a man purse",
    "being regifted",
    "the wrong phone number",
    "a puffy shirt",
    "soup nazi etiquette",
    "a marble rye",
    "accidentally stealing a pen",
    "a low talker",
    "the jerk store",
    "a contest about nothing",
    "finding a good apartment",
    "a bad tip",
    "speeding ticket excuse",
    "gym membership cancellation",
    "airplane etiquette",
    "waiting room magazine theft",
    "a suspicious mole",
    "inviting the wrong person",
    "breaking up by phone",
    "faking being sick",
    "a terrible holiday gift",
    "stolen parking space",
    "coffee shop drama",
    "lost dry cleaning",
    "wearing the same outfit twice",
    "a pony tail on a man",
    "a bad massage",
    "the wrong funeral",
    "a misunderstood voicemail",
    "tipping the doorman",
    "running into an ex",
    "pretending to be someone else",
    "a loud cell phone user",
    "borrowing money from a friend",
    "a bad stand-up set",
    "pigeons attacking",
    "buying a used car",
    "a weird smell in the apartment",
    "losing at trivial pursuit",
    "getting locked out",
    "a strange fortune cookie",
    "accidentally sending a text",
    "a nosy neighbor",
    "cheating at board games",
    "a bad dentist visit",
    "being stood up",
    "the last piece of cake",
    "a terrible roommate",
    "mistaken identity",
    "a broken elevator",
    "holding someone's mail",
    "a suspicious package",
    "the line at the DMV",
    "wearing someone else's coat",
    "a potluck disaster",
    "a wrong turn on the highway",
    "running late to a wedding",
    "accidentally insulting the host",
    "a fake phone call",
    "a stain on a new shirt",
    "arguing about nothing",
    "the best booth at the diner",
    "leaving a party early",
    "a terrible movie recommendation",
    "pet sitting gone wrong",
    "fighting over the remote",
    "a misread invitation",
    "borrowing a car",
    "meeting the parents",
    "a surprise party gone wrong",
    "cheap sunglasses",
    "losing a bet",
    "a bad mechanic",
    "the morning commute",
    "a two-face situation",
    "splitting the check",
    "a haunted apartment",
    "an unwanted guest",
    "finding money on the street",
    "a bubble boy",
    "a vacation disaster",
    "the maestro's baton",
    "a bad museum date",
    "oversleeping for work",
    "a terrible photo",
    "moving day chaos",
    "a suspicious phone call",
    "the wrong medication",
    "a broken remote control",
    "a terrible hairpiece",
    "a coffee stain emergency",
    "lost subway card",
    "a bathroom emergency",
    "cutting in line",
    "a bad blind date",
    "a terrible karaoke night",
    "a broken air conditioner",
    "a suspicious insurance claim",
    "a missing shoe",
    "a disastrous dinner party",
    "hotel room complaints",
    "a terrible cab ride",
    "recycling confusion",
    "a broken zipper",
    "an expired coupon",
    "a bad dog walker",
    "a stolen sandwich",
    "a noisy restaurant",
    "a broken watch",
    "a misheard conversation",
    "a parking garage nightmare",
    "a terrible wedding toast",
    "a broken TV antenna",
    "a suspicious fortune teller",
    "a gym locker mix-up",
    "a bad plumber",
    "a terrible opera date",
    "a broken umbrella in the rain",
    "a missing wallet",
    "being cut off in traffic",
    "a bad Chinese food order",
    "a terrible laundromat experience",
    "a missing button",
    "a suspicious pizza delivery",
    "a bad shoe shine",
    "a terrible weather forecast",
    "a broken doorbell",
    "a misunderstood hand gesture",
    "a bad barista",
    "a terrible book club meeting",
    "a broken printer",
    "a suspicious yard sale",
    "a bad tailor",
    "a terrible beach trip",
    "a broken phone screen",
    "a misunderstood compliment",
    "a bad parking valet",
    "a terrible cooking class",
    "a broken coffee maker",
    "a suspicious new neighbor",
    "a bad locksmith",
    "a terrible bowling night",
    "a broken microwave",
    "a misunderstood email",
    "a bad pet store visit",
    "a terrible camping trip",
    "a broken dishwasher",
    "a suspicious online review",
    "a bad exterminator",
    "a terrible holiday party",
    "a broken bicycle",
    "a misunderstood invitation",
    "a bad moving company",
    "a terrible speed dating event",
    "a broken blender",
    "a suspicious phone bill",
    "a bad window washer",
    "a terrible concert experience",
    "a broken alarm clock",
    "a misunderstood sarcasm",
    "a bad house painter",
    "a terrible road trip",
    "a broken vacuum cleaner",
    "a suspicious parking ticket",
    "a bad carpet cleaner",
    "a terrible magic show",
    "a broken ice maker",
    "fighting over an armrest",
    "a bad seat assignment",
    "the last donut",
    "accidentally waving at a stranger",
    "being trapped in an elevator",
    "a bad toupee",
    "a suspicious muffin",
    "the perfect comeback too late",
    "a double booking disaster",
    "finding a hair in your food",
    "accidentally calling someone the wrong name",
    "a terrible secret santa gift",
    "a broken vending machine",
    "a suspicious dry cleaner",
    "the wrong subway stop",
    "accidentally wearing a costume to a regular party",
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


def generate(
    prompt: str, max_new: int = 180, temperature: float = 0.7, top_k: int = 8, seed: int = 42
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
    return enc.decode(generated)


def generate_with_selective_injection(
    topic: str,
    seed: int = 42,
) -> str:
    """Mimic the web app pipeline: with_chars prompt + selective injection."""
    base_prompt = f"TOPIC: {topic}\n\nCHARACTERS: JERRY, GEORGE, ELAINE, KRAMER\n\n["
    full = generate(base_prompt, max_new=180, seed=seed)

    for round_i in range(2):
        speakers = find_speakers(full)
        remaining = [c for c in MAIN_CHARS if c not in speakers]
        if not remaining or len(speakers) >= 3:
            break
        next_char = remaining[0]
        full += f"\n\n{next_char}: "
        continuation = generate(
            base_prompt + full,
            max_new=80,
            seed=seed + 100 + round_i,
        )
        full += continuation

    return "[" + full


def find_speakers(text: str) -> set[str]:
    speakers = set()
    for name in MAIN_CHARS:
        if re.search(rf"\b{name}\s*:", text):
            speakers.add(name)
            continue
        if re.match(rf"\s*{name}\b", text):
            speakers.add(name)
    return speakers


# ── Post-processing (mirrors web app) ────────────────────────────────────────


def fix_char_typos(text: str) -> str:
    for typo, correct in CHAR_TYPOS.items():
        text = re.sub(rf"\b{typo}\b", correct, text)
    return text


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"!{3,}", "!!", text)
    text = re.sub(r"\?{3,}", "??", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"([.!?,;])([A-Za-z])", r"\1 \2", text)
    return text


def remove_repetitions(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen: Counter[str] = Counter()
    result = []
    for s in sentences:
        key = s.strip().lower()[:50]
        seen[key] += 1
        if seen[key] <= 2:
            result.append(s)
    text = " ".join(result)
    text = re.sub(r"((?:\S+\s+){3,10}?)\1{2,}", r"\1", text)
    return text


def trim_trailing(text: str) -> str:
    m = re.match(r'.*[.!?")\]]', text, re.DOTALL)
    return m.group(0).strip() if m else text


def postprocess(text: str) -> str:
    text = text.replace("[END]", "")
    text = fix_char_typos(text)
    text = normalize_punctuation(text)
    text = remove_repetitions(text)
    text = trim_trailing(text)
    return text.strip()


# ── Quality metrics ──────────────────────────────────────────────────────────


def score_output(text: str, topic: str) -> dict:
    # Characters
    found_chars = set()
    for name in SEINFELD_CHARS:
        if re.search(rf"\b{name}\b", text):
            found_chars.add(name)

    # Dialogue turns (NAME: pattern)
    turns = re.findall(r"\b[A-Z]{2,20}\s*:\s", text)

    # Unique sentence ratio
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
    unique_ratio = len(set(s.lower() for s in sentences)) / max(len(sentences), 1)

    # Location tag
    has_location = bool(re.search(r"\[[A-Z][^\]]+\]", text))

    # Topic relevance (any topic word in output)
    topic_words = {w.lower() for w in topic.split() if len(w) > 3}
    text_lower = text.lower()
    topic_hits = sum(1 for w in topic_words if w in text_lower)
    topic_relevance = topic_hits / max(len(topic_words), 1)

    # Garbage detection
    special_ratio = len(re.findall(r"[*=~_]{2,}", text)) / max(len(text.split()), 1)
    has_garbage = special_ratio > 0.1

    # Fragment lines (NAME: followed by < 5 words)
    fragment_lines = len(re.findall(r"\b[A-Z]{2,20}\s*:\s*(\S+\s*){0,4}$", text, re.MULTILINE))

    # Word count
    words = text.split()
    word_count = len(words)

    # Coherence proxy: does it read like prose or random tokens?
    # Check for non-English / gibberish runs
    gibberish_runs = len(re.findall(r"[^\x00-\x7F]{3,}", text))

    # Composite score
    score = (
        len(found_chars) * 3
        + len(turns) * 1
        + unique_ratio * 5
        + (2 if has_location else 0)
        + topic_relevance * 3
        - (5 if has_garbage else 0)
        - fragment_lines * 2
        - gibberish_runs * 3
    )

    return {
        "score": round(score, 1),
        "n_chars": len(found_chars),
        "chars": sorted(found_chars),
        "n_turns": len(turns),
        "unique_ratio": round(unique_ratio, 2),
        "has_location": has_location,
        "topic_relevance": round(topic_relevance, 2),
        "has_garbage": has_garbage,
        "fragment_lines": fragment_lines,
        "gibberish_runs": gibberish_runs,
        "word_count": word_count,
    }


def classify_quality(result: dict) -> str:
    """Classify output as good/okay/bad."""
    sc = result["metrics"]
    if sc["has_garbage"] or sc["gibberish_runs"] > 0:
        return "bad"
    if sc["n_chars"] >= 2 and sc["n_turns"] >= 2 and sc["unique_ratio"] >= 0.6:
        return "good"
    if sc["n_chars"] >= 1 and sc["unique_ratio"] >= 0.5:
        return "okay"
    return "bad"


def main():
    print(f"Running bulk evaluation on {len(TOPICS)} topics...")
    results = []

    for i, topic in enumerate(TOPICS):
        t0 = time.time()
        raw = generate_with_selective_injection(topic, seed=42 + i * 7)
        gen_time = time.time() - t0
        processed = postprocess(raw)
        metrics = score_output(processed, topic)

        result = {
            "topic": topic,
            "raw": raw,
            "processed": processed,
            "metrics": metrics,
            "gen_time_s": round(gen_time, 1),
        }
        result["quality"] = classify_quality(result)
        results.append(result)

        q = result["quality"]
        print(
            f"  [{i + 1:3d}/{len(TOPICS)}] {q:4s} score={metrics['score']:5.1f} "
            f"chars={metrics['n_chars']} turns={metrics['n_turns']} "
            f"t={gen_time:.1f}s  {topic}"
        )

    # ── Analysis ─────────────────────────────────────────────────────────────
    good = [r for r in results if r["quality"] == "good"]
    okay = [r for r in results if r["quality"] == "okay"]
    bad = [r for r in results if r["quality"] == "bad"]

    avg_score = np.mean([r["metrics"]["score"] for r in results])
    avg_chars = np.mean([r["metrics"]["n_chars"] for r in results])
    avg_turns = np.mean([r["metrics"]["n_turns"] for r in results])
    avg_unique = np.mean([r["metrics"]["unique_ratio"] for r in results])
    avg_time = np.mean([r["gen_time_s"] for r in results])
    pct_location = 100 * np.mean([r["metrics"]["has_location"] for r in results])
    pct_garbage = 100 * np.mean([r["metrics"]["has_garbage"] for r in results])
    avg_relevance = np.mean([r["metrics"]["topic_relevance"] for r in results])
    avg_words = np.mean([r["metrics"]["word_count"] for r in results])

    # Common issues
    low_char_count = sum(1 for r in results if r["metrics"]["n_chars"] <= 1)
    repetitive = sum(1 for r in results if r["metrics"]["unique_ratio"] < 0.5)
    no_dialogue = sum(1 for r in results if r["metrics"]["n_turns"] == 0)
    off_topic = sum(1 for r in results if r["metrics"]["topic_relevance"] == 0)
    has_fragments = sum(1 for r in results if r["metrics"]["fragment_lines"] > 0)
    has_gibberish = sum(1 for r in results if r["metrics"]["gibberish_runs"] > 0)

    # ── Write markdown report ────────────────────────────────────────────────
    report_path = Path(__file__).parent.parent / "docs" / "bulk_eval.md"
    with open(report_path, "w") as f:
        f.write("# Bulk Evaluation: 200 Scenes\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d')}\n")
        f.write("**Model**: gpt2-seinfeld int8 ONNX\n")
        f.write("**Pipeline**: with_chars prompt + selective injection + full postprocessing\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total scenes | {len(results)} |\n")
        f.write(f"| Good | {len(good)} ({100 * len(good) / len(results):.0f}%) |\n")
        f.write(f"| Okay | {len(okay)} ({100 * len(okay) / len(results):.0f}%) |\n")
        f.write(f"| Bad | {len(bad)} ({100 * len(bad) / len(results):.0f}%) |\n")
        f.write(f"| Avg score | {avg_score:.1f} |\n")
        f.write(f"| Avg characters | {avg_chars:.1f} |\n")
        f.write(f"| Avg dialogue turns | {avg_turns:.1f} |\n")
        f.write(f"| Avg unique ratio | {avg_unique:.2f} |\n")
        f.write(f"| Avg topic relevance | {avg_relevance:.2f} |\n")
        f.write(f"| Avg word count | {avg_words:.0f} |\n")
        f.write(f"| Has location tag | {pct_location:.0f}% |\n")
        f.write(f"| Has garbage chars | {pct_garbage:.0f}% |\n")
        f.write(f"| Avg gen time | {avg_time:.1f}s |\n\n")

        f.write("## Issue Breakdown\n\n")
        f.write("| Issue | Count | % |\n|-------|-------|---|\n")
        n = len(results)
        pct = lambda v: f"{100 * v / n:.0f}%"  # noqa: E731
        f.write(f"| Single char only | {low_char_count} | {pct(low_char_count)} |\n")
        f.write(f"| Repetitive | {repetitive} | {pct(repetitive)} |\n")
        f.write(f"| No dialogue | {no_dialogue} | {pct(no_dialogue)} |\n")
        f.write(f"| Off-topic | {off_topic} | {pct(off_topic)} |\n")
        f.write(f"| Fragments | {has_fragments} | {pct(has_fragments)} |\n")
        f.write(f"| Gibberish | {has_gibberish} | {pct(has_gibberish)} |\n\n")

        f.write("## Score Distribution\n\n")
        bins = [(-999, 10), (10, 20), (20, 30), (30, 40), (40, 999)]
        labels = ["< 10", "10-20", "20-30", "30-40", "> 40"]
        f.write("| Range | Count | % |\n|-------|-------|---|\n")
        for (lo, hi), label in zip(bins, labels):
            cnt = sum(1 for r in results if lo <= r["metrics"]["score"] < hi)
            f.write(f"| {label} | {cnt} | {100 * cnt / len(results):.0f}% |\n")

        f.write("\n## Sample Good Outputs\n\n")
        for r in sorted(good, key=lambda x: -x["metrics"]["score"])[:5]:
            f.write(f"### {r['topic']} (score={r['metrics']['score']})\n")
            f.write(f"```\n{r['processed'][:500]}\n```\n\n")

        f.write("## Sample Bad Outputs\n\n")
        for r in sorted(bad, key=lambda x: x["metrics"]["score"])[:5]:
            f.write(f"### {r['topic']} (score={r['metrics']['score']})\n")
            f.write(f"```\n{r['processed'][:500]}\n```\n\n")

    print(f"\nReport saved to {report_path}")

    # Also save raw JSON
    json_path = Path(__file__).parent.parent / "docs" / "bulk_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {json_path}")


if __name__ == "__main__":
    main()
