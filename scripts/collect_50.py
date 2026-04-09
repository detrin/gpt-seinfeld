"""Generate 50 scenes using the web-app pipeline and save for analysis."""

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


def generate(prompt: str, max_new: int = 80, seed: int = 42) -> str:
    np.random.seed(seed)
    prompt_ids = enc.encode(prompt)
    past = empty_past()
    for pos, tok in enumerate(prompt_ids):
        logits, past = decode_step(tok, pos, past)
    past_len = len(prompt_ids)

    generated = []
    next_id = sample_topk(logits, 0.7, 8)
    generated.append(next_id)
    for _ in range(max_new - 1):
        if next_id == EOS_ID:
            break
        logits, past = decode_step(next_id, past_len, past)
        past_len += 1
        next_id = sample_topk(logits, 0.7, 8)
        generated.append(next_id)
    return enc.decode(generated)


# ── Postprocessing (mirrors web app) ─────────────────────────────────────────


def fix_char_typos(text: str) -> str:
    for typo, correct in CHAR_TYPOS.items():
        text = re.sub(rf"\b{typo}\b", correct, text)
    return text


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9 \n.,!?'\":\[\]]", "", text)
    text = re.sub(r"!{3,}", "!!", text)
    text = re.sub(r"\?{3,}", "??", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"-{3,}", "--", text)
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


def cap_monologues(text: str, max_words: int = 80) -> str:
    def _cap(m):
        name, body = m.group(1), m.group(2)
        words = body.strip().split()
        if len(words) <= max_words:
            return f"{name} {body}"
        trimmed = " ".join(words[:max_words])
        last_end = trimmed.rfind(".")
        if last_end < 0:
            last_end = trimmed.rfind("!")
        if last_end < 0:
            last_end = trimmed.rfind("?")
        cut = trimmed[: last_end + 1] if last_end > 0 else trimmed + "..."
        return f"{name} {cut}"

    return re.sub(
        r"\b([A-Z]{2,20}\s*:)\s*([\s\S]*?)(?=\b[A-Z]{2,20}\s*:|$)",
        _cap,
        text,
    )


def postprocess(text: str) -> str:
    text = text.replace("[END]", "")
    text = fix_char_typos(text)
    text = normalize_punctuation(text)
    text = remove_repetitions(text)
    text = cap_monologues(text)
    text = trim_trailing(text)
    return text.strip()


def clean_dialogue_text(text: str) -> str:
    text = re.sub(r"[:\[\];()\-]", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    text = re.sub(r"^[^A-Za-z]+", "", text)
    text = re.sub(r"\.{2}(?!\.)", ".", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


# ── Turn-by-turn generation (mirrors web app) ────────────────────────────────


def find_last_speaker(text: str) -> str | None:
    matches = list(re.finditer(r"\b(JERRY|GEORGE|ELAINE|KRAMER)\s*:", text))
    return matches[-1].group(1) if matches else None


def trim_to_sentence_end(text: str, min_words: int = 0) -> str:
    """Trim at sentence end, but only after min_words have been reached."""
    for m in re.finditer(r'[.!?")\]]', text):
        candidate = text[: m.end()]
        word_count = len(candidate.strip().split())
        if word_count >= min_words:
            return candidate
    # No sentence end found past min_words — return best effort
    fallback = re.match(r'.*[.!?")\]]', text, re.DOTALL)
    return fallback.group(0) if fallback else text


def trim_to_first_turn(text: str) -> str:
    """Keep location tag + first character's turn only, always extended."""
    matches = list(re.finditer(r"\b(JERRY|GEORGE|ELAINE|KRAMER)\s*:", text))
    if len(matches) >= 2:
        return trim_to_sentence_end(text[: matches[1].start()], min_words=25)
    return trim_to_sentence_end(text, min_words=25)


LOCATIONS = [
    "JERRY'S APARTMENT",
    "MONK'S DINER",
    "THE STREET",
    "GEORGE'S APARTMENT",
    "ELAINE'S APARTMENT",
    "THE COFFEE SHOP",
    "YANKEE STADIUM",
    "THE SUBWAY",
    "A RESTAURANT",
    "KRAMER'S APARTMENT",
    "THE OFFICE",
    "A TAXI",
]


def generate_scene(topic: str, seed: int = 42) -> dict:
    np.random.seed(seed + 999)
    location = LOCATIONS[np.random.randint(len(LOCATIONS))]
    base_prompt = f"TOPIC: {topic}\n\nCHARACTERS: JERRY, GEORGE, ELAINE, KRAMER\n\n[{location}]\n\n"

    # Round 1: first character's turn only
    full = generate(base_prompt, max_new=100, seed=seed)
    full = trim_to_first_turn(full)

    # Shuffle remaining characters for injection
    first_speaker = find_last_speaker(full)
    others = [c for c in MAIN_CHARS if c != first_speaker]
    np.random.seed(seed)
    np.random.shuffle(others)

    # Rounds 2-4: inject each remaining character with variable length
    for rnd, nxt in enumerate(others):
        full += f"\n\n{nxt}: "
        is_extended = np.random.random() < 0.65
        min_words = 25 if is_extended else 5
        cont = generate(base_prompt + full, max_new=100, seed=seed + 100 + rnd)
        full += trim_to_sentence_end(cont, min_words=min_words)

    raw = "[" + full
    processed = postprocess(raw)

    # Parse into turns for analysis
    turns = []
    for m in re.finditer(
        r"\b([A-Z]{2,20})\s*:\s*([\s\S]*?)(?=\b[A-Z]{2,20}\s*:|$)",
        processed,
    ):
        turns.append(
            {
                "char": m.group(1),
                "raw_text": m.group(2).strip(),
                "clean_text": clean_dialogue_text(m.group(2).strip()),
            }
        )

    return {
        "topic": topic,
        "raw": raw,
        "processed": processed,
        "location": f"[{location}]",
        "turns": turns,
    }


def main():
    print("Generating 50 scenes...")
    results = []

    for i, topic in enumerate(TOPICS[:50]):
        t0 = time.time()
        scene = generate_scene(topic, seed=42 + i * 7)
        elapsed = time.time() - t0
        scene["gen_time_s"] = round(elapsed, 1)
        results.append(scene)

        n_turns = len(scene["turns"])
        chars = sorted({t["char"] for t in scene["turns"]})
        loc = scene["location"] or "none"
        print(f"  [{i + 1:2d}/50] {elapsed:.1f}s turns={n_turns} chars={chars} loc={loc}")
        for t in scene["turns"]:
            preview = t["clean_text"][:80]
            print(f"    {t['char']:8s}: {preview}")

    # Save
    out = Path(__file__).parent.parent / "docs" / "collect_50.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
