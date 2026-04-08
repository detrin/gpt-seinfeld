from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.preprocess import extract_keywords

RAW_DIR = Path("data/raw")
TRAIN = Path("data/processed/train.jsonl")
MAX_CHARS = 512 * 4  # ~512 GPT-2 tokens at ~4 chars/token
MIN_DIALOGUE = 3
_CHAR_RE = re.compile(r"^(JERRY|GEORGE|ELAINE|KRAMER):")


def _stem_to_display(stem: str) -> str:
    return re.sub(r"([A-Z])", r" \1", stem).strip()


def _build_topic(display_name: str, raw_text: str) -> str:
    kw = extract_keywords(raw_text)
    return f"TOPIC: {display_name} — {', '.join(kw)}" if kw else f"TOPIC: {display_name}"


def _assemble(topic: str, scene_tag: str, dialogue: list[str]) -> str:
    return f"{topic}\n\n{scene_tag}\n\n" + "\n\n".join(dialogue) + "\n\n[END]"


def _split_to_fit(topic: str, scene_tag: str, dialogue: list[str]) -> list[str]:
    """Split dialogue into chunks that each fit within MAX_CHARS."""
    footer = "\n\n[END]"
    header = f"{topic}\n\n{scene_tag}\n\n"
    budget = MAX_CHARS - len(header) - len(footer)

    results: list[str] = []
    current: list[str] = []
    current_chars = 0

    for entry in dialogue:
        cost = len(entry) + 2  # +2 for \n\n separator
        if current and current_chars + cost > budget:
            if len([e for e in current if _CHAR_RE.match(e)]) >= MIN_DIALOGUE:
                results.append(_assemble(topic, scene_tag, current))
            current = [entry]
            current_chars = cost
        else:
            current.append(entry)
            current_chars += cost

    if len([e for e in current if _CHAR_RE.match(e)]) >= MIN_DIALOGUE:
        results.append(_assemble(topic, scene_tag, current))

    return results


def main() -> None:
    # Build display_name -> raw_text lookup
    name_to_text: dict[str, str] = {
        _stem_to_display(p.stem): p.read_text(encoding="utf-8", errors="ignore")
        for p in RAW_DIR.glob("*.txt")
    }

    raw_lines = TRAIN.read_text().splitlines()
    examples = [json.loads(raw)["text"] for raw in raw_lines]

    cleaned: list[str] = []
    dropped_thin = 0
    split_count = 0
    topic_regen = 0

    for text in examples:
        parts = text.split("\n\n")
        if len(parts) < 3:
            dropped_thin += 1
            continue

        topic_line = parts[0]
        scene_tag = parts[1]
        dialogue = [p for p in parts[2:] if p.strip() and p.strip() != "[END]"]
        char_lines = [entry for entry in dialogue if _CHAR_RE.match(entry)]

        # Drop thin examples
        if len(char_lines) < MIN_DIALOGUE:
            dropped_thin += 1
            continue

        # Regenerate TOPIC with updated _STOP
        display_name = topic_line.removeprefix("TOPIC: ").split(" — ")[0].strip()
        raw_text = name_to_text.get(display_name, "")
        if raw_text:
            new_topic = _build_topic(display_name, raw_text)
            if new_topic != topic_line:
                topic_regen += 1
                topic_line = new_topic

        # Split if too long
        full = _assemble(topic_line, scene_tag, dialogue)
        if len(full) > MAX_CHARS:
            chunks = _split_to_fit(topic_line, scene_tag, dialogue)
            cleaned.extend(chunks)
            if len(chunks) > 1:
                split_count += 1
        else:
            cleaned.append(full)

    with TRAIN.open("w", encoding="utf-8") as f:
        for ex in cleaned:
            f.write(json.dumps({"text": ex}) + "\n")

    print(f"Input:              {len(examples):>5} examples")
    print(f"Dropped (thin):   - {dropped_thin:>5}")
    print(f"Split (long):     + {split_count:>5} extra")
    print(f"Topics regenerated: {topic_regen:>5}")
    print(f"Output:             {len(cleaned):>5} examples")

    token_ests = [len(json.loads(raw)["text"]) // 4 for raw in TRAIN.read_text().splitlines()]
    over = sum(1 for t in token_ests if t > 512)
    print(f"Still >512 tokens:  {over} (should be 0)")


if __name__ == "__main__":
    main()
