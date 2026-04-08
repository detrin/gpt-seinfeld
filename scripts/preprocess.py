from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/train.jsonl")
MAIN = {"JERRY", "GEORGE", "ELAINE", "KRAMER"}
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "not",
    "no",
    "that",
    "this",
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "know",
    "just",
    "like",
    "get",
    "got",
    "going",
    "go",
    "said",
    "say",
    "yeah",
    "ok",
    "oh",
    "my",
    "me",
    "your",
    "his",
    "her",
    "our",
    "their",
    "s",
    "t",
    "don",
    "can",
    "m",
    "re",
    "ve",
    "ll",
    "d",
    "around",
    "about",
    "here",
    "there",
    # main character names — too frequent to be useful as topic keywords
    "jerry",
    "george",
    "elaine",
    "kramer",
    "seinfeld",
    "costanza",
    "benes",
    # high-frequency filler words that add no topic signal
    "well",
    "right",
    "back",
    "gonna",
    "really",
    "good",
    "scene",
    "apartment",
    "door",
    "think",
    "from",
    "want",
    "come",
    "look",
    "tell",
    "make",
    "mean",
    "take",
    "need",
    "talk",
    "wait",
    "over",
    "into",
    "then",
    "than",
    "some",
    "time",
    "very",
}

# Matches uppercase names like "JERRY:" or "JERRY SOMETHING:"
_UPPER_CHAR_RE = re.compile(r"^([A-Z][A-Z ]+):\s*(.*)")
# Matches title-case names like "Jerry:" or "George:"
_TITLE_CHAR_RE = re.compile(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*):\s*(.*)")


def strip_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text).strip()


def normalize_scene_tag(line: str) -> str | None:
    # Handle [Setting: Location] format used by many transcripts
    m = re.match(r"\[Setting:\s*([^\]]+)\]", line, re.IGNORECASE)
    if m:
        loc = re.split(r"[.,]", m.group(1))[0].strip()
        return f"[{loc.upper()}]"
    # Handle bare [LOCATION] format — must contain all-caps words to reject stage directions
    m = re.match(r"\[([^\]]+)\]", line)
    if not m:
        return None
    content = m.group(1)
    if not re.search(r"\b[A-Z]{3,}\b", content):
        return None
    location = re.split(r"[.,]", content)[0].strip()
    return f"[{location}]"


def percent_scene_tag(line: str) -> str | None:
    """Extract a scene tag from a '% description' line used by many transcripts."""
    if not line.startswith("%"):
        return None
    desc = line[1:].strip().rstrip(".")
    if len(desc) < 4:
        return None
    # 'At X, ...' / 'Outside X, ...' / 'Inside X, ...'
    m = re.match(r"^(?:At|Inside|Outside of?)\s+(.+)", desc, re.IGNORECASE)
    if m:
        loc = m.group(1).split(",")[0].strip()
    elif re.search(r"\bat\s+\S", desc, re.IGNORECASE):
        m2 = re.search(r"\bat\s+([^.,]+)", desc, re.IGNORECASE)
        loc = m2.group(1).strip() if m2 else desc.split(",")[0]
    else:
        loc = desc.split(",")[0].strip()
    loc = loc.rstrip(".").strip()
    if len(loc) < 3:
        return None
    return f"[{loc.upper()}]"


def extract_keywords(text: str, n: int = 5) -> list[str]:
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    freq = Counter(w for w in words if w not in _STOP)
    return [w for w, _ in freq.most_common(n)]


def parse_script(text: str, episode_name: str) -> list[tuple[str, list[str]]]:
    examples: list[tuple[str, list[str]]] = []
    current_tag: str | None = None
    current_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag = normalize_scene_tag(line) or percent_scene_tag(line)
        if tag:
            if current_tag and len(current_lines) >= 2:
                examples.append((current_tag, current_lines[:]))
            current_tag, current_lines = tag, []
            continue
        # Try uppercase character name match first (JERRY:, GEORGE:)
        m = _UPPER_CHAR_RE.match(line)
        if m:
            char = m.group(1).strip()
            if char in MAIN:
                dialogue = strip_parentheticals(m.group(2))
                if dialogue:
                    current_lines.append(f"{char}: {dialogue}")
            continue
        # Try title-case character name match (Jerry:, George:)
        m = _TITLE_CHAR_RE.match(line)
        if m:
            char = m.group(1).strip().upper()
            if char in MAIN:
                dialogue = strip_parentheticals(m.group(2))
                if dialogue:
                    current_lines.append(f"{char}: {dialogue}")
    if current_tag and len(current_lines) >= 2:
        examples.append((current_tag, current_lines[:]))
    return examples


def build_example(scene_tag: str, lines: list[str], episode_name: str, episode_text: str) -> str:
    kw = extract_keywords(episode_text)
    name = re.sub(r"([A-Z])", r" \1", episode_name).strip()
    topic = f"{name} — {', '.join(kw)}" if kw else name
    return f"TOPIC: {topic}\n\n{scene_tag}\n\n" + "\n\n".join(lines) + "\n\n[END]"


def process_all(raw_dir: Path = RAW_DIR, out_path: Path = OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for path in sorted(raw_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for tag, lines in parse_script(text, path.stem):
                f.write(json.dumps({"text": build_example(tag, lines, path.stem, text)}) + "\n")


if __name__ == "__main__":
    process_all()
