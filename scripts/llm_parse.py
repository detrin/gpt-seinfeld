from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from scripts.preprocess import build_example

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/train.jsonl")
MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
MAIN = {"JERRY", "GEORGE", "ELAINE", "KRAMER"}
WORKERS = 5
MAX_TOKENS = 8192
# Characters per chunk — ~10K tokens of input, leaves headroom for output
CHUNK_SIZE = 40_000

PROMPT = """\
Extract all scenes from this Seinfeld episode transcript excerpt.

For each scene output:
- "location": scene location in ALL CAPS (e.g. "MONK'S DINER", "JERRY'S APARTMENT")
- "dialogue": array of lines ONLY from JERRY, GEORGE, ELAINE, KRAMER in format "NAME: text"

Rules:
- Skip all other characters entirely
- Remove parenthetical or bracketed stage directions from dialogue text
- Merge wrapped/continuation lines into a single dialogue entry
- Include a scene only if it has 2+ lines from main characters
- If location is unclear use your best guess from context

Output a JSON array and NOTHING else. If no valid scenes, output [].

TRANSCRIPT EXCERPT:
{text}"""


def _make_client() -> anthropic.AnthropicBedrock:
    return anthropic.AnthropicBedrock(aws_profile="bedrock-api")


def _parse_json(raw: str) -> list[dict]:  # type: ignore[type-arg]
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _scenes_from_dicts(scenes: list[dict]) -> list[tuple[str, list[str]]]:  # type: ignore[type-arg]
    result = []
    for scene in scenes:
        loc = str(scene.get("location", "")).strip().upper()
        dialogue = scene.get("dialogue", [])
        if not loc or not isinstance(dialogue, list):
            continue
        filtered = [
            line
            for line in dialogue
            if isinstance(line, str) and line.split(":")[0].strip().upper() in MAIN
        ]
        if len(filtered) >= 2:
            result.append((f"[{loc}]", filtered))
    return result


def _call_api(
    client: anthropic.AnthropicBedrock, text: str
) -> tuple[list[tuple[str, list[str]]], bool]:
    """Returns (scenes, was_truncated)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
    )
    truncated = response.stop_reason == "max_tokens"
    scenes = _scenes_from_dicts(_parse_json(response.content[0].text))
    return scenes, truncated


def _extract_scenes(client: anthropic.AnthropicBedrock, text: str) -> list[tuple[str, list[str]]]:
    """Split text into chunks and call API, handling truncation recursively."""
    if len(text) <= CHUNK_SIZE:
        scenes, truncated = _call_api(client, text)
        if not truncated:
            return scenes
        # Output was truncated even on a small chunk — split in half
        mid = len(text) // 2
        return _extract_scenes(client, text[:mid]) + _extract_scenes(client, text[mid:])

    # Split large files into chunks on newline boundaries
    all_scenes: list[tuple[str, list[str]]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + CHUNK_SIZE, len(text))
        # Break on newline boundary
        if end < len(text):
            nl = text.rfind("\n", pos, end)
            if nl > pos:
                end = nl
        chunk = text[pos:end]
        scenes, truncated = _call_api(client, chunk)
        if truncated:
            # Chunk still too big — split it
            all_scenes += _extract_scenes(client, chunk[: len(chunk) // 2])
            all_scenes += _extract_scenes(client, chunk[len(chunk) // 2 :])
        else:
            all_scenes += scenes
        pos = end
    return all_scenes


def process_file(path: Path) -> tuple[str, list[tuple[str, list[str]]], str | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        client = _make_client()
        scenes = _extract_scenes(client, text)
        return path.stem, scenes, None
    except Exception as e:
        return path.stem, [], str(e)


def main() -> None:
    OUT.unlink(missing_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(RAW_DIR.glob("*.txt"))
    print(f"Processing {len(all_files)} files with {MODEL} ({WORKERS} workers)...")

    total_scenes = 0
    completed = 0

    with OUT.open("w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(process_file, p): p for p in all_files}
            for future in as_completed(futures):
                stem, scenes, error = future.result()
                completed += 1
                if error:
                    print(f"[{completed:3d}/{len(all_files)}] {stem:<45} ERROR: {error}")
                else:
                    path = futures[future]
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    for tag, lines in scenes:
                        f.write(json.dumps({"text": build_example(tag, lines, stem, text)}) + "\n")
                    total_scenes += len(scenes)
                    print(f"[{completed:3d}/{len(all_files)}] {stem:<45} {len(scenes)} scenes")

    print(f"\nDone. {total_scenes} total scenes written to {OUT}")


if __name__ == "__main__":
    main()
