"""
Reannotate training data TOPICs using Claude Haiku.

Reads data/processed/train.jsonl (original — NOT modified).
Writes data/processed/train_v2.jsonl with Haiku-generated 1-sentence
scene descriptions replacing the episode-title TOPIC field.

Auth (auto-detected, in priority order):
  1. AWS Bedrock — if AWS credentials are present (supports SSO/duo-sso)
  2. Direct API  — if ANTHROPIC_API_KEY env var is set

Usage (Bedrock / SSO):
    python scripts/reannotate_topics.py

Usage (direct API key):
    ANTHROPIC_API_KEY=<key> python scripts/reannotate_topics.py

Options (env vars):
    INPUT        path to source jsonl   (default: data/processed/train.jsonl)
    OUTPUT       path to output jsonl   (default: data/processed/train_v2.jsonl)
    CONCURRENCY  parallel requests      (default: 20)
    AWS_REGION   Bedrock region         (default: us-east-1)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import anthropic

INPUT = Path(os.getenv("INPUT", "data/processed/train.jsonl"))
OUTPUT = Path(os.getenv("OUTPUT", "data/processed/train_v2.jsonl"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "20"))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Model IDs differ between Bedrock and direct API
# Bedrock requires cross-region inference profile prefix for newer models
MODEL_BEDROCK = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_DIRECT = "claude-haiku-4-5-20251001"

SYSTEM = (
    "You are a creative writing assistant. "
    "Given a Seinfeld scene, write a single short phrase (5–10 words) "
    "describing the specific social situation or awkward predicament depicted. "
    "Write it as something a user might type as a topic request — "
    'e.g. "debating who to eat in a plane crash" or '
    '"getting bumped from career day by a lizard guy". '
    "Output only the phrase, no punctuation at the end, nothing else."
)


def extract_scene_body(text: str) -> str:
    """Strip the TOPIC line; return the [LOCATION]...​[END] body."""
    lines = text.split("\n")
    # Skip lines until we hit a [LOCATION] tag
    for i, line in enumerate(lines):
        if re.match(r"^\[.+\]$", line.strip()):
            return "\n".join(lines[i:])
    return text  # fallback: return everything


def make_client() -> tuple[anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock, str]:
    """Return (async_client, model_id), preferring Bedrock if AWS creds exist."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    try:
        import boto3

        boto3.session.Session().client("sts").get_caller_identity()
        print(f"Auth: AWS Bedrock ({AWS_REGION})")
        return anthropic.AsyncAnthropicBedrock(aws_region=AWS_REGION), MODEL_BEDROCK
    except Exception:
        pass

    if api_key:
        print("Auth: direct Anthropic API key")
        return anthropic.AsyncAnthropic(api_key=api_key), MODEL_DIRECT

    sys.exit("No credentials found. Set ANTHROPIC_API_KEY or configure AWS credentials.")


async def annotate_one(
    client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock,
    model: str,
    sem: asyncio.Semaphore,
    idx: int,
    text: str,
) -> tuple[int, str]:
    scene_body = extract_scene_body(text)
    # Keep the scene short to save tokens — first 600 chars is enough context
    snippet = scene_body[:600]

    async with sem:
        msg = await client.messages.create(
            model=model,
            max_tokens=40,
            system=SYSTEM,
            messages=[{"role": "user", "content": snippet}],
        )
    topic = msg.content[0].text.strip().rstrip(".")
    return idx, topic


async def main() -> None:
    records = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} records from {INPUT}")

    client, model = make_client()
    print(f"Model: {model}")
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [annotate_one(client, model, sem, i, r["text"]) for i, r in enumerate(records)]

    results: dict[int, str] = {}
    done = 0
    for coro in asyncio.as_completed(tasks):
        idx, topic = await coro
        results[idx] = topic
        done += 1
        if done % 100 == 0 or done == len(records):
            print(f"  {done}/{len(records)} done", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        for i, record in enumerate(records):
            original_text = record["text"]
            scene_body = extract_scene_body(original_text)
            new_text = f"TOPIC: {results[i]}\n\n{scene_body}"
            f.write(json.dumps({"text": new_text}) + "\n")

    print(f"Written {len(records)} records to {OUTPUT}")

    # Spot-check: print first 5 new TOPICs
    print("\n--- Sample new TOPICs ---")
    for i in range(min(5, len(records))):
        print(f"  [{i}] {results[i]}")


if __name__ == "__main__":
    asyncio.run(main())
