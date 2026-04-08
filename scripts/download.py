from __future__ import annotations

import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "http://seinfeldscripts.com"
INDEX = f"{BASE}/seinfeld-scripts.html"
RAW_DIR = Path("data/raw")
_EXCLUDED = {"seinfeld-scripts.html", "index.html", "index.htm"}


def fetch_script_urls(index_url: str = INDEX) -> list[str]:
    r = requests.get(index_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if (
            href.endswith((".html", ".htm"))
            and not href.startswith("http")
            and href not in _EXCLUDED
            and href not in seen
        ):
            seen.add(href)
            urls.append(f"{BASE}/{href.lstrip('/')}")
    return urls


def fetch_script_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")


def download_all(output_dir: Path = RAW_DIR, delay: float = 1.0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for url in fetch_script_urls():
        name = url.rsplit("/", 1)[-1].removesuffix(".html").removesuffix(".htm")
        path = output_dir / f"{name}.txt"
        if path.exists():
            continue
        path.write_text(fetch_script_text(url), encoding="utf-8")
        time.sleep(delay)


if __name__ == "__main__":
    download_all()
