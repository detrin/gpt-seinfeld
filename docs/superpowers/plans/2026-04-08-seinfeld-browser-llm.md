# Seinfeld Browser LLM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scrape Seinfeld scripts, finetune GPT-2 Medium, export to ONNX int8, and serve a client-side scene generator via a vanilla JS web app.

**Architecture:** Four sequential phases — data pipeline (download + preprocess), Colab Pro training (A100), ONNX int8 export via `optimum`, and a single-page browser app using Transformers.js for inference with no backend.

**Tech Stack:** Python 3.10+, requests, beautifulsoup4, transformers, datasets, torch, optimum[onnxruntime], pytest; JavaScript + @xenova/transformers; Google Colab Pro (A100)

---

## File Map

| File | Responsibility |
|---|---|
| `scripts/download.py` | Scrape all episode scripts from seinfeldscripts.com → `data/raw/*.txt` |
| `scripts/preprocess.py` | Clean raw scripts → `data/processed/train.jsonl` |
| `scripts/export.py` | HF checkpoint → ONNX int8 → `web/model/gpt2-seinfeld/` |
| `training/train.py` | Colab Pro training script: load dataset, tokenize, finetune gpt2-medium, save checkpoint |
| `web/index.html` | Single-page app shell: topic input, model loading progress, scene output |
| `web/app.js` | Transformers.js inference: load model, build prompt, generate, parse output |
| `tests/test_download.py` | Unit tests for download.py (HTTP mocked) |
| `tests/test_preprocess.py` | Unit tests for preprocess.py |
| `tests/test_export.py` | Unit tests for export.py (model calls mocked) |
| `.gitignore` | Exclude `model/`, `data/raw/`, `data/processed/`, `web/model/`, `.venv/` |
| `requirements.txt` | Python dependencies |

---

### Task 1: Project Setup

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: directory scaffolding

- [ ] **Step 1: Create `requirements.txt`**

```
requests==2.31.0
beautifulsoup4==4.12.3
transformers==4.40.0
datasets==2.19.0
optimum[onnxruntime]==1.19.0
torch==2.2.0
pytest==8.1.0
scikit-learn==1.4.2
```

- [ ] **Step 2: Create virtual environment and install**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/
model/
web/model/
.superpowers/
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p data/raw data/processed model/gpt2-seinfeld web/model scripts tests training
touch data/raw/.gitkeep data/processed/.gitkeep web/model/.gitkeep
touch tests/__init__.py scripts/__init__.py
```

- [ ] **Step 5: Verify**

```bash
find . -not -path './.git/*' -not -path './.venv/*' | sort
```

Expected: `data/raw/.gitkeep`, `scripts/`, `tests/__init__.py`, `training/`, `web/model/.gitkeep` all present.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt data/ scripts/ tests/ training/ web/
git commit -m "chore: project setup — structure, gitignore, requirements"
```

---

### Task 2: Script Downloader (`scripts/download.py`)

**Files:**
- Create: `scripts/download.py`
- Create: `tests/test_download.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_download.py`:

```python
from unittest.mock import patch, MagicMock
from scripts.download import fetch_script_urls, fetch_script_text

SAMPLE_INDEX = """<html><body>
<a href="TheStakeout.html">The Stakeout</a>
<a href="TheRobbery.htm">The Robbery</a>
<a href="https://external.com/other">External</a>
<a href="seinfeld-scripts.html">Home</a>
</body></html>"""

SAMPLE_SCRIPT = """<html><body>
<p>[JERRY'S APARTMENT]</p>
<p>JERRY: Hello.</p>
<p>GEORGE: Hi there.</p>
</body></html>"""


def test_fetch_script_urls_returns_episode_links():
    resp = MagicMock()
    resp.text = SAMPLE_INDEX
    with patch("scripts.download.requests.get", return_value=resp):
        urls = fetch_script_urls()
    assert "http://seinfeldscripts.com/TheStakeout.html" in urls
    assert "http://seinfeldscripts.com/TheRobbery.htm" in urls


def test_fetch_script_urls_excludes_external_and_index():
    resp = MagicMock()
    resp.text = SAMPLE_INDEX
    with patch("scripts.download.requests.get", return_value=resp):
        urls = fetch_script_urls()
    assert not any("external.com" in u for u in urls)
    assert not any("seinfeld-scripts.html" in u for u in urls)


def test_fetch_script_text_strips_html():
    resp = MagicMock()
    resp.text = SAMPLE_SCRIPT
    with patch("scripts.download.requests.get", return_value=resp):
        text = fetch_script_text("http://seinfeldscripts.com/TheStakeout.html")
    assert "JERRY" in text
    assert "<p>" not in text
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_download.py -v
```

Expected: `ImportError` — `scripts.download` does not exist yet.

- [ ] **Step 3: Implement `scripts/download.py`**

```python
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "http://seinfeldscripts.com"
INDEX = f"{BASE}/seinfeld-scripts.html"
RAW_DIR = Path("data/raw")
_EXCLUDED = {"seinfeld-scripts.html", "index.html", "index.htm"}


def fetch_script_urls(index_url=INDEX):
    r = requests.get(index_url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    seen, urls = set(), []
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


def fetch_script_text(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")


def download_all(output_dir=RAW_DIR, delay=1.0):
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_download.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Run the downloader**

```bash
source .venv/bin/activate
python scripts/download.py
```

Expected: `data/raw/` fills with ~170 `.txt` files. Verify:

```bash
ls data/raw/ | wc -l
head -30 data/raw/TheStakeout.txt
```

- [ ] **Step 6: Commit**

```bash
git add scripts/download.py tests/test_download.py
git commit -m "feat: script downloader — scrapes seinfeldscripts.com to data/raw/"
```

---

### Task 3: Script Preprocessor (`scripts/preprocess.py`)

**Files:**
- Create: `scripts/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_preprocess.py`:

```python
from scripts.preprocess import (
    strip_parentheticals,
    normalize_scene_tag,
    parse_script,
    build_example,
    extract_keywords,
)


def test_strip_parentheticals_removes_parens():
    assert strip_parentheticals("(to George) You know what?") == "You know what?"


def test_strip_parentheticals_no_parens():
    assert strip_parentheticals("I don't think so.") == "I don't think so."


def test_normalize_scene_tag_simple():
    assert normalize_scene_tag("[JERRY'S APARTMENT]") == "[JERRY'S APARTMENT]"


def test_normalize_scene_tag_strips_description():
    assert normalize_scene_tag("[JERRY'S APARTMENT. JERRY AND GEORGE TALKING.]") == "[JERRY'S APARTMENT]"


def test_normalize_scene_tag_returns_none_for_non_tag():
    assert normalize_scene_tag("JERRY: Hello.") is None


def test_parse_script_filters_non_main_characters():
    text = "[MONK'S DINER]\nJERRY: You know what I love?\nNEWMAN: Stamps.\nGEORGE: What do you love?\n"
    result = parse_script(text, "TheEpisode")
    assert len(result) == 1
    _, lines = result[0]
    assert any(l.startswith("JERRY:") for l in lines)
    assert any(l.startswith("GEORGE:") for l in lines)
    assert not any(l.startswith("NEWMAN:") for l in lines)


def test_parse_script_strips_parentheticals_from_dialogue():
    text = "[JERRY'S APARTMENT]\nJERRY: (laughing) That's insane.\nELAINE: (entering) What's going on?\nGEORGE: Unbelievable.\n"
    _, lines = parse_script(text, "TheEpisode")[0]
    assert "JERRY: That's insane." in lines
    assert "ELAINE: What's going on?" in lines


def test_build_example_has_correct_structure():
    result = build_example(
        "[MONK'S DINER]",
        ["JERRY: Hello.", "GEORGE: Hi.", "ELAINE: Hey."],
        "TheDiner",
        "jerry george diner coffee parking tickets",
    )
    assert result.startswith("TOPIC:")
    assert "[MONK'S DINER]" in result
    assert "JERRY: Hello." in result
    assert result.strip().endswith("[END]")


def test_extract_keywords_excludes_stopwords():
    text = "parking spot buick circling driving around block"
    kw = extract_keywords(text, n=3)
    assert "parking" in kw
    assert "buick" in kw
    assert "the" not in kw
    assert "around" not in kw
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_preprocess.py -v
```

Expected: `ImportError` — `scripts.preprocess` does not exist yet.

- [ ] **Step 3: Implement `scripts/preprocess.py`**

```python
import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/train.jsonl")
MAIN = {"JERRY", "GEORGE", "ELAINE", "KRAMER"}
_STOP = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "i","you","he","she","we","they","it","is","are","was","were","be",
    "been","have","has","had","do","does","did","will","would","could",
    "should","not","no","that","this","what","how","why","when","where",
    "who","know","just","like","get","got","going","go","said","say",
    "yeah","ok","oh","my","me","your","his","her","our","their",
    "s","t","don","can","m","re","ve","ll","d","around","about","here","there",
}


def strip_parentheticals(text):
    return re.sub(r"\([^)]*\)", "", text).strip()


def normalize_scene_tag(line):
    m = re.match(r"\[([^\]]+)\]", line)
    if not m:
        return None
    location = re.split(r"[.,]", m.group(1))[0].strip()
    return f"[{location}]"


def extract_keywords(text, n=5):
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    freq = Counter(w for w in words if w not in _STOP)
    return [w for w, _ in freq.most_common(n)]


def parse_script(text, episode_name):
    examples, current_tag, current_lines = [], None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag = normalize_scene_tag(line)
        if tag:
            if current_tag and len(current_lines) >= 3:
                examples.append((current_tag, current_lines[:]))
            current_tag, current_lines = tag, []
            continue
        m = re.match(r"^([A-Z][A-Z ]+):\s*(.*)", line)
        if m:
            char = m.group(1).strip()
            if char in MAIN:
                dialogue = strip_parentheticals(m.group(2))
                if dialogue:
                    current_lines.append(f"{char}: {dialogue}")
    if current_tag and len(current_lines) >= 3:
        examples.append((current_tag, current_lines[:]))
    return examples


def build_example(scene_tag, lines, episode_name, episode_text):
    kw = extract_keywords(episode_text)
    name = re.sub(r"([A-Z])", r" \1", episode_name).strip()
    topic = f"{name} — {', '.join(kw)}" if kw else name
    return f"TOPIC: {topic}\n\n{scene_tag}\n\n" + "\n\n".join(lines) + "\n\n[END]"


def process_all(raw_dir=RAW_DIR, out_path=OUT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for path in sorted(raw_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for tag, lines in parse_script(text, path.stem):
                f.write(json.dumps({"text": build_example(tag, lines, path.stem, text)}) + "\n")


if __name__ == "__main__":
    process_all()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Run the preprocessor**

```bash
python scripts/preprocess.py
wc -l data/processed/train.jsonl
head -n 1 data/processed/train.jsonl | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['text'])"
```

Expected: hundreds of lines; first entry shows a properly structured TOPIC/scene/dialogue/[END] example.

- [ ] **Step 6: Commit**

```bash
git add scripts/preprocess.py tests/test_preprocess.py
git commit -m "feat: preprocessor — raw scripts → TOPIC/scene/dialogue training examples"
```

---

### Task 4: Training Script (`training/train.py`)

**Files:**
- Create: `training/train.py`

Runs on Google Colab Pro (A100). No local unit tests — correctness verified by loss curve and sanity-check generation at the end.

- [ ] **Step 1: Create `training/train.py`**

```python
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline,
)

CHECKPOINT = os.environ.get("CHECKPOINT_DIR", "/content/drive/MyDrive/gpt2-seinfeld")
DATA_FILE = os.environ.get("DATA_FILE", "train.jsonl")

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATA_FILE, split="train")


def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
print(f"Dataset: {len(tokenized)} examples")

model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

args = TrainingArguments(
    output_dir=CHECKPOINT,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=5e-5,
    fp16=True,
    save_steps=500,
    logging_steps=50,
    report_to="none",
)

Trainer(model=model, args=args, train_dataset=tokenized).train()
model.save_pretrained(CHECKPOINT)
tokenizer.save_pretrained(CHECKPOINT)
print(f"Checkpoint saved to {CHECKPOINT}")

gen = pipeline("text-generation", model=CHECKPOINT, tokenizer=tokenizer)
out = gen("TOPIC: parking tickets\n\n[", max_new_tokens=200, do_sample=True, temperature=0.9)
print("\n--- Sanity check ---\n")
print(out[0]["generated_text"])
```

- [ ] **Step 2: Commit the training script**

```bash
git add training/train.py
git commit -m "feat: Colab training script for gpt2-medium finetune"
```

- [ ] **Step 3: Run on Colab Pro**

1. Open [colab.research.google.com](https://colab.research.google.com), create a new notebook
2. Runtime → Change runtime type → **A100 GPU**
3. Mount Drive and upload `data/processed/train.jsonl`:

```python
from google.colab import drive, files
drive.mount('/content/drive')
files.upload()  # select data/processed/train.jsonl
```

4. Upload and run `training/train.py`:

```bash
!pip install -q transformers datasets accelerate
!python train.py
```

5. After training, download `gpt2-seinfeld/` from Google Drive to `model/gpt2-seinfeld/` locally.

Expected sanity check output: coherent Seinfeld dialogue with scene tag, speaker labels, ending near `[END]`.

---

### Task 5: ONNX Export (`scripts/export.py`)

**Files:**
- Create: `scripts/export.py`
- Create: `tests/test_export.py`

Prerequisite: `model/gpt2-seinfeld/` must exist locally (from Task 4).

- [ ] **Step 1: Write failing test**

Create `tests/test_export.py`:

```python
from unittest.mock import patch, MagicMock
from pathlib import Path
from scripts.export import export


def test_export_calls_model_quantizer_and_tokenizer(tmp_path):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_quantizer = MagicMock()
    mock_qconfig = MagicMock()

    # Pre-create files that would exist after model save + quantize
    (tmp_path / "model.onnx").touch()
    (tmp_path / "model_quantized.onnx").touch()

    with (
        patch("scripts.export.ORTModelForCausalLM.from_pretrained", return_value=mock_model),
        patch("scripts.export.ORTQuantizer.from_pretrained", return_value=mock_quantizer),
        patch("scripts.export.AutoQuantizationConfig.avx2", return_value=mock_qconfig),
        patch("scripts.export.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        export(model_dir=Path("fake/model"), out_dir=tmp_path)

    mock_model.save_pretrained.assert_called_once_with(str(tmp_path))
    mock_quantizer.quantize.assert_called_once()
    mock_tokenizer.save_pretrained.assert_called_once_with(str(tmp_path))
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_export.py -v
```

Expected: `ImportError` — `scripts.export` does not exist yet.

- [ ] **Step 3: Implement `scripts/export.py`**

```python
from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_DIR = Path("model/gpt2-seinfeld")
OUT_DIR = Path("web/model/gpt2-seinfeld")


def export(model_dir=MODEL_DIR, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForCausalLM.from_pretrained(str(model_dir), export=True)
    model.save_pretrained(str(out_dir))

    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(out_dir), quantization_config=qconfig)

    (out_dir / "model.onnx").unlink(missing_ok=True)
    (out_dir / "model_quantized.onnx").rename(out_dir / "model.onnx")

    AutoTokenizer.from_pretrained(str(model_dir)).save_pretrained(str(out_dir))


if __name__ == "__main__":
    export()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_export.py -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Run the export (after Task 4 complete)**

```bash
python scripts/export.py
ls -lh web/model/gpt2-seinfeld/
```

Expected: `model.onnx` (~450MB), `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.

- [ ] **Step 6: Commit**

```bash
git add scripts/export.py tests/test_export.py
git commit -m "feat: ONNX export — gpt2-seinfeld → int8 quantized for Transformers.js"
```

---

### Task 6: Web App (`web/index.html` + `web/app.js`)

**Files:**
- Create: `web/index.html`
- Create: `web/app.js`

No unit tests — validated visually in Chrome.

- [ ] **Step 1: Create `web/app.js`**

```javascript
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = true;
env.localModelPath = './model/';

let generator = null;

async function loadModel() {
    const bar = document.getElementById('progress-fill');
    const status = document.getElementById('status-text');
    document.getElementById('model-status').classList.remove('hidden');

    generator = await pipeline('text-generation', 'gpt2-seinfeld', {
        progress_callback: ({ status: s, progress }) => {
            if (s === 'progress') {
                bar.style.width = `${Math.round(progress)}%`;
                status.textContent = `Loading model… ${Math.round(progress)}%`;
            }
        },
    });

    document.getElementById('model-status').classList.add('hidden');
    document.getElementById('generate-btn').disabled = false;
}

function parseScene(text) {
    const scene = { tag: null, dialogue: [] };
    for (const raw of text.split('\n')) {
        const line = raw.trim();
        if (!line || line === '[END]') break;
        if (line.startsWith('[')) { scene.tag = line; continue; }
        const m = line.match(/^(JERRY|GEORGE|ELAINE|KRAMER):\s*(.+)/);
        if (m) scene.dialogue.push({ char: m[1], text: m[2] });
    }
    return scene;
}

function renderScene(scene) {
    document.getElementById('scene-tag').textContent = scene.tag ?? '';
    document.getElementById('dialogue').innerHTML = scene.dialogue
        .map(({ char, text }) =>
            `<div class="line"><span class="char">${char}</span><span class="text">${text}</span></div>`
        )
        .join('');
    document.getElementById('output').classList.remove('hidden');
}

async function generate() {
    const topic = document.getElementById('topic-input').value.trim();
    if (!topic || !generator) return;

    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.textContent = 'Generating…';

    const prompt = `TOPIC: ${topic}\n\n[`;
    const result = await generator(prompt, {
        max_new_tokens: 300,
        do_sample: true,
        temperature: 0.9,
        repetition_penalty: 1.2,
    });
    renderScene(parseScene(result[0].generated_text.slice(prompt.length)));

    btn.disabled = false;
    btn.textContent = 'Generate Scene →';
}

document.getElementById('generate-btn').addEventListener('click', generate);
document.getElementById('regen-btn').addEventListener('click', generate);
document.getElementById('copy-btn').addEventListener('click', () => {
    const tag = document.getElementById('scene-tag').textContent;
    const lines = [...document.querySelectorAll('.line')]
        .map(el => `${el.querySelector('.char').textContent}: ${el.querySelector('.text').textContent}`)
        .join('\n');
    navigator.clipboard.writeText(`${tag}\n\n${lines}`);
});

loadModel();
```

- [ ] **Step 2: Create `web/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Seinfeld Scene Generator</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #fafafa; color: #111; min-height: 100vh; display: flex; justify-content: center; padding: 48px 16px; }
    #app { width: 100%; max-width: 680px; }
    .header { margin-bottom: 32px; }
    .header h1 { font-size: 1.6rem; font-weight: 700; }
    .header p { color: #666; margin-top: 4px; font-size: 0.95rem; }
    #model-status { background: #f0f7ff; border: 1px solid #c0d8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; font-size: 0.85rem; color: #336; }
    #model-status.hidden { display: none; }
    .progress-bar { background: #dde; border-radius: 4px; height: 4px; margin-top: 8px; overflow: hidden; }
    #progress-fill { background: #336; height: 100%; width: 0; transition: width 0.2s; }
    label { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em; color: #555; display: block; margin-bottom: 6px; }
    input[type="text"] { width: 100%; border: 1px solid #ddd; border-radius: 8px; padding: 12px 14px; font-size: 1rem; outline: none; transition: border-color 0.15s; }
    input[type="text"]:focus { border-color: #999; }
    #generate-btn { width: 100%; margin-top: 12px; background: #111; color: #fff; border: none; border-radius: 8px; padding: 13px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
    #generate-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    #output { margin-top: 28px; border-top: 1px solid #eee; padding-top: 24px; }
    #output.hidden { display: none; }
    .output-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; color: #aaa; margin-bottom: 14px; }
    #scene-tag { font-size: 0.85rem; color: #888; font-style: italic; margin-bottom: 16px; }
    .line { display: flex; gap: 14px; margin-bottom: 14px; }
    .char { font-size: 0.78rem; font-weight: 700; color: #555; letter-spacing: 0.05em; min-width: 64px; padding-top: 3px; }
    .text { font-size: 0.95rem; line-height: 1.6; color: #111; }
    .actions { display: flex; gap: 10px; margin-top: 20px; }
    .actions button { background: #f5f5f5; border: 1px solid #ddd; border-radius: 6px; padding: 8px 16px; font-size: 0.85rem; cursor: pointer; color: #333; }
    .actions button:hover { background: #eee; }
  </style>
</head>
<body>
  <div id="app">
    <div class="header">
      <h1>Seinfeld Scene Generator</h1>
      <p>Describe a situation. Get a scene.</p>
    </div>

    <div id="model-status" class="hidden">
      <span id="status-text">Loading model…</span>
      <div class="progress-bar"><div id="progress-fill"></div></div>
    </div>

    <div>
      <label for="topic-input">TOPIC / SITUATION</label>
      <input type="text" id="topic-input" placeholder="e.g. losing a parking spot" />
    </div>
    <button id="generate-btn" disabled>Generate Scene →</button>

    <div id="output" class="hidden">
      <div class="output-label">GENERATED SCENE</div>
      <div id="scene-tag"></div>
      <div id="dialogue"></div>
      <div class="actions">
        <button id="regen-btn">↺ Regenerate</button>
        <button id="copy-btn">Copy</button>
      </div>
    </div>
  </div>
  <script type="module" src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Validate in browser**

```bash
cd web && python3 -m http.server 8080
```

Open `http://localhost:8080` in Chrome. Verify:
- Model loading bar appears and progresses
- After load, Generate button enables
- Typing "parking tickets" and clicking Generate produces a scene with `[LOCATION]` tag and JERRY/GEORGE/ELAINE/KRAMER lines
- Regenerate produces a different scene
- Copy writes the scene to clipboard

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/app.js
git commit -m "feat: web app — Transformers.js scene generator with topic input and dialogue renderer"
```
