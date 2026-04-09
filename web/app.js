import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0';

env.allowLocalModels = false;

let generator = null;

async function loadModel() {
    const bar = document.getElementById('progress-fill');
    const status = document.getElementById('status-text');
    document.getElementById('model-status').classList.remove('hidden');

    generator = await pipeline('text-generation', 'hermanda/gpt2-seinfeld', {
        progress_callback: ({ status: s, progress }) => {
            if (s === 'progress') {
                bar.style.width = `${Math.round(progress)}%`;
                status.textContent = `Loading model… ${Math.round(progress)}%`;
            }
        },
    }).catch(err => {
        status.textContent = `Failed to load model: ${err.message}`;
        throw err;
    });

    document.getElementById('model-status').classList.add('hidden');
    document.getElementById('generate-btn').disabled = false;
}

// Known Seinfeld character names for inline detection (no colon needed)
const KNOWN_CHARS = new Set([
    'JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN', 'MORTY',
    'FRANK', 'ESTELLE', 'SUSAN', 'PETERMAN', 'PUDDY', 'STEINBRENNER',
]);

function parseCharChunk(chunk, dialogue) {
    const trimmed = chunk.trim();
    if (!trimmed || trimmed === '[END]') return;

    // CHARACTER: text (with colon)
    const colonMatch = trimmed.match(/^([A-Z][A-Z\s]{1,20}):\s*(.+)/s);
    if (colonMatch && colonMatch[2]?.trim()) {
        dialogue.push({ char: colonMatch[1].trim(), text: colonMatch[2].trim() });
        return;
    }

    // Known character name without colon (e.g. "ELAINE... (exasperation) text")
    const nameMatch = trimmed.match(/^([A-Z]{2,20})\b(.*)/s);
    if (nameMatch && KNOWN_CHARS.has(nameMatch[1]) && nameMatch[2]?.trim()) {
        const txt = nameMatch[2].replace(/^[.\s!?,;()\\/]+/, '').trim();
        if (txt) { dialogue.push({ char: nameMatch[1], text: txt }); return; }
    }

    // Unknown character name followed by whitespace gap
    const unknownMatch = trimmed.match(/^([A-Z][A-Z]{1,20})\s{2,}(.+)/s);
    if (unknownMatch && unknownMatch[2]?.trim()) {
        dialogue.push({ char: unknownMatch[1].trim(), text: unknownMatch[2].trim() });
        return;
    }

    // Append to previous dialogue line if no character detected
    if (dialogue.length > 0) {
        dialogue[dialogue.length - 1].text += ' ' + trimmed;
    }
}

function parseScene(raw) {
    // The prompt ends with "[" so prepend it back for parsing
    const text = '[' + raw;
    const scene = { tag: null, dialogue: [], raw: null };

    // Strip [END] tokens from text before parsing
    const cleaned = text.replace(/\[END\]/g, '\n');

    // Extract first location tag (skip [END])
    const tagMatch = cleaned.match(/^\[([^\]]+)\]/);
    if (tagMatch) scene.tag = tagMatch[0];

    // Try line-by-line parsing first (model sometimes produces proper format)
    for (const line of cleaned.split('\n').map(l => l.trim())) {
        if (!line) continue;
        if (line.startsWith('[') && line.endsWith(']')) {
            scene.tag = line;
            continue;
        }
        const m = line.match(/^([A-Z][A-Z\s]{1,20}):\s*(.+)/);
        if (m) scene.dialogue.push({ char: m[1].trim(), text: m[2] });
    }
    if (scene.dialogue.length > 1) return scene;

    // Inline parsing: model often produces everything on one line.
    // Pass 1: split on [LOCATION] tags to get text chunks per location
    scene.dialogue = [];
    const rest = tagMatch ? cleaned.slice(tagMatch[0].length) : cleaned;
    const locParts = rest.split(/(\[[A-Z][^\]]*\])/);

    // Pass 2: within each chunk, split on character name patterns
    const nameAlt = [...KNOWN_CHARS].join('|');
    const charSplitter = new RegExp(
        `(?=\\b(?:${nameAlt})\\b)|(?=\\b[A-Z][A-Z\\s]{1,20}:\\s)`, 'g'
    );

    for (const part of locParts) {
        const trimmed = part.trim();
        if (!trimmed) continue;
        // Location tag — update scene tag
        if (/^\[[A-Z][^\]]*\]$/.test(trimmed)) {
            scene.tag = trimmed;
            continue;
        }
        // Split on character names and parse each chunk
        const charParts = trimmed.split(charSplitter).filter(p => p.trim());
        for (const cp of charParts) {
            parseCharChunk(cp, scene.dialogue);
        }
    }

    scene.raw = cleaned;
    return scene;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderScene(scene) {
    document.getElementById('scene-tag').textContent = scene.tag ?? '';

    if (scene.dialogue.length > 0) {
        document.getElementById('dialogue').innerHTML = scene.dialogue
            .map(({ char, text }) =>
                `<div class="line"><span class="char">${char}</span><span class="text">${escapeHtml(text)}</span></div>`
            )
            .join('');
    } else {
        // Fallback: show raw text when parsing finds no structured dialogue
        const rawText = scene.tag ? scene.raw.slice(scene.tag.length).trim() : scene.raw;
        document.getElementById('dialogue').innerHTML =
            `<div class="line"><span class="text">${escapeHtml(rawText)}</span></div>`;
    }

    document.getElementById('output').classList.remove('hidden');
}

async function generate() {
    const topic = document.getElementById('topic-input').value.trim();
    if (!topic || !generator) return;

    const btn = document.getElementById('generate-btn');
    const regen = document.getElementById('regen-btn');
    btn.disabled = true;
    regen.disabled = true;
    btn.textContent = 'Generating…';

    // Show output area with a streaming raw text box while generating
    const rawBox = document.getElementById('raw-stream');
    rawBox.textContent = '';
    document.getElementById('output').classList.remove('hidden');
    document.getElementById('dialogue').innerHTML = '';
    document.getElementById('scene-tag').textContent = '';

    const prompt = `TOPIC: ${topic}\n\n[`;
    let generated = '';

    const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        callback_function: (token) => {
            generated += token;
            rawBox.textContent = '[' + generated;
        },
    });

    try {
        await generator(prompt, {
            max_new_tokens: 250,
            do_sample: true,
            temperature: 0.7,
            top_k: 8,
            repetition_penalty: 1.15,
            streamer,
        });
        rawBox.textContent = '';
        renderScene(parseScene(generated));
    } catch (err) {
        console.error('Generation error:', err);
        rawBox.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
        regen.disabled = false;
        btn.textContent = 'Generate Scene →';
    }
}

document.getElementById('generate-btn').addEventListener('click', generate);
document.getElementById('regen-btn').addEventListener('click', generate);
document.getElementById('copy-btn').addEventListener('click', () => {
    const tag = document.getElementById('scene-tag').textContent;
    const lines = [...document.querySelectorAll('.line')]
        .map(el => {
            const char = el.querySelector('.char');
            const text = el.querySelector('.text');
            return char ? `${char.textContent}: ${text.textContent}` : text.textContent;
        })
        .join('\n');
    navigator.clipboard.writeText(`${tag}\n\n${lines}`);
});

loadModel();
