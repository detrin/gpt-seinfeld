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

// Post-processing

const CHAR_TYPOS = {
    'JERREY': 'JERRY', 'JERRRY': 'JERRY', 'JERR': 'JERRY', 'JERY': 'JERRY',
    'GEROGE': 'GEORGE', 'GOERGE': 'GEORGE', 'GEOGE': 'GEORGE', 'GORGE': 'GEORGE',
    'ELIANE': 'ELAINE', 'EALINE': 'ELAINE', 'ELANE': 'ELAINE',
    'KARMER': 'KRAMER', 'KRAMR': 'KRAMER', 'KRAMAR': 'KRAMER', 'KRAMRE': 'KRAMER',
    'NEWMAM': 'NEWMAN', 'NEWMANN': 'NEWMAN',
};

function fixCharTypos(text) {
    for (const [typo, correct] of Object.entries(CHAR_TYPOS)) {
        text = text.replace(new RegExp(`\\b${typo}\\b`, 'g'), correct);
    }
    return text;
}

function normalizePunctuation(text) {
    text = text.replace(/!{3,}/g, '!!');
    text = text.replace(/\?{3,}/g, '??');
    text = text.replace(/\.{4,}/g, '...');
    text = text.replace(/ {2,}/g, ' ');
    text = text.replace(/([.!?,;])([A-Za-z])/g, '$1 $2');
    return text;
}

function removeRepetitions(text) {
    const sentences = text.split(/(?<=[.!?])\s+/);
    const seen = new Map();
    const result = [];
    for (const s of sentences) {
        const key = s.trim().toLowerCase().slice(0, 50);
        const count = (seen.get(key) || 0) + 1;
        seen.set(key, count);
        if (count <= 2) result.push(s);
    }
    text = result.join(' ');
    text = text.replace(/((?:\S+\s+){3,10}?)\1{2,}/g, '$1');
    return text;
}

function trimTrailing(text) {
    const m = text.match(/.*[.!?")\]]/s);
    return m ? m[0].trim() : text;
}

function capMonologues(text, maxWords = 80) {
    return text.replace(
        /\b([A-Z]{2,20}\s*:)\s*([\s\S]*?)(?=\b[A-Z]{2,20}\s*:|$)/g,
        (_, name, body) => {
            const words = body.trim().split(/\s+/);
            if (words.length <= maxWords) return `${name} ${body}`;
            const trimmed = words.slice(0, maxWords).join(' ');
            const lastEnd = trimmed.search(/[.!?][^.!?]*$/);
            const cut = lastEnd > 0 ? trimmed.slice(0, lastEnd + 1) : trimmed + '...';
            return `${name} ${cut}`;
        }
    );
}

function postprocess(text) {
    text = text.replace(/\[END\]/g, '');
    text = fixCharTypos(text);
    text = normalizePunctuation(text);
    text = removeRepetitions(text);
    text = capMonologues(text);
    text = trimTrailing(text);
    return text.trim();
}

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

const MAIN_CHARS = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER'];

const GEN_OPTS = { do_sample: true, temperature: 0.7, top_k: 8, repetition_penalty: 1.15 };

function findLastSpeaker(text) {
    const matches = [...text.matchAll(/\b(JERRY|GEORGE|ELAINE|KRAMER)\s*:/g)];
    return matches.length > 0 ? matches[matches.length - 1][1] : null;
}

function trimToSentenceEnd(text) {
    const m = text.match(/.*[.!?")\]]/s);
    return m ? m[0] : text;
}

function pickNextChar(lastSpeaker, roundIndex) {
    const others = MAIN_CHARS.filter(c => c !== lastSpeaker);
    return others[roundIndex % others.length];
}

async function generateRound(prompt, maxTokens) {
    let text = '';
    await generator(prompt, {
        max_new_tokens: maxTokens,
        ...GEN_OPTS,
        streamer: new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            callback_function: (token) => { text += token; },
        }),
    });
    return text;
}

async function generate() {
    const topic = document.getElementById('topic-input').value.trim();
    if (!topic || !generator) return;

    const btn = document.getElementById('generate-btn');
    const regen = document.getElementById('regen-btn');
    btn.disabled = true;
    regen.disabled = true;
    btn.textContent = 'Generating…';

    const genIndicator = document.getElementById('generating');
    genIndicator.classList.remove('hidden');
    document.getElementById('output').classList.remove('hidden');
    document.getElementById('dialogue').innerHTML = '';
    document.getElementById('scene-tag').textContent = '';

    const basePrompt = `TOPIC: ${topic}\n\nCHARACTERS: JERRY, GEORGE, ELAINE, KRAMER\n\n[`;
    let fullGenerated = '';

    try {
        // Round 1: generate location + first character's turn
        fullGenerated = await generateRound(basePrompt, 80);
        fullGenerated = trimToSentenceEnd(fullGenerated);

        // Rounds 2-5: trim after each turn, inject next character
        for (let round = 0; round < 4; round++) {
            const lastSpeaker = findLastSpeaker(fullGenerated);
            const nextChar = pickNextChar(lastSpeaker, round);

            fullGenerated += `\n\n${nextChar}: `;
            const cont = await generateRound(basePrompt + fullGenerated, 60);
            fullGenerated += trimToSentenceEnd(cont);
        }

        genIndicator.classList.add('hidden');
        const cleaned = postprocess(fullGenerated);
        renderScene(parseScene(cleaned));
    } catch (err) {
        console.error('Generation error:', err);
        genIndicator.textContent = `Error: ${err.message}`;
    } finally {
        genIndicator.classList.add('hidden');
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
