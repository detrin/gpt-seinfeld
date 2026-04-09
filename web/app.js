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

function parseScene(raw) {
    // The prompt ends with "[" so prepend it back for parsing
    const text = '[' + raw;
    const scene = { tag: null, dialogue: [], raw: text };

    // Extract location tag: [ANYTHING] at start of text or on its own line
    const tagMatch = text.match(/^\[([^\]]+)\]/);
    if (tagMatch) {
        scene.tag = tagMatch[0];
    }

    // Try line-by-line parsing first (model sometimes produces proper format)
    for (const line of text.split('\n').map(l => l.trim())) {
        if (!line || line === '[END]') break;
        if (line.startsWith('[') && line.endsWith(']')) {
            scene.tag = line;
            continue;
        }
        const m = line.match(/^([A-Z][A-Z\s]{1,20}):\s*(.+)/);
        if (m) scene.dialogue.push({ char: m[1].trim(), text: m[2] });
    }

    // If no dialogue found, try splitting on character name patterns within the text
    // (model often produces prose with embedded CHARACTER: patterns)
    if (scene.dialogue.length === 0) {
        const rest = tagMatch ? text.slice(tagMatch[0].length) : text;
        const parts = rest.split(/(?=\b[A-Z][A-Z\s]{1,20}:\s)/);
        for (const part of parts) {
            const m = part.match(/^([A-Z][A-Z\s]{1,20}):\s*(.+)/s);
            if (m) scene.dialogue.push({ char: m[1].trim(), text: m[2].trim() });
        }
    }

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
            max_new_tokens: 300,
            do_sample: true,
            temperature: 0.9,
            repetition_penalty: 1.2,
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
