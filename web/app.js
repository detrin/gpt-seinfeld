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
    const scene = { tag: null, dialogue: [] };
    for (const line of text.split('\n').map(l => l.trim())) {
        if (!line || line === '[END]') break;
        if (line.startsWith('[') && line.endsWith(']')) { scene.tag = line; continue; }
        const m = line.match(/^([A-Z][A-Z\s]{1,20}):\s*(.+)/);
        if (m) scene.dialogue.push({ char: m[1].trim(), text: m[2] });
    }
    return scene;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderScene(scene) {
    document.getElementById('scene-tag').textContent = scene.tag ?? '';
    document.getElementById('dialogue').innerHTML = scene.dialogue
        .map(({ char, text }) =>
            `<div class="line"><span class="char">${char}</span><span class="text">${escapeHtml(text)}</span></div>`
        )
        .join('');
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
            rawBox.textContent = generated;
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
        .map(el => `${el.querySelector('.char').textContent}: ${el.querySelector('.text').textContent}`)
        .join('\n');
    navigator.clipboard.writeText(`${tag}\n\n${lines}`);
});

loadModel();
