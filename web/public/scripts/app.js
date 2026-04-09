import * as llamaBackend from './llama-backend.js';
import * as gpt2Backend from './gpt2-backend.js';
import { renderScene } from './renderer.js';

const MODEL_INFO = {
    llama: '~1.9 GB download \u00b7 better quality \u00b7 1-4 min generation',
    gpt2: '~700 MB download \u00b7 faster \u00b7 15-30 sec generation',
};

const backends = { llama: llamaBackend, gpt2: gpt2Backend };
let activeKey = null;
let activeBackend = null;
let modelReady = false;

// Settings bounds
const TOKENS_MIN = 50, TOKENS_MAX = 500;
const MINWORDS_MIN = 3, MINWORDS_MAX = 100;

// DOM refs
const btn = document.getElementById('generate-btn');
const regen = document.getElementById('regen-btn');
const statusBox = document.getElementById('model-status');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const modelInfo = document.getElementById('model-info');
const genIndicator = document.getElementById('generating');
const tokensInput = document.getElementById('tokens-input');
const minwordsInput = document.getElementById('minwords-input');
const tokensError = document.getElementById('tokens-error');
const minwordsError = document.getElementById('minwords-error');

function validateField(input, errorEl, min, max, label) {
    const val = parseInt(input.value, 10);
    if (isNaN(val) || val < min || val > max) {
        const msg = isNaN(val) ? `${label} is required` : `Must be ${min}\u2013${max}`;
        errorEl.textContent = msg;
        errorEl.classList.remove('hidden');
        input.classList.add('invalid');
        return null;
    }
    errorEl.classList.add('hidden');
    input.classList.remove('invalid');
    return val;
}

function getSettings() {
    const tokens = validateField(tokensInput, tokensError, TOKENS_MIN, TOKENS_MAX, 'Tokens');
    const minWords = validateField(minwordsInput, minwordsError, MINWORDS_MIN, MINWORDS_MAX, 'Min words');
    if (tokens === null || minWords === null) return null;
    return { maxTokens: tokens, minWords };
}

// Validate on input
tokensInput.addEventListener('input', () => validateField(tokensInput, tokensError, TOKENS_MIN, TOKENS_MAX, 'Tokens'));
minwordsInput.addEventListener('input', () => validateField(minwordsInput, minwordsError, MINWORDS_MIN, MINWORDS_MAX, 'Min words'));

function onProgress(pct) {
    progressFill.style.width = `${pct}%`;
    statusText.textContent = `Loading model\u2026 ${pct}%`;
}

async function switchModel(key) {
    if (key === activeKey && modelReady) return;

    modelReady = false;
    btn.disabled = true;

    // Unload previous
    if (activeBackend) {
        try { await activeBackend.unloadModel(); } catch (_) {}
    }

    activeKey = key;
    activeBackend = backends[key];
    modelInfo.textContent = MODEL_INFO[key];

    // Show progress
    statusBox.classList.remove('hidden');
    progressFill.style.width = '0%';
    statusText.textContent = 'Loading model\u2026 0%';

    try {
        await activeBackend.loadModel(onProgress);
        modelReady = true;
        statusBox.classList.add('hidden');
        btn.disabled = false;
        btn.classList.add('ready');
        btn.addEventListener('animationend', () => btn.classList.remove('ready'), { once: true });
    } catch (err) {
        statusText.textContent = `Failed to load model: ${err.message}`;
        console.error('Model load error:', err);
    }
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function generate() {
    const topic = document.getElementById('topic-input').value.trim();
    if (!topic || !modelReady) return;

    const settings = getSettings();
    if (!settings) return;

    btn.disabled = true;
    regen.disabled = true;
    btn.textContent = 'Generating\u2026';
    genIndicator.classList.remove('hidden');
    const outputEl = document.getElementById('output');
    const dialogueEl = document.getElementById('dialogue');
    const streamEl = document.getElementById('stream-raw');
    outputEl.classList.remove('hidden');
    dialogueEl.innerHTML = '';
    document.getElementById('scene-tag').textContent = '';

    // Show streaming area
    streamEl.classList.remove('hidden');
    streamEl.textContent = '';

    const onToken = (text) => {
        streamEl.innerHTML = escapeHtml(text) + '<span class="stream-cursor"></span>';
        streamEl.scrollTop = streamEl.scrollHeight;
    };

    try {
        const scene = await activeBackend.generateScene(topic, { ...settings, onToken });
        // Hide stream, show parsed result
        streamEl.classList.add('hidden');
        genIndicator.classList.add('hidden');
        renderScene(scene);
    } catch (err) {
        console.error('Generation error:', err);
        genIndicator.textContent = `Error: ${err.message}`;
    } finally {
        genIndicator.classList.add('hidden');
        btn.disabled = false;
        regen.disabled = false;
        btn.textContent = 'Generate Scene \u2192';
    }
}

// Event listeners
btn.addEventListener('click', generate);
regen.addEventListener('click', generate);

function getSceneText() {
    const tag = document.getElementById('scene-tag').textContent;
    const lines = [...document.querySelectorAll('.line')]
        .map(el => {
            const char = el.querySelector('.char');
            const text = el.querySelector('.text');
            return char ? `${char.textContent}: ${text.textContent}` : text.textContent;
        })
        .join('\n');
    return `${tag}\n\n${lines}`;
}

document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(getSceneText());
});

document.getElementById('share-btn').addEventListener('click', async () => {
    const topic = document.getElementById('topic-input').value.trim();
    const sceneText = getSceneText();
    const shareData = {
        title: `Seinfeld scene: ${topic}`,
        text: sceneText,
        url: 'https://gpt-seinfeld.hermandaniel.com',
    };
    if (navigator.share) {
        try { await navigator.share(shareData); } catch (_) {}
    } else {
        navigator.clipboard.writeText(`${sceneText}\n\ngpt-seinfeld.hermandaniel.com`);
    }
});

// Model selector
for (const radio of document.querySelectorAll('input[name="model"]')) {
    radio.addEventListener('change', (e) => switchModel(e.target.value));
}

// Auto-load default model
switchModel('llama');
