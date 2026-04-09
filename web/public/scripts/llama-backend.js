import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js';
import { postprocess } from './postprocess.js';
import { parseScene } from './scene-parser.js';
import {
    MAIN_CHARS, SAMPLING, findLastSpeaker,
    trimToSentenceEnd, trimToFirstTurn, shuffled, randomLocation,
} from './generation.js';

const HF_REPO = 'hermanda/Llama-3.2-3B-Seinfeld-GGUF';
const GGUF_FILE = 'seinfeld-3b-q4_k_m-00001-of-00004.gguf';

let wllama = null;

export async function loadModel(onProgress) {
    wllama = new Wllama({
        'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/src/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm':  'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/src/multi-thread/wllama.wasm',
    });

    await wllama.loadModelFromHF(HF_REPO, GGUF_FILE, {
        progressCallback: ({ loaded, total }) => {
            const pct = total > 0 ? Math.round(loaded / total * 100) : 0;
            onProgress(pct);
        },
    });
}

export async function generateScene(topic, { maxTokens = 150, minWords = 20, onToken } = {}) {
    const location = randomLocation();
    const basePrompt = `TOPIC: ${topic}\n\nCHARACTERS: JERRY, GEORGE, ELAINE, KRAMER\n\n[${location}]\n\n`;

    const streamPrefix = (prefix) => (token, piece, currentText) => {
        if (onToken) onToken(prefix + currentText);
    };

    // Round 1: first character's turn
    let fullGenerated = await wllama.createCompletion(basePrompt, {
        nPredict: maxTokens, sampling: SAMPLING,
        onNewToken: streamPrefix(''),
    });
    fullGenerated = trimToFirstTurn(fullGenerated);
    if (onToken) onToken(fullGenerated);

    // Build shuffled list of remaining characters to inject
    const firstSpeaker = findLastSpeaker(fullGenerated);
    const others = shuffled(MAIN_CHARS.filter(c => c !== firstSpeaker));

    // Rounds 2-4: inject each remaining character
    for (const nextChar of others) {
        fullGenerated += `\n\n${nextChar}: `;
        if (onToken) onToken(fullGenerated);
        const isExtended = Math.random() < 0.65;
        const turnMin = isExtended ? minWords : Math.max(3, Math.round(minWords / 4));
        const cont = await wllama.createCompletion(basePrompt + fullGenerated, {
            nPredict: maxTokens, sampling: SAMPLING,
            onNewToken: streamPrefix(fullGenerated),
        });
        fullGenerated += trimToSentenceEnd(cont, turnMin);
        if (onToken) onToken(fullGenerated);
    }

    const cleaned = postprocess(fullGenerated);
    const scene = parseScene(cleaned);
    scene.tag = `[${location}]`;
    return scene;
}

export async function unloadModel() {
    if (wllama) {
        try { await wllama.exit(); } catch (_) {}
        wllama = null;
    }
}
