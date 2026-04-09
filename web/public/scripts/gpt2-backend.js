import { pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
import { postprocess } from './postprocess.js';
import { parseScene } from './scene-parser.js';
import { randomLocation } from './generation.js';

const HF_MODEL = 'hermanda/gpt2-seinfeld';

let generator = null;

export async function loadModel(onProgress) {
    onProgress(0);
    generator = await pipeline('text-generation', HF_MODEL, {
        dtype: 'q8',
        progress_callback: (info) => {
            if (info.status === 'progress' && info.total) {
                const pct = Math.round(info.loaded / info.total * 100);
                onProgress(pct);
            }
        },
    });
    onProgress(100);
}

export async function generateScene(topic, { maxTokens = 300, onToken } = {}) {
    const location = randomLocation();
    const prompt = `TOPIC: ${topic}\n\n[`;

    let streamed = '';
    const opts = {
        max_new_tokens: maxTokens,
        temperature: 0.7,
        top_k: 8,
        do_sample: true,
    };

    if (onToken && generator.tokenizer) {
        const streamer = new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (text) => {
                streamed += text;
                onToken(streamed);
            },
        });
        opts.streamer = streamer;
    }

    const output = await generator(prompt, opts);

    // Strip the prompt prefix from generated text
    let raw = output[0].generated_text;
    if (raw.startsWith(prompt)) {
        raw = raw.slice(prompt.length);
    }

    const cleaned = postprocess(raw);
    const scene = parseScene(cleaned);
    scene.tag = `[${location}]`;
    return scene;
}

export async function unloadModel() {
    generator = null;
}
