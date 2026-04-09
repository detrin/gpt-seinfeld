export const KNOWN_CHARS = new Set([
    'JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN', 'MORTY',
    'FRANK', 'ESTELLE', 'SUSAN', 'PETERMAN', 'PUDDY', 'STEINBRENNER',
]);

function parseCharChunk(chunk, dialogue) {
    const trimmed = chunk.trim();
    if (!trimmed || trimmed === '[END]') return;

    const colonMatch = trimmed.match(/^([A-Z][A-Z\s]{1,20}):\s*(.+)/s);
    if (colonMatch && colonMatch[2]?.trim()) {
        dialogue.push({ char: colonMatch[1].trim(), text: colonMatch[2].trim() });
        return;
    }

    const nameMatch = trimmed.match(/^([A-Z]{2,20})\b(.*)/s);
    if (nameMatch && KNOWN_CHARS.has(nameMatch[1]) && nameMatch[2]?.trim()) {
        const txt = nameMatch[2].replace(/^[.\s!?,;()\\/]+/, '').trim();
        if (txt) { dialogue.push({ char: nameMatch[1], text: txt }); return; }
    }

    const unknownMatch = trimmed.match(/^([A-Z][A-Z]{1,20})\s{2,}(.+)/s);
    if (unknownMatch && unknownMatch[2]?.trim()) {
        dialogue.push({ char: unknownMatch[1].trim(), text: unknownMatch[2].trim() });
        return;
    }

    if (dialogue.length > 0) {
        dialogue[dialogue.length - 1].text += ' ' + trimmed;
    }
}

export function parseScene(raw) {
    const text = '[' + raw;
    const scene = { tag: null, dialogue: [], raw: null };

    const cleaned = text.replace(/\[END\]/g, '\n');

    const tagMatch = cleaned.match(/^\[([^\]]+)\]/);
    if (tagMatch) scene.tag = tagMatch[0];

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

    scene.dialogue = [];
    const rest = tagMatch ? cleaned.slice(tagMatch[0].length) : cleaned;
    const locParts = rest.split(/(\[[A-Z][^\]]*\])/);

    const nameAlt = [...KNOWN_CHARS].join('|');
    const charSplitter = new RegExp(
        `(?=\\b(?:${nameAlt})\\b)|(?=\\b[A-Z][A-Z\\s]{1,20}:\\s)`, 'g'
    );

    for (const part of locParts) {
        const trimmed = part.trim();
        if (!trimmed) continue;
        if (/^\[[A-Z][^\]]*\]$/.test(trimmed)) {
            scene.tag = trimmed;
            continue;
        }
        const charParts = trimmed.split(charSplitter).filter(p => p.trim());
        for (const cp of charParts) {
            parseCharChunk(cp, scene.dialogue);
        }
    }

    scene.raw = cleaned;
    return scene;
}
