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
    text = text.replace(/[^A-Za-z0-9 \n.,!?'":[\]]/g, '');
    text = text.replace(/!{3,}/g, '!!');
    text = text.replace(/\?{3,}/g, '??');
    text = text.replace(/\.{4,}/g, '...');
    text = text.replace(/-{3,}/g, '--');
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

export function postprocess(text) {
    text = text.replace(/\[END\]/g, '');
    text = fixCharTypos(text);
    text = normalizePunctuation(text);
    text = removeRepetitions(text);
    text = capMonologues(text);
    text = trimTrailing(text);
    return text.trim();
}
