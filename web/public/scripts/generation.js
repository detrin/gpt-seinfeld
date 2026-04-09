export const MAIN_CHARS = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER'];

export const LOCATIONS = [
    "JERRY'S APARTMENT", "MONK'S DINER", "THE STREET",
    "GEORGE'S APARTMENT", "ELAINE'S APARTMENT", "THE COFFEE SHOP",
    "YANKEE STADIUM", "THE SUBWAY", "A RESTAURANT",
    "KRAMER'S APARTMENT", "THE OFFICE", "A TAXI",
];

export const SAMPLING = { temp: 0.7, top_k: 8, penalty_repeat: 1.15 };

export function findLastSpeaker(text) {
    const matches = [...text.matchAll(/\b(JERRY|GEORGE|ELAINE|KRAMER)\s*:/g)];
    return matches.length > 0 ? matches[matches.length - 1][1] : null;
}

export function trimToSentenceEnd(text, minWords = 0) {
    const endPattern = /[.!?")\]]/g;
    let match;
    while ((match = endPattern.exec(text)) !== null) {
        const candidate = text.slice(0, match.index + 1);
        const wordCount = candidate.trim().split(/\s+/).filter(w => w).length;
        if (wordCount >= minWords) return candidate;
    }
    const m = text.match(/.*[.!?")\]]/s);
    return m ? m[0] : text;
}

export function trimToFirstTurn(text) {
    const charPattern = /\b(JERRY|GEORGE|ELAINE|KRAMER)\s*:/g;
    const matches = [...text.matchAll(charPattern)];
    if (matches.length >= 2) {
        return trimToSentenceEnd(text.slice(0, matches[1].index), 25);
    }
    return trimToSentenceEnd(text, 25);
}

export function shuffled(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

export function randomLocation() {
    return LOCATIONS[Math.floor(Math.random() * LOCATIONS.length)];
}
