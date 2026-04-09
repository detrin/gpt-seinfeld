function cleanDialogueText(str) {
    str = str.replace(/[:\[\];()\-]/g, ' ').replace(/ {2,}/g, ' ').trim();
    str = str.replace(/^[^A-Za-z]+/, '');
    str = str.replace(/\.{2}(?!\.)/g, '.');
    if (str) str = str[0].toUpperCase() + str.slice(1);
    return str;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function renderScene(scene) {
    document.getElementById('scene-tag').textContent = scene.tag ?? '';

    const lines = scene.dialogue
        .map(({ char, text }) => ({ char, text: cleanDialogueText(text) }))
        .filter(({ text }) => text.split(/\s+/).filter(w => w).length >= 3);

    if (lines.length > 0) {
        document.getElementById('dialogue').innerHTML = lines
            .map(({ char, text }) =>
                `<div class="line"><span class="char">${char}</span><span class="text">${escapeHtml(text)}</span></div>`
            )
            .join('');
    } else {
        const rawText = scene.tag ? scene.raw.slice(scene.tag.length).trim() : scene.raw;
        document.getElementById('dialogue').innerHTML =
            `<div class="line"><span class="text">${escapeHtml(rawText)}</span></div>`;
    }

    document.getElementById('output').classList.remove('hidden');
}
