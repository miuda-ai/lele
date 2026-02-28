// ─────────────────────────────────────────────
// Lele WASM ML Inference Demo
// Each model is a separate WASM module
// ─────────────────────────────────────────────

import initSenseVoice, {
    SenseVoiceEngine,
    decode_wav,
    encode_wav as sv_encode_wav,
} from './pkg/sensevoice-wasm/sensevoice_wasm.js';

import initYolo26, {
    Yolo26Engine,
} from './pkg/yolo26-wasm/yolo26_wasm.js';

import initSupertonic, {
    SupertonicEngine,
    encode_wav,
} from './pkg/supertonic-wasm/supertonic_wasm.js';

// ── State ──
let asrEngine = null;
let yoloEngine = null;
let ttsEngine = null;
let asrAudioData = null;
let asrSampleRate = null;
let detectImageData = null;
let detectImageWidth = null;
let detectImageHeight = null;

// ── Base URL for model files (can be overridden) ──
const MODEL_BASE = './models/';

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
    });
});

// ── Helpers ──
function setStatus(id, text, cls = '') {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = 'status ' + cls;
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Stats table state ──
const modelStats = {
    sensevoice: { wasm: null, bin: null, rtf: null, rtfLabel: '' },
    supertonic: { wasm: null, bin: null, rtf: null, rtfLabel: '' },
    yolo26: { wasm: null, bin: null, rtf: null, rtfLabel: '' },
};

function updateStatsTable() {
    for (const [model, s] of Object.entries(modelStats)) {
        const row = document.getElementById(`stats-${model}`);
        if (!row) continue;
        const wasmCell = row.querySelector('.stat-wasm');
        const binCell = row.querySelector('.stat-bin');
        const rtfCell = row.querySelector('.stat-rtf');

        if (s.wasm !== null) {
            wasmCell.textContent = formatBytes(s.wasm);
            wasmCell.setAttribute('data-loaded', '');
        }
        if (s.bin !== null) {
            binCell.textContent = formatBytes(s.bin);
            binCell.setAttribute('data-loaded', '');
        }
        if (s.rtf !== null) {
            rtfCell.textContent = s.rtfLabel;
            rtfCell.setAttribute('data-live', '');
        }
    }

    // Totals
    const totalRow = document.getElementById('stats-total');
    if (totalRow) {
        const wasmTotal = Object.values(modelStats).reduce((a, s) => a + (s.wasm || 0), 0);
        const binTotal = Object.values(modelStats).reduce((a, s) => a + (s.bin || 0), 0);
        const wasmCell = totalRow.querySelector('.stat-wasm');
        const binCell = totalRow.querySelector('.stat-bin');
        if (wasmTotal > 0) { wasmCell.textContent = formatBytes(wasmTotal); wasmCell.setAttribute('data-loaded', ''); }
        if (binTotal > 0) { binCell.textContent = formatBytes(binTotal); binCell.setAttribute('data-loaded', ''); }
    }
}

async function fetchModel(name) {
    const resp = await fetch(MODEL_BASE + name);
    if (!resp.ok) throw new Error(`Failed to fetch ${name}: ${resp.status}`);
    return new Uint8Array(await resp.arrayBuffer());
}

async function fetchText(name) {
    const resp = await fetch(MODEL_BASE + name);
    if (!resp.ok) throw new Error(`Failed to fetch ${name}: ${resp.status}`);
    return await resp.text();
}

// ─────────────────────────────────────────────
// ASR (SenseVoice)
// ─────────────────────────────────────────────
async function initASR() {
    try {
        setStatus('asr-status', '⏳ Downloading model weights...');
        const weights = await fetchModel('sensevoice_weights.bin');
        const tokens = await fetchText('sensevoice.int8.tokens.txt');

        modelStats.sensevoice.bin = weights.length;
        updateStatsTable();
        document.getElementById('asr-model-size').textContent = formatBytes(weights.length);
        setStatus('asr-status', '⏳ Initializing model...');

        asrEngine = new SenseVoiceEngine(weights, tokens);

        setStatus('asr-status', '✅ Model ready', 'ready');
        document.getElementById('asr-file').disabled = false;
        document.getElementById('asr-upload-label').style.opacity = '1';
    } catch (e) {
        setStatus('asr-status', '❌ ' + e.message, 'error');
        console.error('ASR init error:', e);
    }
}

document.getElementById('asr-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    document.getElementById('asr-filename').textContent = file.name;

    // Show audio preview
    const url = URL.createObjectURL(file);
    const audioEl = document.getElementById('asr-audio');
    audioEl.src = url;
    document.getElementById('asr-audio-preview').style.display = 'block';

    // Decode audio
    try {
        const bytes = new Uint8Array(await file.arrayBuffer());

        if (file.name.endsWith('.wav')) {
            // Use our WAV decoder
            const resultJson = decode_wav(bytes);
            const result = JSON.parse(resultJson);
            asrAudioData = new Float32Array(result.samples);
            asrSampleRate = result.sampleRate;
        } else {
            // Use Web Audio API for other formats
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer.slice(0));
            asrAudioData = audioBuffer.getChannelData(0);
            asrSampleRate = audioBuffer.sampleRate;
            audioCtx.close();
        }

        document.getElementById('asr-run').disabled = false;
        setStatus('asr-status', `✅ Audio loaded: ${asrAudioData.length} samples @ ${asrSampleRate}Hz`, 'ready');
    } catch (err) {
        setStatus('asr-status', '❌ Failed to decode audio: ' + err.message, 'error');
    }
});

document.getElementById('asr-run').addEventListener('click', () => {
    if (!asrEngine || !asrAudioData) return;

    const btn = document.getElementById('asr-run');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Recognizing...';

    // Use setTimeout to let UI update
    setTimeout(() => {
        try {
            const start = performance.now();
            const text = asrEngine.recognize(asrAudioData, asrSampleRate);
            const elapsed = performance.now() - start;
            const audioDur = asrAudioData.length / asrSampleRate;
            const rtf = elapsed / 1000 / audioDur;

            modelStats.sensevoice.rtf = rtf;
            modelStats.sensevoice.rtfLabel = rtf.toFixed(3) + 'x';
            updateStatsTable();

            document.getElementById('asr-text').textContent = text || '(empty transcription)';
            document.getElementById('asr-timing').textContent =
                `Inference: ${elapsed.toFixed(0)}ms | Audio: ${audioDur.toFixed(1)}s | RTF: ${rtf.toFixed(3)}`;
            document.getElementById('asr-result').style.display = 'block';
        } catch (err) {
            setStatus('asr-status', '❌ Recognition failed: ' + err.message, 'error');
            console.error(err);
        }

        btn.disabled = false;
        btn.innerHTML = '▶ Recognize Speech';
    }, 50);
});

// ─────────────────────────────────────────────
// TTS (Supertonic)
// ─────────────────────────────────────────────
async function initTTS() {
    try {
        setStatus('tts-status', '⏳ Downloading models (4 models)...');

        const [teWeights, dpWeights, veWeights, voWeights, configJson, indexerJson, styleJson] =
            await Promise.all([
                fetchModel('textencoder_weights.bin'),
                fetchModel('durationpredictor_weights.bin'),
                fetchModel('vectorestimator_weights.bin'),
                fetchModel('vocoder_weights.bin'),
                fetchText('tts.json'),
                fetchText('unicode_indexer.json'),
                fetchText('M1.json'),
            ]);

        const totalSize = teWeights.length + dpWeights.length + veWeights.length + voWeights.length;
        modelStats.supertonic.bin = totalSize;
        updateStatsTable();
        document.getElementById('tts-model-size').textContent = formatBytes(totalSize);
        setStatus('tts-status', '⏳ Initializing models...');

        ttsEngine = new SupertonicEngine(
            teWeights, dpWeights, veWeights, voWeights,
            configJson, indexerJson
        );
        ttsEngine.load_style('M1', styleJson);

        setStatus('tts-status', '✅ Models ready', 'ready');
        document.getElementById('tts-text').disabled = false;
        document.getElementById('tts-speed').disabled = false;
        document.getElementById('tts-steps').disabled = false;
        document.getElementById('tts-run').disabled = false;
    } catch (e) {
        setStatus('tts-status', '❌ ' + e.message, 'error');
        console.error('TTS init error:', e);
    }
}

document.getElementById('tts-speed').addEventListener('input', (e) => {
    document.getElementById('tts-speed-val').textContent = e.target.value + 'x';
});

document.getElementById('tts-run').addEventListener('click', () => {
    if (!ttsEngine) return;

    const text = document.getElementById('tts-text').value.trim();
    if (!text) return;

    const speed = parseFloat(document.getElementById('tts-speed').value);
    const steps = parseInt(document.getElementById('tts-steps').value);

    const btn = document.getElementById('tts-run');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Synthesizing...';

    setTimeout(() => {
        try {
            const start = performance.now();
            const samples = ttsEngine.synthesize(text, 'en', 'M1', speed, steps);
            const elapsed = performance.now() - start;

            const sampleRate = ttsEngine.sample_rate();
            const audioDur = samples.length / sampleRate;
            const rtf = elapsed / 1000 / audioDur;
            const wavBytes = encode_wav(samples, sampleRate);

            modelStats.supertonic.rtf = rtf;
            modelStats.supertonic.rtfLabel = rtf.toFixed(3) + 'x';
            updateStatsTable();

            // Create audio URL
            const blob = new Blob([wavBytes], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);

            const audioEl = document.getElementById('tts-audio');
            audioEl.src = url;

            const downloadLink = document.getElementById('tts-download');
            downloadLink.href = url;
            downloadLink.style.display = 'inline-block';

            document.getElementById('tts-timing').textContent =
                `Synthesis: ${elapsed.toFixed(0)}ms | Audio: ${audioDur.toFixed(1)}s @ ${sampleRate}Hz | RTF: ${rtf.toFixed(3)}`;
            document.getElementById('tts-result').style.display = 'block';
        } catch (err) {
            setStatus('tts-status', '❌ Synthesis failed: ' + err.message, 'error');
            console.error(err);
        }

        btn.disabled = false;
        btn.innerHTML = '▶ Synthesize Speech';
    }, 50);
});

// ─────────────────────────────────────────────
// Object Detection (YOLO26)
// ─────────────────────────────────────────────
async function initDetection() {
    try {
        setStatus('detect-status', '⏳ Downloading model weights...');
        const weights = await fetchModel('yolo26_weights.bin');

        modelStats.yolo26.bin = weights.length;
        updateStatsTable();
        document.getElementById('detect-model-size').textContent = formatBytes(weights.length);
        setStatus('detect-status', '⏳ Initializing model...');

        yoloEngine = new Yolo26Engine(weights);

        setStatus('detect-status', '✅ Model ready', 'ready');
        document.getElementById('detect-file').disabled = false;
        document.getElementById('detect-upload-label').style.opacity = '1';
    } catch (e) {
        setStatus('detect-status', '❌ ' + e.message, 'error');
        console.error('Detection init error:', e);
    }
}

document.getElementById('detect-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    document.getElementById('detect-filename').textContent = file.name;

    // Load image onto canvas to get RGBA data
    const img = new Image();
    img.onload = () => {
        const canvas = document.getElementById('detect-canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        detectImageData = ctx.getImageData(0, 0, img.width, img.height).data;
        detectImageWidth = img.width;
        detectImageHeight = img.height;

        document.getElementById('detect-run').disabled = false;
        document.getElementById('detect-result').style.display = 'block';
        document.getElementById('detect-list').innerHTML = '';
        document.getElementById('detect-timing').textContent = '';
        setStatus('detect-status', `✅ Image loaded: ${img.width}×${img.height}`, 'ready');
    };
    img.src = URL.createObjectURL(file);
});

document.getElementById('detect-run').addEventListener('click', () => {
    if (!yoloEngine || !detectImageData) return;

    const btn = document.getElementById('detect-run');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Detecting...';

    setTimeout(() => {
        try {
            const start = performance.now();
            const resultJson = yoloEngine.detect(
                detectImageData,
                detectImageWidth,
                detectImageHeight,
                0.3
            );
            const elapsed = performance.now() - start;

            const detections = JSON.parse(resultJson);

            // Draw detections on canvas
            const canvas = document.getElementById('detect-canvas');
            const ctx = canvas.getContext('2d');

            // Redraw image first (clear previous detections)
            const file = document.getElementById('detect-file').files[0];
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0);
                drawDetections(ctx, detections);
            };
            img.src = URL.createObjectURL(file);

            // Generate detection list
            const listEl = document.getElementById('detect-list');
            if (detections.length === 0) {
                listEl.innerHTML = '<div class="detection-item"><span style="color:var(--text-dim)">No objects detected above threshold.</span></div>';
            } else {
                listEl.innerHTML = detections.map((d, i) =>
                    `<div class="detection-item">
                        <span class="class-name">${i + 1}. ${d.class_name}</span>
                        <span class="score">${(d.score * 100).toFixed(1)}%</span>
                        <span class="bbox">[${d.bbox.map(v => v.toFixed(0)).join(', ')}]</span>
                    </div>`
                ).join('');
            }

            const fps = 1000 / elapsed;
            modelStats.yolo26.rtf = elapsed;
            modelStats.yolo26.rtfLabel = `${elapsed.toFixed(0)}ms (${fps.toFixed(1)} fps)`;
            updateStatsTable();

            document.getElementById('detect-timing').textContent =
                `Inference: ${elapsed.toFixed(0)}ms | ${fps.toFixed(1)} fps | ${detections.length} detections`;
        } catch (err) {
            setStatus('detect-status', '❌ Detection failed: ' + err.message, 'error');
            console.error(err);
        }

        btn.disabled = false;
        btn.innerHTML = '▶ Detect Objects';
    }, 50);
});

function drawDetections(ctx, detections) {
    const colors = [
        '#ef4444', '#f97316', '#f59e0b', '#22c55e', '#06b6d4',
        '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#a855f7',
    ];

    detections.forEach((det, i) => {
        const color = colors[det.class_id % colors.length];
        const [x1, y1, x2, y2] = det.bbox;

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw label
        const label = `${det.class_name} ${(det.score * 100).toFixed(0)}%`;
        ctx.font = 'bold 14px sans-serif';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width + 8;
        const textHeight = 20;

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight, textWidth, textHeight);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + 4, y1 - 5);
    });
}

// ─────────────────────────────────────────────
// Initialization - each model loads its own WASM
// ─────────────────────────────────────────────
async function initAndLoadASR() {
    try {
        setStatus('asr-status', '⏳ Loading WASM module...');
        await initSenseVoice();
        console.log('SenseVoice WASM loaded');
        await initASR();
    } catch (e) {
        setStatus('asr-status', '❌ Failed to load WASM: ' + e.message, 'error');
        console.error('SenseVoice WASM error:', e);
    }
}

async function initAndLoadTTS() {
    try {
        setStatus('tts-status', '⏳ Loading WASM module...');
        await initSupertonic();
        console.log('Supertonic WASM loaded');
        await initTTS();
    } catch (e) {
        setStatus('tts-status', '❌ Failed to load WASM: ' + e.message, 'error');
        console.error('Supertonic WASM error:', e);
    }
}

async function initAndLoadDetection() {
    try {
        setStatus('detect-status', '⏳ Loading WASM module...');
        await initYolo26();
        console.log('YOLO26 WASM loaded');
        await initDetection();
    } catch (e) {
        setStatus('detect-status', '❌ Failed to load WASM: ' + e.message, 'error');
        console.error('YOLO26 WASM error:', e);
    }
}

async function main() {
    console.log('Initializing Lele WASM demo (separate modules)...');

    // Fetch WASM module sizes and populate stats table
    async function getWasmSize(url) {
        try {
            const resp = await fetch(url, { method: 'HEAD' });
            if (resp.ok) {
                const size = resp.headers.get('content-length');
                return size ? parseInt(size) : null;
            }
        } catch (_) { }
        return null;
    }

    Promise.all([
        getWasmSize('./pkg/sensevoice-wasm/sensevoice_wasm_bg.wasm'),
        getWasmSize('./pkg/supertonic-wasm/supertonic_wasm_bg.wasm'),
        getWasmSize('./pkg/yolo26-wasm/yolo26_wasm_bg.wasm'),
    ]).then(([svSize, stSize, yoloSize]) => {
        if (svSize) modelStats.sensevoice.wasm = svSize;
        if (stSize) modelStats.supertonic.wasm = stSize;
        if (yoloSize) modelStats.yolo26.wasm = yoloSize;
        updateStatsTable();
    });

    // Load all models in parallel (each loads its own WASM)
    initAndLoadASR();
    initAndLoadTTS();
    initAndLoadDetection();
}

main();
