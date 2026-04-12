// Lele WASM browser-side benchmark
// Runs inside bench.html, results written to window.__benchmarkResults
//
// Measurements use performance.now() (sub-ms resolution in Chrome).
// Each model is benchmarked independently:
//   - SenseVoice ASR  : recognize(3s synthetic audio) x ITERS
//   - YOLO26 Detection: detect(640x640 synthetic image) x ITERS
//   - Supertonic TTS  : synthesize("Hello world", 10 steps) x ITERS

import initSenseVoice, {
    SenseVoiceEngine,
    decode_wav,
} from './pkg/sensevoice-wasm/sensevoice_wasm.js';
import initYolo26, { Yolo26Engine } from './pkg/yolo26-wasm/yolo26_wasm.js';
import initSupertonic, {
    SupertonicEngine,
} from './pkg/supertonic-wasm/supertonic_wasm.js';

const MODEL_BASE = './models/';
const ITERS_ASR = 5;
const ITERS_YOLO = 10;
const ITERS_TTS = 3;

const log = document.getElementById('log');
function print(msg, cls = '') {
    console.log(msg);
    const span = document.createElement('span');
    if (cls) span.className = cls;
    span.textContent = msg + '\n';
    log.appendChild(span);
}

function stats(timings) {
    const sorted = [...timings].sort((a, b) => a - b);
    const mean = timings.reduce((s, x) => s + x, 0) / timings.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const median = sorted[Math.floor(sorted.length / 2)];
    const p90 = sorted[Math.floor(sorted.length * 0.9)];
    return { mean, min, max, median, p90 };
}

function fmtMs(v) { return v.toFixed(1) + 'ms'; }

async function fetchBytes(name) {
    const resp = await fetch(MODEL_BASE + name);
    if (!resp.ok) throw new Error(`fetch ${name} => ${resp.status}`);
    return new Uint8Array(await resp.arrayBuffer());
}

async function fetchText(name) {
    const resp = await fetch(MODEL_BASE + name);
    if (!resp.ok) throw new Error(`fetch ${name} => ${resp.status}`);
    return resp.text();
}

// ─── Synthetic data ───────────────────────────────────────────────────────────

/** 3 s of 16 kHz PCM (low-amplitude sine wave) */
function makeSyntheticAudio(durationSec = 3.0, sampleRate = 16000) {
    const n = Math.floor(durationSec * sampleRate);
    const audio = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        audio[i] = 0.05 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }
    return audio;
}

/** 640×640 RGBA synthetic image (grey gradient) */
function makeSyntheticImage(width = 640, height = 640) {
    const buf = new Uint8Array(width * height * 4);
    for (let i = 0; i < buf.length; i += 4) {
        const v = (i / 4) % 256;
        buf[i] = v; buf[i + 1] = v; buf[i + 2] = v; buf[i + 3] = 255;
    }
    return { buf, width, height };
}

// ─── Benchmark runners ────────────────────────────────────────────────────────

async function benchSenseVoice(results) {
    print('\n── SenseVoice ASR ──────────────────────────────────', 'bold');
    try {
        print('  Loading WASM module...', 'dim');
        const t0 = performance.now();
        await initSenseVoice();
        print(`  WASM init: ${fmtMs(performance.now() - t0)}`, 'dim');

        print('  Loading model weights (224 MB)...', 'dim');
        const t1 = performance.now();
        const [weights, tokens] = await Promise.all([
            fetchBytes('sensevoice_weights.bin'),
            fetchText('sensevoice.int8.tokens.txt'),
        ]);
        print(`  Model load: ${fmtMs(performance.now() - t1)}`, 'dim');

        print('  Initializing engine...', 'dim');
        const t2 = performance.now();
        const engine = new SenseVoiceEngine(weights, tokens);
        print(`  Engine init: ${fmtMs(performance.now() - t2)}`, 'dim');

        const audio = makeSyntheticAudio(3.0);
        const audioDurSec = audio.length / 16000;

        // Warmup (1 iter, not counted)
        print('  Warmup...', 'dim');
        engine.recognize(audio, 16000);

        // Timed iterations
        const timings = [];
        for (let i = 0; i < ITERS_ASR; i++) {
            const t = performance.now();
            engine.recognize(audio, 16000);
            timings.push(performance.now() - t);
            print(`  [${i + 1}/${ITERS_ASR}] ${fmtMs(timings[timings.length - 1])}`, 'dim');
        }

        const s = stats(timings);
        const rtf = s.mean / 1000 / audioDurSec;
        print(`  mean=${fmtMs(s.mean)} min=${fmtMs(s.min)} p90=${fmtMs(s.p90)} RTF=${rtf.toFixed(3)}x`, 'ok');

        results.push({
            name: 'SenseVoice ASR (3s audio)',
            iters: ITERS_ASR,
            mean_ms: +s.mean.toFixed(2),
            min_ms: +s.min.toFixed(2),
            p90_ms: +s.p90.toFixed(2),
            rtf: +rtf.toFixed(4),
        });
    } catch (e) {
        print(`  ERROR: ${e.message}`, 'err');
        console.error(e);
        results.push({ name: 'SenseVoice ASR', error: e.message });
    }
}

async function benchYolo26(results) {
    print('\n── YOLO26 Object Detection ─────────────────────────', 'bold');
    try {
        print('  Loading WASM module...', 'dim');
        const t0 = performance.now();
        await initYolo26();
        print(`  WASM init: ${fmtMs(performance.now() - t0)}`, 'dim');

        print('  Loading model weights (53 MB)...', 'dim');
        const t1 = performance.now();
        const weights = await fetchBytes('yolo26_weights.bin');
        print(`  Model load: ${fmtMs(performance.now() - t1)}`, 'dim');

        print('  Initializing engine...', 'dim');
        const t2 = performance.now();
        const engine = new Yolo26Engine(weights);
        print(`  Engine init: ${fmtMs(performance.now() - t2)}`, 'dim');

        const { buf, width, height } = makeSyntheticImage(640, 640);

        // Warmup
        print('  Warmup...', 'dim');
        engine.detect(buf, width, height, 0.3);

        // Timed iterations
        const timings = [];
        for (let i = 0; i < ITERS_YOLO; i++) {
            const t = performance.now();
            engine.detect(buf, width, height, 0.3);
            timings.push(performance.now() - t);
            print(`  [${i + 1}/${ITERS_YOLO}] ${fmtMs(timings[timings.length - 1])}`, 'dim');
        }

        const s = stats(timings);
        const fps = 1000 / s.mean;
        print(`  mean=${fmtMs(s.mean)} min=${fmtMs(s.min)} p90=${fmtMs(s.p90)} FPS=${fps.toFixed(2)}`, 'ok');

        results.push({
            name: 'YOLO26 detect (640×640)',
            iters: ITERS_YOLO,
            mean_ms: +s.mean.toFixed(2),
            min_ms: +s.min.toFixed(2),
            p90_ms: +s.p90.toFixed(2),
            fps: +fps.toFixed(2),
        });
    } catch (e) {
        print(`  ERROR: ${e.message}`, 'err');
        console.error(e);
        results.push({ name: 'YOLO26 detect', error: e.message });
    }
}

async function benchSupertonic(results) {
    print('\n── Supertonic TTS ──────────────────────────────────', 'bold');
    try {
        print('  Loading WASM module...', 'dim');
        const t0 = performance.now();
        await initSupertonic();
        print(`  WASM init: ${fmtMs(performance.now() - t0)}`, 'dim');

        print('  Loading model weights (~250 MB)...', 'dim');
        const t1 = performance.now();
        const [teW, dpW, veW, voW, configJson, indexerJson, styleJson] = await Promise.all([
            fetchBytes('textencoder_weights.bin'),
            fetchBytes('durationpredictor_weights.bin'),
            fetchBytes('vectorestimator_weights.bin'),
            fetchBytes('vocoder_weights.bin'),
            fetchText('tts.json'),
            fetchText('unicode_indexer.json'),
            fetchText('M1.json'),
        ]);
        print(`  Model load: ${fmtMs(performance.now() - t1)}`, 'dim');

        print('  Initializing engine...', 'dim');
        const t2 = performance.now();
        const engine = new SupertonicEngine(teW, dpW, veW, voW, configJson, indexerJson);
        engine.load_style('M1', styleJson);
        print(`  Engine init: ${fmtMs(performance.now() - t2)}`, 'dim');

        const TEXT = 'Hello world, this is a test.';
        const STEPS = 10;

        // Warmup
        print('  Warmup...', 'dim');
        const warmupSamples = engine.synthesize(TEXT, 'en', 'M1', 1.0, STEPS);
        const sampleRate = engine.sample_rate();
        const audioDurSec = warmupSamples.length / sampleRate;

        // Timed iterations
        const timings = [];
        for (let i = 0; i < ITERS_TTS; i++) {
            const t = performance.now();
            engine.synthesize(TEXT, 'en', 'M1', 1.0, STEPS);
            timings.push(performance.now() - t);
            print(`  [${i + 1}/${ITERS_TTS}] ${fmtMs(timings[timings.length - 1])}`, 'dim');
        }

        const s = stats(timings);
        const rtf = s.mean / 1000 / audioDurSec;
        print(`  mean=${fmtMs(s.mean)} min=${fmtMs(s.min)} p90=${fmtMs(s.p90)} RTF=${rtf.toFixed(3)}x steps=${STEPS}`, 'ok');

        results.push({
            name: `Supertonic TTS (${STEPS} steps)`,
            iters: ITERS_TTS,
            mean_ms: +s.mean.toFixed(2),
            min_ms: +s.min.toFixed(2),
            p90_ms: +s.p90.toFixed(2),
            rtf: +rtf.toFixed(4),
            steps: STEPS,
            out_dur_sec: +audioDurSec.toFixed(3),
        });
    } catch (e) {
        print(`  ERROR: ${e.message}`, 'err');
        console.error(e);
        results.push({ name: 'Supertonic TTS', error: e.message });
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

(async () => {
    print('=== Lele WASM Browser Benchmark ===', 'bold');
    print(`User-Agent: ${navigator.userAgent}`, 'dim');
    print(`SIMD support: ${typeof WebAssembly.validate === 'function'
        ? WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]))
            ? 'YES' : 'NO (fallback scalar path)'
        : 'unknown'}`, 'dim');

    const results = [];

    // Parse ?model= query to run only a specific benchmark
    const params = new URLSearchParams(location.search);
    const only = params.get('model') || 'all';

    if (only === 'all' || only === 'sensevoice') await benchSenseVoice(results);
    if (only === 'all' || only === 'yolo26') await benchYolo26(results);
    if (only === 'all' || only === 'supertonic') await benchSupertonic(results);

    print('\n=== Summary ═══════════════════════════════════════', 'bold');
    for (const r of results) {
        if (r.error) {
            print(`  ${r.name}: FAILED - ${r.error}`, 'err');
        } else {
            const extra = r.rtf != null
                ? `RTF=${r.rtf}x`
                : r.fps != null ? `FPS=${r.fps}` : '';
            print(`  ${r.name}: mean=${fmtMs(r.mean_ms)} p90=${fmtMs(r.p90_ms)} ${extra}`, 'ok');
        }
    }

    // Expose results to Playwright/CDP
    window.__benchmarkResults = results;
    window.__benchmarkDone = true;
    print('\n[DONE] Results available in window.__benchmarkResults', 'dim');
})();
