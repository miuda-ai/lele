// bench_node.mjs - Run WASM benchmarks directly in Node.js (no browser needed)
//
// Node.js 18+ supports WebAssembly natively and has TextDecoder/TextEncoder/
// FinalizationRegistry built-in. wasm-bindgen output can be used directly
// by passing ArrayBuffer instead of a URL to the init function.
//
// This gives fast feedback during development. For true browser-accurate
// numbers use run_bench.mjs (Playwright/CDP with Chrome V8).
//
// Usage:
//   node bench_node.mjs               # benchmark all models
//   node bench_node.mjs --model yolo26
//   node bench_node.mjs --model yolo26n-seg
//   node bench_node.mjs --model sensevoice

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { performance } from 'perf_hooks';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = join(__dirname, 'web');
const MODELS_DIR = join(WEB_DIR, 'models');
const PKG_DIR = join(WEB_DIR, 'pkg');

const args = process.argv.slice(2);
const only = args.includes('--model') ? args[args.indexOf('--model') + 1] : 'all';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function readModel(name) {
    return readFileSync(join(MODELS_DIR, name)).buffer;
}

function readModelText(name) {
    return readFileSync(join(MODELS_DIR, name), 'utf8');
}

function readWasm(pkg, file) {
    return readFileSync(join(PKG_DIR, pkg, file)).buffer;
}

function stats(timings) {
    const sorted = [...timings].sort((a, b) => a - b);
    const mean = timings.reduce((s, x) => s + x, 0) / timings.length;
    return {
        mean,
        min: sorted[0],
        max: sorted[sorted.length - 1],
        p90: sorted[Math.floor(sorted.length * 0.9)],
    };
}

function fmtMs(v) { return v.toFixed(1) + 'ms'; }

function makeSyntheticAudio(durationSec = 3.0, sampleRate = 16000) {
    const n = Math.floor(durationSec * sampleRate);
    const audio = new Float32Array(n);
    for (let i = 0; i < n; i++) audio[i] = 0.05 * Math.sin(2 * Math.PI * 440 * i / sampleRate);
    return audio;
}

function makeSyntheticImage(w = 640, h = 640) {
    const buf = new Uint8Array(w * h * 4);
    for (let i = 0; i < buf.length; i += 4) {
        const v = (i / 4) % 256;
        buf[i] = v; buf[i + 1] = v; buf[i + 2] = v; buf[i + 3] = 255;
    }
    return { buf, width: w, height: h };
}

// ─── YOLO26 Benchmark ─────────────────────────────────────────────────────────

async function benchYolo26(results) {
    console.log('\n── YOLO26 Object Detection ─────────────────────────');
    const ITERS = 3;
    try {
        // Load WASM module (pass ArrayBuffer to bypass fetch)
        const wasmBytes = readWasm('yolo26-wasm', 'yolo26_wasm_bg.wasm');
        const { default: init, Yolo26Engine } = await import(
            join(PKG_DIR, 'yolo26-wasm', 'yolo26_wasm.js')
        );
        const t0 = performance.now();
        await init({ module_or_path: wasmBytes });
        console.log(`  WASM init: ${fmtMs(performance.now() - t0)}`);

        const t1 = performance.now();
        const weights = readModel('yolo26_weights.bin');
        console.log(`  Model load: ${fmtMs(performance.now() - t1)}`);

        const t2 = performance.now();
        const engine = new Yolo26Engine(new Uint8Array(weights));
        console.log(`  Engine init: ${fmtMs(performance.now() - t2)}`);

        const { buf, width, height } = makeSyntheticImage(640, 640);

        // Warmup
        engine.detect(buf, width, height, 0.3);

        // Timed iterations
        const timings = [];
        for (let i = 0; i < ITERS; i++) {
            const t = performance.now();
            engine.detect(buf, width, height, 0.3);
            timings.push(performance.now() - t);
            console.log(`  [${i + 1}/${ITERS}] ${fmtMs(timings[timings.length - 1])}`);
        }

        const s = stats(timings);
        const fps = 1000 / s.mean;
        console.log(`  mean=${fmtMs(s.mean)} min=${fmtMs(s.min)} p90=${fmtMs(s.p90)} FPS=${fps.toFixed(2)}`);

        results.push({
            name: 'YOLO26 detect (640×640) [Node.js WASM]',
            iters: ITERS,
            mean_ms: +s.mean.toFixed(2),
            min_ms: +s.min.toFixed(2),
            p90_ms: +s.p90.toFixed(2),
            fps: +fps.toFixed(2),
        });
    } catch (e) {
        console.error(`  ERROR: ${e.message}`);
        results.push({ name: 'YOLO26 detect', error: e.message });
    }
}

// ─── YOLO26n-Seg Benchmark ───────────────────────────────────────────────────

async function benchYolo26NSeg(results) {
    console.log('\n── YOLO26n-Seg Instance Segmentation ──────────────');
    const ITERS = 3;
    try {
        const wasmBytes = readWasm('yolo26n-seg-wasm', 'yolo26n_seg_wasm_bg.wasm');
        const { default: init, Yolo26NSegEngine } = await import(
            join(PKG_DIR, 'yolo26n-seg-wasm', 'yolo26n_seg_wasm.js')
        );

        const t0 = performance.now();
        await init({ module_or_path: wasmBytes });
        console.log(`  WASM init: ${fmtMs(performance.now() - t0)}`);

        const t1 = performance.now();
        const weights = readModel('yolo26seg_weights.bin');
        console.log(`  Model load: ${fmtMs(performance.now() - t1)}`);

        const t2 = performance.now();
        const engine = new Yolo26NSegEngine(new Uint8Array(weights));
        console.log(`  Engine init: ${fmtMs(performance.now() - t2)}`);

        const { buf, width, height } = makeSyntheticImage(640, 640);

        // Warmup
        engine.segment_profile(buf, width, height, 0.3);

        const timings = [];
        const preprocessTimings = [];
        const forwardTimings = [];
        const postprocessTimings = [];
        const serializeTimings = [];
        for (let i = 0; i < ITERS; i++) {
            const t = performance.now();
            const profileJson = engine.segment_profile(buf, width, height, 0.3);
            const profile = JSON.parse(profileJson);
            timings.push(performance.now() - t);
            preprocessTimings.push(profile.preprocess_ms);
            forwardTimings.push(profile.forward_ms);
            postprocessTimings.push(profile.postprocess_ms);
            serializeTimings.push(profile.serialize_ms);
            console.log(
                `  [${i + 1}/${ITERS}] total=${fmtMs(timings[timings.length - 1])}`
                + ` pre=${fmtMs(profile.preprocess_ms)}`
                + ` fwd=${fmtMs(profile.forward_ms)}`
                + ` post=${fmtMs(profile.postprocess_ms)}`
                + ` ser=${fmtMs(profile.serialize_ms)}`
                + ` detections=${profile.detections}`
            );
        }

        const s = stats(timings);
        const sPre = stats(preprocessTimings);
        const sFwd = stats(forwardTimings);
        const sPost = stats(postprocessTimings);
        const sSer = stats(serializeTimings);
        const fps = 1000 / s.mean;
        console.log(`  mean=${fmtMs(s.mean)} min=${fmtMs(s.min)} p90=${fmtMs(s.p90)} FPS=${fps.toFixed(2)}`);
        console.log(
            `  stage-mean: pre=${fmtMs(sPre.mean)} fwd=${fmtMs(sFwd.mean)} post=${fmtMs(sPost.mean)} ser=${fmtMs(sSer.mean)}`
        );

        results.push({
            name: 'YOLO26n-Seg segment (640×640) [Node.js WASM]',
            iters: ITERS,
            mean_ms: +s.mean.toFixed(2),
            min_ms: +s.min.toFixed(2),
            p90_ms: +s.p90.toFixed(2),
            fps: +fps.toFixed(2),
            preprocess_ms: +sPre.mean.toFixed(2),
            forward_ms: +sFwd.mean.toFixed(2),
            postprocess_ms: +sPost.mean.toFixed(2),
            serialize_ms: +sSer.mean.toFixed(2),
        });
    } catch (e) {
        console.error(`  ERROR: ${e.message}`);
        results.push({ name: 'YOLO26n-Seg segment', error: e.message });
    }
}

// ─── SenseVoice Benchmark ────────────────────────────────────────────────────

async function benchSenseVoice(results) {
    console.log('\n── SenseVoice ASR ──────────────────────────────────');
    const ITERS = 2;
    try {
        const wasmBytes = readWasm('sensevoice-wasm', 'sensevoice_wasm_bg.wasm');
        const { default: init, SenseVoiceEngine } = await import(
            join(PKG_DIR, 'sensevoice-wasm', 'sensevoice_wasm.js')
        );

        const t0 = performance.now();
        await init({ module_or_path: wasmBytes });
        console.log(`  WASM init: ${fmtMs(performance.now() - t0)}`);

        console.log('  Loading model weights (224 MB)...');
        const t1 = performance.now();
        const weights = readModel('sensevoice_weights.bin');
        const tokens = readModelText('sensevoice.int8.tokens.txt');
        console.log(`  Model load: ${fmtMs(performance.now() - t1)}`);

        const t2 = performance.now();
        const engine = new SenseVoiceEngine(new Uint8Array(weights), tokens);
        console.log(`  Engine init: ${fmtMs(performance.now() - t2)}`);

        const audio = makeSyntheticAudio(3.0);
        const audioDurSec = audio.length / 16000;

        // Warmup
        engine.recognize(audio, 16000);

        const timings = [];
        for (let i = 0; i < ITERS; i++) {
            const t = performance.now();
            engine.recognize(audio, 16000);
            timings.push(performance.now() - t);
            console.log(`  [${i + 1}/${ITERS}] ${fmtMs(timings[timings.length - 1])}`);
        }

        const s = stats(timings);
        const rtf = s.mean / 1000 / audioDurSec;
        console.log(`  mean=${fmtMs(s.mean)} min=${fmtMs(s.min)} p90=${fmtMs(s.p90)} RTF=${rtf.toFixed(3)}x`);

        results.push({
            name: 'SenseVoice ASR (3s audio) [Node.js WASM]',
            iters: ITERS,
            mean_ms: +s.mean.toFixed(2),
            min_ms: +s.min.toFixed(2),
            p90_ms: +s.p90.toFixed(2),
            rtf: +rtf.toFixed(4),
        });
    } catch (e) {
        console.error(`  ERROR: ${e.message}`);
        results.push({ name: 'SenseVoice ASR', error: e.message });
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const HR = '═'.repeat(55);
console.log(`\n${HR}`);
console.log('  Lele WASM Benchmark (Node.js v' + process.versions.node + ')');
console.log('  NOTE: Node.js WASM = no V8 JIT warm-up lag mismatch vs browser,');
console.log('        but same V8 WASM engine → numbers are comparable');
console.log(HR);

const results = [];
if (only === 'all' || only === 'yolo26') await benchYolo26(results);
if (only === 'all' || only === 'yolo26n-seg' || only === 'yolo26nseg') await benchYolo26NSeg(results);
if (only === 'all' || only === 'sensevoice') await benchSenseVoice(results);

console.log(`\n${HR}`);
console.log('  Summary');
console.log(HR);
for (const r of results) {
    if (r.error) {
        console.log(`  ✗ ${r.name}: ${r.error}`);
    } else {
        const extra = r.rtf != null ? `RTF=${r.rtf}x` : r.fps != null ? `FPS=${r.fps}` : '';
        console.log(`  ✓ ${r.name}: mean=${fmtMs(r.mean_ms)} p90=${fmtMs(r.p90_ms)} ${extra}`);
    }
}
console.log(`\n${HR}\n`);
