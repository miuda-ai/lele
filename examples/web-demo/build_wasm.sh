#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
# Lele Web Demo - WASM Build Script
# Builds 3 separate WASM modules:
#   1. sensevoice-wasm  (ASR)
#   2. yolo26-wasm      (Object Detection)
#   3. supertonic-wasm  (TTS)
# ─────────────────────────────────────────────
# Prerequisites:
#   cargo install wasm-pack
#   rustup target add wasm32-unknown-unknown
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEB_DIR="$SCRIPT_DIR/web"
MODELS_DIR="$WEB_DIR/models"
CRATES_DIR="$SCRIPT_DIR/crates"

echo "=== Lele WASM Build (Separate Modules) ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check wasm-pack
if ! command -v wasm-pack &>/dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# ── Build each WASM module ──
build_module() {
    local name=$1
    local crate_dir="$CRATES_DIR/$name"
    local out_dir="$WEB_DIR/pkg/$name"

    echo "--- Building $name ---"
    cd "$crate_dir"
    # Use env overrides for optimal WASM release settings
    # (workspace profile may have different settings for native builds)
    CARGO_PROFILE_RELEASE_LTO=true \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_PROFILE_RELEASE_PANIC=abort \
    CARGO_PROFILE_RELEASE_OPT_LEVEL=3 \
    wasm-pack build --target web --out-dir "$out_dir" --release
    local wasm_file="$out_dir/${name//-/_}_bg.wasm"
    if [ -f "$wasm_file" ]; then
        local size=$(ls -lh "$wasm_file" | awk '{print $5}')
        echo "✓ $name: $size"
    fi
    echo ""
}

build_module "sensevoice-wasm"
build_module "yolo26-wasm"
build_module "supertonic-wasm"

# ── Copy model weights ──
echo "--- Copying model files ---"
mkdir -p "$MODELS_DIR"

# SenseVoice
copy_if_exists() {
    local src=$1
    local dest=$2
    local name=$3
    if [ -f "$src" ]; then
        cp "$src" "$dest"
        echo "✓ $name"
        return 0
    fi
    return 1
}

copy_if_exists "$PROJECT_ROOT/examples/sensevoice/src/sensevoice_weights.bin" "$MODELS_DIR/" "sensevoice_weights.bin" || \
copy_if_exists "$CRATES_DIR/sensevoice-wasm/src/gen/sensevoice_weights.bin" "$MODELS_DIR/" "sensevoice_weights.bin (from gen)" || \
echo "⚠ sensevoice_weights.bin not found"

copy_if_exists "$PROJECT_ROOT/examples/sensevoice/sensevoice.int8.tokens.txt" "$MODELS_DIR/" "sensevoice.int8.tokens.txt" || \
echo "⚠ sensevoice.int8.tokens.txt not found"

# YOLO26
copy_if_exists "$PROJECT_ROOT/examples/yolo26/src/yolo26_weights.bin" "$MODELS_DIR/" "yolo26_weights.bin" || \
copy_if_exists "$CRATES_DIR/yolo26-wasm/src/gen/yolo26_weights.bin" "$MODELS_DIR/" "yolo26_weights.bin (from gen)" || \
echo "⚠ yolo26_weights.bin not found"

# Supertonic weights
for model in textencoder durationpredictor vectorestimator vocoder; do
    copy_if_exists "$PROJECT_ROOT/examples/supertonic/src/${model}_weights.bin" "$MODELS_DIR/" "${model}_weights.bin" || \
    copy_if_exists "$CRATES_DIR/supertonic-wasm/src/gen/${model}_weights.bin" "$MODELS_DIR/" "${model}_weights.bin (from gen)" || \
    echo "⚠ ${model}_weights.bin not found"
done

# Supertonic config files
copy_if_exists "$PROJECT_ROOT/examples/supertonic/onnx/tts.json" "$MODELS_DIR/" "tts.json" || \
copy_if_exists "$SCRIPT_DIR/onnx/tts.json" "$MODELS_DIR/" "tts.json (local)" || \
echo "⚠ tts.json not found"

copy_if_exists "$PROJECT_ROOT/examples/supertonic/onnx/unicode_indexer.json" "$MODELS_DIR/" "unicode_indexer.json" || \
copy_if_exists "$SCRIPT_DIR/onnx/unicode_indexer.json" "$MODELS_DIR/" "unicode_indexer.json (local)" || \
echo "⚠ unicode_indexer.json not found"

copy_if_exists "$PROJECT_ROOT/examples/supertonic/voice_styles/M1.json" "$MODELS_DIR/" "M1.json" || \
copy_if_exists "$SCRIPT_DIR/voice_styles/M1.json" "$MODELS_DIR/" "M1.json (local)" || \
echo "⚠ M1.json not found"

# ── Summary ──
echo ""
echo "=== WASM Module Sizes ==="
for name in sensevoice-wasm yolo26-wasm supertonic-wasm; do
    wasm_file="$WEB_DIR/pkg/$name/${name//-/_}_bg.wasm"
    if [ -f "$wasm_file" ]; then
        size=$(ls -lh "$wasm_file" | awk '{print $5}')
        printf "  %-20s %s\n" "$name:" "$size"
    fi
done

echo ""
if command -v du &>/dev/null; then
    MODEL_SIZE=$(du -sh "$MODELS_DIR" 2>/dev/null | head -1 | cut -f1)
    echo "Model files total: ${MODEL_SIZE:-unknown}"
fi

echo ""
echo "To serve the demo:"
echo "  cd $WEB_DIR"
echo "  python3 -m http.server 8080"
echo "  # Then open http://localhost:8080"
