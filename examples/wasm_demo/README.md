# WASM Demo

Interactive demo of scry-engine rendering to an HTML5 `<canvas>` via WebAssembly.

## Prerequisites

```bash
# Install wasm-pack (if not already installed)
cargo install wasm-pack

# Install the WASM target
rustup target add wasm32-unknown-unknown
```

## Build

```bash
# From the repository root:
wasm-pack build --target web --no-default-features --features wasm -- -p scry-engine

# This produces pkg/ in the repo root with:
#   scry_engine.js      — JS glue code
#   scry_engine_bg.wasm — compiled WASM binary
```

## Run

```bash
# Copy pkg/ next to the demo page, then serve:
cp -r pkg examples/wasm_demo/
cd examples/wasm_demo
python3 -m http.server 8080

# Open http://localhost:8080 in your browser.
```

## What the demo does

- Creates a `WasmCanvas` (400×300) with a dark background
- Renders circles and rectangles via scry-engine's rasterizer
- Blits the RGBA pixels to an HTML `<canvas>` using `ImageData`
- Interactive buttons to add random shapes or clear the scene
