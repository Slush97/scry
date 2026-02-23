#!/usr/bin/env bash
# Quick launcher for scry-stt live transcription
# Usage: ./stt.sh [bench]
#   ./stt.sh        — live mic transcription
#   ./stt.sh bench  — cold-start benchmark (no mic needed)

set -e
cd "$(dirname "$0")"

case "${1:-live}" in
  live)
    cargo run --release -p scry-stt --features "safetensors,live" --example live_transcribe
    ;;
  bench)
    cargo run --release -p scry-stt --features safetensors --example bench_cold_start
    ;;
  vs)
    cargo run --release -p scry-stt --features "safetensors,live" --example bench_vs_python
    ;;
  *)
    echo "Usage: ./stt.sh [live|bench|vs]"
    echo "  live   — live mic transcription (default)"
    echo "  bench  — cold-start benchmark (synthetic audio, no mic)"
    echo "  vs     — side-by-side vs Python whisper (needs mic + pip install openai-whisper)"
    exit 1
    ;;
esac
