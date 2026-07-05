# scry-viz

Pixel-perfect terminal music visualizer powered by `scry-engine`.

The product vision: the ricer's best friend for music visualization. It should
fit cleanly into customized terminal and desktop setups, with shareable presets,
themeable visuals, transparent-friendly rendering, and enough musical
intelligence to feel composed rather than merely reactive.

`scry-viz` captures live desktop audio through PulseAudio/PipeWire, analyzes it
with a real-time FFT pipeline, and renders reactive vector graphics through the
same terminal transport stack used by the rest of Scry.

## Current Status

This crate is an alpha scaffold with a working live visualizer:

- Live capture from the default sink monitor or a named PulseAudio source.
- Log-spaced FFT bands behind an `AnalysisFrame` contract with smoothing, peak
  hold, auto-gain, RMS, bass/low-mid/high-mid/treble groups, transient, beat,
  and zero-crossing-aligned waveform data.
- Eleven 2D visual modes: `silk`, `ridge`, `mandala`, `nova`, `bars`,
  `radial`, `wave`, `spectrogram`, `vortex`, `constellation`, and `prism`.
- Theme cycling across `neon`, `aurora`, `sunset`, `matrix`, `ice`, and `ember`.
- Terminal rendering through Kitty when available, with halfblock fallback.

The long-term goal is more ambitious: a music-aware visual engine that turns
beats, sections, stems, timbre, and energy into a scored 2D/3D performance.

## Run

```bash
cargo run -p scry-viz -- --mode silk --theme neon
```

Useful flags:

```bash
cargo run -p scry-viz -- --mode nova --bands 96 --fps 60
cargo run -p scry-viz -- --device "@DEFAULT_MONITOR@" --theme aurora
cargo run -p scry-viz -- --visual-only --mode silk --theme neon
cargo run -p scry-viz -- --no-hud --mode radial
cargo run -p scry-viz -- --mode vortex --theme ice --bands 128
cargo run -p scry-viz -- --mode constellation --theme aurora
```

Runtime controls:

| Key | Action |
|-----|--------|
| `tab` | Next visual mode |
| `1`-`9`, `0`, `-` | Select a mode directly |
| `t` | Next theme |
| `s` | Toggle HUD |
| `space` | Pause analysis |
| `q` / `esc` | Quit |

## Documentation

- [Product Spec](docs/SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Audio Analysis](docs/AUDIO_ANALYSIS.md)
- [Visual Language](docs/VISUAL_LANGUAGE.md)
- [Aesthetic Integration](docs/AESTHETIC_INTEGRATION.md)
- [Roadmap](docs/ROADMAP.md)
- [Research Notes](docs/RESEARCH.md)

## Design Direction

Most visualizers react to amplitude. `scry-viz` should react to music.

The next architecture should preserve the current low-latency live mode while
adding richer analysis and visual scoring:

```text
audio capture -> analysis frame -> visual score -> 2D/3D scene -> terminal/window/export
```

Live mode should feel immediate. Offline/prepared mode should feel composed,
with section-aware transitions, stem-aware motion, and visual memory across
phrases instead of one-frame reactions.
