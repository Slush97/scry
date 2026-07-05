# scry-viz Architecture

## Overview

`scry-viz` should be organized as a staged audiovisual pipeline:

```text
capture/file input
    -> sample ring
    -> analysis frame
    -> visual score
    -> visual scene
    -> render target
    -> transport/export
```

The current crate implements the first, second, third, and fifth stages in a
minimal form. The next architecture formalizes the middle contracts so richer
2D and 3D visuals can be added without coupling them directly to FFT details.

## Current Runtime

```text
PulseAudio thread
    -> SharedAudio ring
    -> Analyzer::update()
    -> AnalysisFrame
    -> Mode::build()
    -> PixelCanvas
    -> PixelCanvasWidget
    -> Kitty or halfblock transport
```

Important current modules:

| Module | Responsibility |
|--------|----------------|
| `audio.rs` | Live PulseAudio/PipeWire capture and shared sample ring |
| `analysis.rs` | Stable visual-facing analysis contract and band-group helpers |
| `dsp.rs` | FFT, log bands, smoothing, auto-gain, grouped energy, transient, beat, waveform |
| `theme.rs` | Palette sampling and theme definitions |
| `viz/mod.rs` | Mode dispatch, shared mode state, utility shaping functions |
| `viz/*.rs` | Individual visual modes that build `PixelCanvas` scenes |
| `main.rs` | CLI, terminal lifecycle, input loop, transport selection |

## Target Module Boundaries

Planned layout:

```text
src/
  audio.rs             live capture backends and file-input adapters
  analysis/
    mod.rs             public analysis contract
    realtime.rs        low-latency FFT analyzer
    offline.rs         prepared-track analysis
    onset.rs           onset and beat features
    cache.rs           serialized analysis cache
  score/
    mod.rs             visual score contract
    conductor.rs       maps analysis to scene controls
    timeline.rs        sections, phrases, bars, events
  render/
    mod.rs             target-independent frame rendering
    terminal.rs        Ratatui + scry-engine target
    export.rs          deterministic image/frame export
  viz/
    mod.rs             mode registry and shared state
    ...
```

Do not add these modules all at once unless they remove real coupling. The
first useful step is to introduce `analysis` and `score` contracts while keeping
the current runtime path working.

## Data Contracts

### AnalysisFrame

The visual layer should not pull raw FFT state directly from the analyzer.
It should consume an analysis frame with normalized, semantically named fields.

```rust
pub struct AnalysisFrame {
    pub time_s: f32,
    pub dt_s: f32,
    pub sample_rate: u32,
    pub bands: Vec<f32>,
    pub peaks: Vec<f32>,
    pub waveform: Vec<f32>,
    pub rms: f32,
    pub bass: f32,
    pub low_mid: f32,
    pub high_mid: f32,
    pub treble: f32,
    pub transient: f32,
    pub beat: BeatState,
    pub tonal: TonalState,
}
```

### BeatState

```rust
pub struct BeatState {
    pub onset: f32,
    pub envelope: f32,
    pub confidence: f32,
    pub bpm: Option<f32>,
    pub phase: Option<f32>,
    pub bar_phase: Option<f32>,
}
```

### VisualScore

`VisualScore` is the conductor layer. It maps analysis into scene-level controls
that remain stable across frames.

```rust
pub struct VisualScore {
    pub intensity: f32,
    pub density: f32,
    pub impact: f32,
    pub flow: f32,
    pub brightness: f32,
    pub palette_shift: f32,
    pub camera_pressure: f32,
    pub section: SectionHint,
    pub events: Vec<ScoreEvent>,
}
```

Visual modes may still use spectrum bands for detailed geometry, but large
decisions should come from the score. This avoids every mode reinventing
thresholds, beat detection, and intensity smoothing.

## Thread Model

Current:

- Capture thread writes mono samples into a mutex-protected ring.
- UI/render thread reads the ring, analyzes, builds a scene, draws, and flushes.

Target:

- Capture thread remains isolated.
- Real-time analysis stays in the render thread unless profiling shows it needs
  a dedicated worker.
- Offline analysis can run before playback and cache results.
- Export mode should avoid terminal event polling entirely.

## Timing Model

Live mode:

- Wall-clock time drives animation.
- Analyzer uses latest samples.
- Visual score smooths noisy inputs and emits impact events.

Prepared/export mode:

- Audio timeline time drives animation.
- Analysis frames are indexed by timestamp.
- Render stepping is deterministic and independent of wall-clock time.

## Rendering Targets

Initial target:

- `TerminalTarget`: `PixelCanvasWidget` into Ratatui, Kitty preferred, halfblock
  fallback.

Future targets:

- `PixmapTarget`: render to RGBA frames for tests and export.
- `WindowTarget`: native window via Scry window transport.
- `FrameStreamTarget`: pipe frames to external encoders or VJ tools.

## Performance Budget

At 60 fps, each frame has 16.67 ms. A practical budget:

| Stage | Budget |
|-------|--------|
| Audio copy + analysis | 1.5 ms |
| Score update | 0.2 ms |
| Scene build | 2.0 ms |
| Raster | 5.0 ms |
| Transport flush | 6.0 ms |
| Input/event overhead | 1.0 ms |

Transport cost can dominate in terminals. Visual modes should avoid huge numbers
of tiny shapes unless they are batched or cached well.

## Failure Behavior

Audio capture failure:

- Return a clear error before entering alternate screen.
- Suggest passing `--device` if the default monitor cannot be opened.

Terminal protocol fallback:

- Use Kitty when detected.
- Use halfblock otherwise.
- Keep visual density readable in halfblock mode.

Render overload:

- Drop visual detail before dropping terminal cleanup.
- Clamp fps to a safe range.
- Keep controls responsive under load.

## Testing Strategy

Unit tests:

- Analyzer: silence, sine peaks, transient detection, smoothing decay.
- Score: event emission, smoothing, deterministic seeding.
- Visual helpers: band grouping, spline degeneracy, palette sampling.

Snapshot/golden tests:

- Deterministic mode render from fixed analysis frames.
- Tiny, normal, and large canvases.
- Theme contrast and alpha behavior.

Manual tests:

- Live capture on PipeWire/PulseAudio.
- Kitty graphics and halfblock fallback.
- Terminal cleanup after quit and after capture failure.
