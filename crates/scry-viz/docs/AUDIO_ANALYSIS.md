# Audio Analysis Spec

## Goal

The analyzer should turn sound into stable musical signals that visual systems
can trust. Raw FFT bins are useful for detail, but high-quality visuals need
semantic controls: impact, flow, density, brightness, section energy, and
musical phase.

## Current Analyzer

`dsp.rs` currently emits `AnalysisFrame` from `analysis.rs`:

| Field | Meaning |
|-------|---------|
| `time_s`, `dt_s` | Analysis timeline and frame delta |
| `sample_rate` | Source sample rate for live analysis |
| `bands` | Smoothed log-spaced frequency bands, low to high |
| `peaks` | Falling peak markers per band |
| `waveform` | Recent mono waveform aligned to a rising zero crossing |
| `rms` | Overall loudness |
| `bass`, `low_mid`, `high_mid`, `treble` | Smoothed broad frequency-role controls |
| `transient` | Short-term positive spectrum change |
| `beat` | Beat onset impulse, envelope, confidence, and optional tempo state |
| `tonal` | Optional spectral centroid/rolloff/chroma/key state |

This is a good low-latency baseline. It should remain lightweight and usable
without offline preprocessing.

## Analysis Layers

### Layer 1: Real-Time Signal Features

Required for live mode:

- RMS/loudness.
- Log-spaced spectrum bands.
- Band peaks and short-term deltas.
- Bass, low-mid, high-mid, treble energy.
- Transient/onset envelope.
- Waveform.
- Silence/noise gate.

### Layer 2: Beat and Tempo

Useful in both live and prepared modes:

- Beat onset confidence.
- BPM estimate when stable.
- Beat phase from 0..1.
- Bar phase from 0..1 when confidence is high.
- Downbeat hints in prepared mode.

Live mode may expose `None` for BPM and bar phase until confidence is high.
Visual modes must degrade gracefully.

### Layer 3: Tonal and Timbre Features

Prepared mode should add:

- Spectral centroid.
- Spectral rolloff.
- Spectral flux.
- Chroma/key estimate where possible.
- Harmonic/percussive split if available.
- Vocal or lead-presence hints if available.

These features should guide palette, camera, and foreground shape decisions.

### Layer 4: Structure

Prepared mode should produce timeline-level features:

- Section boundaries.
- Energy envelope over multiple bars.
- Build/drop/breakdown/chorus hints.
- Repeated motif hints if available.
- Cue points for major visual transitions.

## Stem-Aware Direction

The strongest jump in quality comes from separating musical roles:

| Audio Role | Visual Role |
|------------|-------------|
| Kick/bass | Scale, gravity, core pulse, camera pressure |
| Snare/clap | Impact rings, cuts, flashes, edge accents |
| Hats/percussion | Fine particles, shimmer, detail fields |
| Vocals/lead | Foreground ribbons, typography, facial/organic motion |
| Harmony/pads | Background fields, color drift, atmosphere |

Offline stem separation can be optional and cached. Live mode should approximate
these roles with band groups and transients.

## Normalization

Every visual-facing scalar should be normalized to 0..1 unless there is a strong
reason not to. Normalization must be stable:

- Avoid pinning to 1.0 for entire loud tracks.
- Avoid amplifying digital silence into visual noise.
- Use attack/release smoothing rather than a single smoothing constant.
- Keep sudden impact events separate from long-term intensity.

Recommended control families:

| Control | Attack | Release | Usage |
|---------|--------|---------|-------|
| `impact` | Instant/fast | Fast | Beat flashes, shockwaves |
| `intensity` | Medium | Medium | Overall density and scale |
| `flow` | Slow | Slow | Camera drift, fluid motion |
| `brightness` | Medium | Slow | Palette/luminance |
| `density` | Medium | Medium | Particle spawn, path count |

## Contract

The analyzer should produce a frame that can be consumed by both live and
prepared renderers:

```rust
pub struct AnalysisFrame {
    pub time_s: f32,
    pub dt_s: f32,
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

Keep raw samples out of the visual contract. If a mode needs waveform geometry,
it should use `waveform`.

## Prepared Analysis Cache

Prepared mode should cache analysis by:

- Audio content hash.
- Analyzer version.
- Analyzer settings.
- Optional stem model/version.

The cache should be invalidated when any of those inputs change.

Suggested format:

```text
.scry-viz-cache/
  <audio-hash>/
    analysis-v1.json
    stems/
      drums.wav
      bass.wav
      vocals.wav
      other.wav
```

Use a compact binary format later if JSON becomes a bottleneck.

## Acceptance Tests

Minimum:

- Silence decays to near zero and does not trigger beats.
- A 440 Hz sine peaks in the expected log band.
- A bass impulse triggers one beat event, not a burst of repeated events.
- A constant tone does not create continuous transient events.
- Increasing band count preserves low-to-high ordering.
- Analysis output is deterministic for fixed input samples and settings.
