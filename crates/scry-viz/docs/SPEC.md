# scry-viz Product Spec

## Mission

`scry-viz` turns music into a high-quality real-time visual performance inside
the terminal, then grows into a broader Scry visual engine for live, recorded,
and exportable audio-reactive work.

The product vision is to become the ricer's best friend for music visualization:
an aesthetic system component that fits into customized terminals, desktops,
streams, wallpapers, and dotfiles instead of feeling like a generic media-player
plugin.

The product should move beyond ordinary spectrum visualizers. It should behave
like an audiovisual conductor: it listens to musical structure, preserves visual
state over time, and directs scenes, camera motion, palettes, particles, and
transitions with musical intent.

## North Star

A user can play a track, launch `scry-viz`, and see a polished performance that
feels composed to the music:

- Bass and kick drive large-scale geometry, camera pressure, and impact events.
- Snare, hats, and transients drive accents, sparks, cuts, and detail motion.
- Vocals and melodic material influence foreground shapes, typography, and color.
- Sections such as intro, build, drop, breakdown, chorus, and outro drive scene
  selection and transition timing.
- Visual memory persists across bars so the piece evolves instead of blinking.

## Current Baseline

The current crate provides:

- PulseAudio/PipeWire capture through `libpulse-simple-binding`.
- A real-time FFT analyzer that emits `AnalysisFrame` with log bands, RMS,
  broad band groups, transient, beat state, peak hold, auto-gain, and waveform
  extraction.
- Seven terminal-native 2D modes rendered through `scry-engine`.
- Manual mode, theme, pause, and HUD controls, plus no-HUD launch flags.

This is useful as the live low-latency baseline. The next work should avoid
throwing it away; instead, it should layer a visual score above the analysis
contract.

## Users

Primary users:

- Desktop ricers and terminal power users who want visuals that match their
  theme, transparency, launcher, and dotfiles workflow.
- Developers using Scry who want a high-end terminal demo.
- Musicians, DJs, and visual artists who want live reactive visuals without a
  heavyweight show-control stack.
- Creative coders who want a Rust-native visual engine with terminal output.

Secondary users:

- People rendering loopable clips for social media or documentation.
- Artists experimenting with shader-like visual systems in a constrained medium.
- Scry maintainers who need a stress test for animation, transport, and raster.

## Use Cases

### Live Desktop Audio

The user runs `scry-viz` while music plays in another application. The visualizer
captures the default sink monitor and reacts with low latency.

Acceptance:

- Startup succeeds without requiring explicit device configuration on common
  PipeWire/PulseAudio Linux systems.
- Visual response to kick/snare feels under 100 ms.
- The terminal is restored on exit.

### Riced Terminal Companion

The user launches `scry-viz` from a keybind or shell alias inside a transparent
terminal. The visualizer matches their theme, hides chrome when requested, and
can be reproduced from a preset.

Acceptance:

- Visual-only mode produces a clean surface without HUD chrome.
- Theme and seed can be stored in a preset file.
- Alpha and contrast remain readable over transparent terminal backgrounds.

### Prepared Track Performance

The user points `scry-viz` at an audio file. The app performs an offline analysis
pass, builds a visual score, then plays/render the visual performance.

Acceptance:

- Beat grid, section hints, energy curves, and optional stem data are available
  before playback starts.
- Scene transitions align to musical boundaries.
- Visual density changes are deliberate rather than continuously noisy.

### Export

The user renders a visual performance to image frames, video, or a frame stream
for external tools.

Acceptance:

- Output can be deterministic from a seed and analysis cache.
- The renderer supports fixed dimensions independent of terminal cell size.
- Frame export can run without a TUI.

## Product Principles

- Music first: Map musical roles to visual roles. Do not let arbitrary FFT bins
  control everything.
- State matters: Visual elements should accumulate, decay, orbit, split, merge,
  and transition over musical time.
- Taste over noise: Strong composition, restrained defaults, and controlled
  motion beat maximal twitchiness.
- Live and prepared modes share a contract: The visual layer consumes analysis
  frames whether they come from live capture or offline analysis.
- Terminal-native, not terminal-limited: The terminal renderer is the signature
  output, but the system should be able to target windows and export later.
- Ricer-friendly by design: Themes, presets, transparency, launch ergonomics,
  screenshots, and reproducible dotfile configs are core product concerns.
- Deterministic when needed: A performance should be reproducible from track,
  seed, settings, and analysis cache.

## Non-Goals

- Building a DAW.
- Replacing TouchDesigner, Notch, Resolume, or Synesthesia as a full show-control
  platform.
- Adding cloud-only AI dependencies to the default workflow.
- Shipping copyrighted preset packs or shader libraries without clear licensing.
- Optimizing for unsupported terminal protocols before the core experience works.

## Quality Bar

The visualizer is not acceptable if it looks like a standard equalizer with glow.
At least one flagship mode must demonstrate:

- A clear foreground, midground, and background.
- Energy-aware composition that still looks good during quiet passages.
- Beat accents that are visible but not obnoxious.
- Palette control with perceptual color interpolation.
- Smooth temporal behavior with no random jitter.
- A recognizable visual identity in screenshots.

## Interaction Model

Command-line flags select the initial mode, theme, device, band count, and frame
rate. Runtime keys select visual modes and themes without restarting.

Future interactions:

- `[` / `]`: decrease/increase intensity.
- `c`: cycle camera behavior.
- `p`: cycle palette relationship.
- `r`: reseed deterministic generative systems.
- `m`: switch between automatic score and manual mode.
- `e`: mark export start/stop in prepared mode.

## Performance Targets

Live terminal mode:

- 60 fps target on modern terminals with Kitty graphics.
- 30 fps graceful fallback on slower transports.
- Under 100 ms perceived audio-to-visual latency.
- No unbounded allocation in the render loop.

Prepared/export mode:

- Deterministic frame stepping.
- No dependency on wall-clock time during export.
- Analysis cache reused across runs.

## Platform Targets

Initial:

- Linux with PulseAudio or PipeWire Pulse compatibility.
- Kitty graphics protocol with halfblock fallback.

Future:

- File input on all platforms.
- Native window output through existing Scry window transport.
- Image/video frame export.
- Optional macOS/Windows live capture backends.

## Milestone Acceptance

M1: Documented alpha

- Current behavior documented.
- Architecture and analysis contracts written.
- Root README lists `scry-viz`.

M2: Analysis contract

- `AnalysisFrame` exists as a stable internal data structure.
- Live analysis emits bands, peaks, waveform, grouped energy, transient, and
  beat state through the contract.
- Tests cover silence, sine peaks, beat onset, and smoothing behavior.

M3: Visual score

- A score layer translates analysis into stable control signals.
- Existing modes consume score controls where useful.
- At least one mode has phrase-scale memory.

M4: Prepared track mode

- Audio file input.
- Offline analysis cache.
- Deterministic playback and export frame stepping.

M5: Flagship mode

- One mode demonstrates 2D/3D hybrid ambition with strong composition,
  section-aware transitions, and persistent musical state.
