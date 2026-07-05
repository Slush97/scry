# scry-viz Roadmap

## Phase 0: Documented Alpha

Status: current scaffold.

Deliverables:

- Crate README.
- Product, architecture, audio, visual language, roadmap, and research docs.
- Aesthetic integration spec for ricer/dotfiles workflows.
- Root README workspace entry.

Acceptance:

- A new contributor can understand what exists and what the next architecture is.
- The current live visualizer remains untouched.

## Phase 1: Stabilize the Live Baseline

Status: implemented for the live baseline.

Deliverables:

- Move the current `Spectrum` output behind an `AnalysisFrame` contract.
- Add band group helpers for bass, low-mid, high-mid, and treble.
- Add a visual-only/no-HUD launch path for clean terminal-rice usage.
- Add tests for beat gating and no-signal behavior.
- Add a render fixture that builds each mode from a fixed analysis frame.

Acceptance:

- Existing modes render through the new contract.
- Live mode still starts and exits cleanly.
- Analyzer tests are deterministic.

## Phase 2: Visual Score Layer

Deliverables:

- Add `VisualScore`.
- Add a `Conductor` that maps analysis frames to stable controls:
  intensity, density, impact, flow, brightness, palette shift, and camera pressure.
- Convert `nova` and `silk` to consume score controls for large-scale behavior.
- Add seeded randomness to score-driven events.

Acceptance:

- One beat produces one visible impact event.
- Quiet passages remain composed.
- Visual density changes are smooth and intentional.

## Phase 3: Flagship 2D Mode

Deliverables:

- Add `conductor` mode.
- Combine persistent history, foreground ribbon, impact particles, and transition
  events.
- Add mode-specific docs using the visual authoring template.

Acceptance:

- Screenshots clearly differ from ordinary equalizer visuals.
- The mode looks good on silence, ambient tracks, and percussive tracks.
- It can hold 60 fps on a modern Kitty terminal at normal sizes.

## Phase 4: Prepared Track Mode

Deliverables:

- Add file input.
- Add offline analysis pass.
- Add analysis cache.
- Add deterministic playback clock.
- Add section and beat-grid hints.

Acceptance:

- The same track and seed produce the same performance.
- Scene changes align to musical boundaries where detected.
- Export mode can render without terminal event polling.

## Phase 5: 3D and GPU Expansion

Deliverables:

- Define a `RenderTarget` abstraction for terminal, pixmap, window, and export.
- Add a 3D visual mode using existing Scry 3D/SDF/GPU capabilities where possible.
- Add depth-aware or parallax terminal fallback.
- Add profiling and quality tiers.

Acceptance:

- 3D mode uses the same analysis and score contracts.
- Lower-quality fallback remains coherent in halfblock output.
- Quality tiers trade particle count/effects for frame rate predictably.

## Phase 6: Export and Integration

Deliverables:

- Frame export.
- Optional video pipeline integration.
- Preset/config file format.
- Mode/theme registry.
- Palette/config loading for dotfiles workflows.
- Optional external analysis/stem hooks.

Acceptance:

- Users can render deterministic frames from an audio file.
- Presets can be shared without recompiling.
- External analysis is optional, not required for live mode.

## Near-Term Task List

1. Add `score` module with `VisualScore` and `Conductor`.
2. Add unit tests around score event emission.
3. Add `conductor` mode as the first flagship composition mode.
4. Add crate-level docs for CLI examples and troubleshooting capture devices.

## Key Risks

| Risk | Mitigation |
|------|------------|
| Terminal transport becomes the bottleneck | Keep quality tiers and shape budgets |
| Visuals become noisy | Route major behavior through `VisualScore` |
| Beat detection is unreliable live | Treat beat phase as optional and use confidence |
| Offline mode grows too large | Keep live path dependency-light |
| 3D overreaches before score works | Prove score with 2D first |
