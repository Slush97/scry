# Visual Language Spec

## Goal

`scry-viz` should look like a designed audiovisual instrument, not a demo of
frequency bins. The visual system should have composition, hierarchy, musical
memory, and restraint.

It should also look at home in a riced desktop. Visual modes need clean
silhouettes, configurable palettes, transparent-friendly contrast, and strong
screenshots at odd terminal sizes.

## Current Modes

| Mode | Role | Current Behavior |
|------|------|------------------|
| `silk` | Elegant spectrum surface | Layered smoothed ribbons, glow, reflection |
| `ridge` | Historical memory | Spectrum-history waterfall with occluding rows |
| `mandala` | Symmetry and ritual | Rotating petals and rings driven by band groups |
| `nova` | Bass impact and particles | Core pulse, orbiting sparks, beat shockwaves |
| `bars` | Familiar utility | Spectrum bars and peaks |
| `radial` | Circular spectrum | Mirrored spoke spectrum around a ring |
| `wave` | Oscilloscope identity | Waveform-led visual mode |

These modes should remain available, but at least one flagship mode should push
the system into section-aware, scored, persistent 2D/3D behavior.

## Visual Roles

Map musical roles to visual roles:

| Musical Signal | Visual Mapping |
|----------------|----------------|
| Bass/kick | Scale, gravity, shockwaves, camera pressure |
| Snare/clap | Rings, cuts, edge flashes, geometry splits |
| Hats/noise | Fine particles, shimmer, high-frequency texture |
| Vocal/lead | Foreground curves, typography, organic forms |
| Harmony | Background gradients, volumetric fields, slow palette drift |
| Silence | Negative space, slow decay, persistence |
| Build | Density ramp, camera compression, rising tension |
| Drop | Scene reveal, impact event, palette or topology shift |

## Composition Rules

Every flagship visual should define:

- Foreground: primary musical subject.
- Midground: secondary motion/detail.
- Background: atmospheric field or persistent memory.
- Impact layer: short-lived beat/transient accents.
- Transition layer: section and phrase changes.

Do not let every layer respond to every signal. This is the main cause of visual
mud. Each layer gets a narrow musical responsibility.

## Motion Rules

- Use attack/release envelopes, not raw values.
- Reserve instant changes for impact events.
- Let slow systems keep moving during quiet passages.
- Use camera motion sparingly; too much camera motion hides audio reactivity.
- Prefer phrase-length evolution over frame-length randomness.
- Randomness must be seeded and stable in prepared/export mode.

## Color Rules

- Palettes should contain contrast across hue and luminance.
- Bass should not always mean red and treble should not always mean blue.
- Use perceptual interpolation where available.
- Quiet sections can desaturate or reduce brightness, but should not disappear.
- Beat flashes should usually be alpha/brightness changes, not full-screen white.
- Presets should be able to match a user's desktop palette without rewriting a
  visual mode.

## 2D Grammar

High-quality 2D modes should use combinations of:

- Multipass feedback.
- Persistent trails and decays.
- Flow fields.
- Reaction/diffusion-like texture.
- Smoothed spectrum ribbons.
- Typographic overlays in prepared mode.
- Geometric symmetry with controlled imperfection.
- Distortion fields driven by waveform or transients.

Existing `PixelCanvas` modes can approximate many of these. Future GPU paths can
add feedback buffers and shader-style post-processing.

## 3D Grammar

High-quality 3D modes should use:

- Instanced particles or meshes.
- SDF/raymarch forms for fluid abstract geometry.
- Audio-driven materials, not only audio-driven positions.
- Camera choreography tied to bars and sections.
- Depth-aware 2D post-processing.
- Volumetric or fog-like atmosphere where performance allows.

The 3D layer should not be a separate product. It should consume the same
`AnalysisFrame` and `VisualScore` contracts as 2D modes.

## Scene Transitions

Transitions should be musically justified:

- Major section boundary: scene or topology transition.
- Downbeat: camera cut, palette lock, or impact reveal.
- Build: density ramp and compression.
- Breakdown: remove layers and expose waveform/foreground.
- Outro: decay and simplify.

Avoid changing scenes on a fixed timer without musical input.

## Mode Authoring Template

Each new mode should define:

```text
name:
visual premise:
musical role mapping:
foreground:
midground:
background:
impact events:
persistent state:
theme behavior:
performance risks:
fallback behavior:
rice knobs:
```

## Flagship Candidate: Conductor

The first barrier-pushing mode should combine current strengths:

- `ridge` history as a background memory surface.
- `nova` particles and shockwaves as impact layer.
- `silk` ribbons as foreground musical line.
- A camera/depth illusion using parallax and scale.
- Section-aware palette and density changes.

This can ship in terminal 2D before true 3D exists, while proving the score
architecture.
