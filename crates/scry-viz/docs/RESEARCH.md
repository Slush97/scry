# Music Visualizer Research Notes

## Landscape

High-quality music visualizers fall into several families:

| Family | References | What Matters |
|--------|------------|--------------|
| MilkDrop lineage | projectM, Butterchurn | Presets, shader feedback, beat response, smooth transitions |
| VJ/live tools | Resolume, Synesthesia, Magic Music Visuals | Scene libraries, MIDI/OSC, shader import, output routing |
| Pro show systems | TouchDesigner, Notch | Node graphs, sensors, GPU particles, volumetrics, media-server workflows |
| Live coding/shaders | Hydra, ISF, Shadertoy | Multipass buffers, remixable shader grammar, fast iteration |
| Web/audio APIs | Web Audio, AudioWorklet, WebGPU | Portable real-time analysis and modern GPU compute |

## References

- projectM: https://github.com/projectM-visualizer/projectm
- Butterchurn: https://github.com/jberg/butterchurn
- Synesthesia: https://synesthesia.live/
- Magic Music Visuals: https://magicmusicvisuals.com/
- Resolume: https://resolume.com/software
- TouchDesigner: https://derivative.ca/
- Notch: https://www.notch.one/
- Hydra: https://github.com/hydra-synth/hydra
- ISF: https://isf.video/
- MDN AnalyserNode: https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode
- MDN AudioWorklet: https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet
- MDN WebGPU: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- Meyda: https://meyda.js.org/
- Essentia.js: https://github.com/MTG/essentia.js
- Demucs: https://github.com/facebookresearch/demucs

## Lessons

### Preset Ecosystems Matter

MilkDrop stayed relevant because it separated the engine from authored visual
presets. `scry-viz` should eventually support shareable presets or mode configs,
even if the first version keeps modes compiled into Rust.

### Shaders Are Not Enough

Shader systems can look stunning, but music connection often remains shallow.
The barrier to push is not just higher rendering complexity. It is stronger
music interpretation feeding the visual system.

### Pro Tools Are Conductors

TouchDesigner and Notch succeed because they let artists route signals,
build stateful systems, and integrate with show control. `scry-viz` should not
copy their whole UI, but it should copy the architecture lesson: separate signal
analysis, control mapping, visual systems, and outputs.

### Live and Offline Are Different Products

Live mode needs latency and resilience. Offline/prepared mode can analyze more
deeply and render deterministically. They should share contracts, not code paths
that pretend both timing models are the same.

### Stem Awareness Is a Quality Leap

Full-mix FFT makes every element fight every other element. Stem-aware or
role-aware mapping lets drums, bass, vocals, and harmony control different
visual layers. Even approximate live role detection is better than one global
spectrum controlling everything.

## Strategic Choice

The best path for Scry is:

1. Keep the terminal-native live visualizer lightweight and responsive.
2. Formalize analysis and visual score contracts.
3. Build one flagship score-driven mode in 2D.
4. Add prepared-track analysis and deterministic export.
5. Expand into 3D/GPU once the music-to-visual mapping is already strong.

