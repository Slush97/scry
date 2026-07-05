# Aesthetic Integration Spec

## Positioning

`scry-viz` should be the ricer's best friend for music visualization: a tool
that can disappear into a carefully tuned desktop, terminal, stream layout, or
wallpaper setup while still looking technically serious.

The app should not feel like a generic media-player plugin. It should feel like
an aesthetic system component: themeable, scriptable, deterministic, and easy to
compose with the rest of a user's environment.

## Design Promise

For desktop ricers and visual power users, `scry-viz` should provide:

- Clean terminal-native visuals that respect transparency and color schemes.
- Presets that can match a dotfiles theme without code changes.
- Low-friction CLI control for launchers, keybinds, scripts, and status bars.
- Export and frame-output paths for wallpapers, streams, and lock screens.
- A visual language that rewards tuning without requiring a heavyweight VJ app.

## Aesthetic Personas

| Persona | Need |
|---------|------|
| Terminal ricer | Looks good in Kitty/WezTerm/Alacritty-style workflows with transparency |
| Wayland compositor user | Wants keybind-launchable visual surfaces and predictable windows |
| Streamer | Wants a reactive visual layer that can be captured cleanly |
| Dotfiles author | Wants config files, named themes, screenshots, and reproducible presets |
| Music listener | Wants something beautiful with no manual setup |

## Integration Surfaces

### Terminal Surface

Current default. This should remain the signature Scry surface.

Requirements:

- Works with transparent terminal backgrounds.
- Does not depend on full-screen text UI chrome.
- HUD can be hidden.
- Palette choices avoid muddy alpha on dark and translucent backgrounds.
- Halfblock fallback remains usable for non-Kitty terminals.

### Floating Window Surface

Future target using Scry's native window transport.

Requirements:

- Borderless mode.
- Always-on-top optional.
- Fixed aspect ratio optional.
- Transparent background desirable where supported.
- Deterministic size for screenshots and screen capture.

### Wallpaper/Background Surface

Future target for desktop background workflows.

Requirements:

- Long-running low-power mode.
- Slow visual drift during silence.
- No seizure-like full-screen flashes by default.
- Configurable fps and quality tier.

### Export Surface

Future target for videos, clips, and generated assets.

Requirements:

- Deterministic seeded output.
- Fixed output dimensions.
- Analysis cache reuse.
- Transparent-background frame option.

## Presets

A preset should capture the user-facing visual identity:

```toml
name = "neon-cathedral"
mode = "conductor"
theme = "neon"
seed = 1337
fps = 60
quality = "high"

[visual]
intensity = 0.82
density = 0.70
motion = 0.55
glow = 0.76
symmetry = 0.40

[palette]
background = "#080810"
accent = "#ff3cdc"
stops = ["#00f0ff", "#5078ff", "#c83cff", "#ff3cb4"]
alpha = 0.92
```

Presets should be stable, shareable, and small enough to live in dotfiles.

## Theme Ingestion

Future theme helpers should make common ricing workflows easy:

- Load palette from a TOML/JSON file.
- Accept direct CLI palette overrides.
- Import terminal color slots where possible.
- Import a generated palette file from external tools.
- Save the active mode/theme/seed as a preset.

Avoid hard-coding integration with a single theme ecosystem at first. Start with
plain config files and predictable CLI flags.

## CLI Direction

Future flags:

```bash
scry-viz --preset neon-cathedral
scry-viz --no-hud --transparent
scry-viz --mode conductor --seed 1337 --quality high
scry-viz --palette ~/.config/scry-viz/palettes/catppuccin-mocha.toml
scry-viz export song.wav --preset neon-cathedral --size 1920x1080 --fps 60
```

The CLI should be scriptable and composable. Every interactive choice should
eventually have a config or CLI equivalent.

## Default Aesthetic Bar

Defaults should satisfy a user who cares about their desktop:

- No bulky borders.
- HUD off or minimal when requested.
- Strong contrast but not one-note color.
- Transparent-friendly alpha.
- Slow idle motion.
- Clean screenshots.
- Sensible behavior on ultrawide and tiny terminal panes.

## Acceptance Criteria

The ricer workflow is successful when:

- A user can launch `scry-viz` from a keybind with no HUD.
- A preset can match a desktop theme without rebuilding.
- The visualizer looks intentional over a transparent terminal.
- The same preset and seed produce reproducible screenshots.
- Exported frames match the live composition closely.

