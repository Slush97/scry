// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ratatui widget integration for rendering Mermaid diagrams.
//!
//! Provides [`MermaidWidget`] and re-exports [`PixelCanvasState`] as
//! [`MermaidState`] for managing protocol state across frames.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::StatefulWidget;

use scry_engine::prelude::{PixelCanvasState, PixelCanvasWidget};
use scry_engine::transport::backend::FontSize;
use scry_engine::transport::{self, ProtocolBackend};

use crate::theme::MermaidTheme;
use crate::Mermaid;

// ---------------------------------------------------------------------------
// MermaidState
// ---------------------------------------------------------------------------

/// Persistent state for [`MermaidWidget`] across render frames.
///
/// Wraps [`PixelCanvasState`] to manage the graphics protocol connection.
pub struct MermaidState {
    pixel_state: PixelCanvasState,
}

impl MermaidState {
    /// Create state from a pixel canvas state.
    #[must_use]
    pub fn new(pixel_state: PixelCanvasState) -> Self {
        Self { pixel_state }
    }

    /// Auto-detect the best terminal graphics protocol.
    ///
    /// This is the recommended way to create state:
    /// ```ignore
    /// let mut state = MermaidState::auto();
    /// ```
    #[must_use]
    pub fn auto() -> Self {
        use scry_engine::prelude::{Picker, ProtocolKind};

        let picker = Picker::detect();
        let backend: Box<dyn ProtocolBackend> = match picker.protocol() {
            ProtocolKind::Kitty => {
                Box::new(transport::kitty::KittyBackend::new(picker.font_size()))
            }
            _ => Box::new(transport::halfblock::HalfblockBackend::new()),
        };
        Self::new(PixelCanvasState::new(backend, picker.font_size()))
    }

    /// Get the font size for cell-to-pixel calculations.
    #[must_use]
    pub fn font_size(&self) -> FontSize {
        self.pixel_state.font_size()
    }

    /// Transmit any pending image to the terminal.
    ///
    /// Call this **after** `terminal.draw()` to avoid flicker.
    pub fn flush(&mut self) -> Result<(), scry_engine::PixelCanvasError> {
        self.pixel_state.flush()
    }

    /// Clean up resources.
    pub fn cleanup(&mut self) {
        self.pixel_state.cleanup();
    }
}

impl std::fmt::Debug for MermaidState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MermaidState")
            .field("pixel_state", &self.pixel_state)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MermaidWidget
// ---------------------------------------------------------------------------

/// A ratatui widget that renders a Mermaid diagram as pixel graphics.
///
/// # Example
///
/// ```ignore
/// use scry_mermaid::prelude::*;
/// use scry_mermaid::widget::{MermaidWidget, MermaidState};
///
/// let diagram = Mermaid::parse("graph TD\n    A --> B").unwrap();
/// let mut state = MermaidState::auto();
///
/// frame.render_stateful_widget(
///     MermaidWidget::new(&diagram),
///     area,
///     &mut state,
/// );
/// state.flush().unwrap();
/// ```
pub struct MermaidWidget<'a> {
    diagram: &'a Mermaid,
    theme_override: Option<MermaidTheme>,
    z_index: i32,
}

impl<'a> MermaidWidget<'a> {
    /// Create a widget referencing a parsed diagram.
    #[must_use]
    pub fn new(diagram: &'a Mermaid) -> Self {
        Self {
            diagram,
            theme_override: None,
            z_index: -1,
        }
    }

    /// Override the theme for this render pass.
    #[must_use]
    pub fn theme(mut self, theme: MermaidTheme) -> Self {
        self.theme_override = Some(theme);
        self
    }

    /// Set the z-index for Kitty layering.
    #[must_use]
    pub fn z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }
}

impl StatefulWidget for MermaidWidget<'_> {
    type State = MermaidState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        if area.width < 2 || area.height < 2 {
            return;
        }

        let font = state.font_size();
        let pixel_w = u32::from(area.width) * u32::from(font.width);
        let pixel_h = u32::from(area.height) * u32::from(font.height);

        // If a theme override was provided, render with a temporary Mermaid
        // that has the overridden theme applied. Otherwise use the diagram as-is.
        let rendered = if let Some(theme) = &self.theme_override {
            let m = self.diagram.clone_with_theme(theme.clone());
            m.render(pixel_w, pixel_h)
        } else {
            self.diagram.render(pixel_w, pixel_h)
        };

        let pixel_widget = PixelCanvasWidget::new(rendered.canvas).z_index(self.z_index);
        pixel_widget.render(area, buf, &mut state.pixel_state);
    }
}
