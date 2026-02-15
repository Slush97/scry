//! SVG widget for Ratatui — renders SVG images via the pixel transport layer.
//!
//! [`SvgWidget`] is a [`StatefulWidget`] that takes a reference to a parsed
//! [`SvgImage`] and renders it through the same protocol pipeline as
//! [`PixelCanvasWidget`] (Kitty, Sixel, or Halfblock fallback).
//!
//! # Example
//!
//! ```no_run
//! use scry_engine::svg::SvgImage;
//! use scry_engine::widget::{SvgWidget, PixelCanvasState};
//!
//! let svg = SvgImage::from_str("<svg xmlns='http://www.w3.org/2000/svg'/>").unwrap();
//! let widget = SvgWidget::new(&svg);
//!
//! # let area = ratatui::layout::Rect::default();
//! # let mut state: PixelCanvasState = todo!();
//! # let frame: &mut ratatui::Frame = todo!();
//! frame.render_stateful_widget(widget, area, &mut state);
//! state.flush().unwrap();
//! ```

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::StatefulWidget;

use super::widget_impl::PixelCanvasState;
use crate::svg::SvgImage;

/// A Ratatui widget that renders an SVG image via a graphics protocol.
///
/// This widget takes a reference to a parsed [`SvgImage`] and renders it
/// through the existing pixel transport infrastructure. The SVG is scaled
/// to fit the widget area while preserving aspect ratio.
pub struct SvgWidget<'a> {
    svg: &'a SvgImage,
    z_index: i32,
    /// If true, stretch to fill (don't preserve aspect ratio).
    stretch: bool,
}

impl<'a> SvgWidget<'a> {
    /// Create a new SVG widget from a parsed SVG image.
    pub const fn new(svg: &'a SvgImage) -> Self {
        Self {
            svg,
            z_index: -1,
            stretch: false,
        }
    }

    /// Set the z-index for Kitty protocol layering.
    pub const fn z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }

    /// Stretch the SVG to fill the entire widget area (ignore aspect ratio).
    pub const fn stretch(mut self) -> Self {
        self.stretch = true;
        self
    }
}

impl StatefulWidget for SvgWidget<'_> {
    type State = PixelCanvasState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        let font_size = state.font_size();
        let px_width = u32::from(area.width) * u32::from(font_size.width);
        let px_height = u32::from(area.height) * u32::from(font_size.height);

        if px_width == 0 || px_height == 0 {
            return;
        }

        // Render SVG to a pixmap
        let pixmap = if self.stretch {
            match self.svg.render_fill(px_width, px_height) {
                Ok(p) => p,
                Err(_) => return,
            }
        } else {
            match self.svg.render(px_width, px_height) {
                Ok(p) => p,
                Err(_) => return,
            }
        };

        // Store the pixmap for deferred transmission via flush()
        // This works for all backends (Kitty, Sixel, Halfblock) through
        // the same pipeline as PixelCanvasWidget.
        state.store_raw_pixmap(pixmap, area, self.z_index);

        // Fill the area with spaces so ratatui doesn't overwrite the image
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(' ');
                }
            }
        }
    }
}
