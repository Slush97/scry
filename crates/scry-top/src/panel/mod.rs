pub mod cpu;
pub mod disk;
pub mod header;
pub mod memory;
pub mod network;
pub mod process;

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::data::SystemSnapshot;

pub trait Panel {
    fn update(&mut self, snap: &SystemSnapshot);
    fn render(&mut self, frame: &mut Frame, area: Rect);
    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    fn cleanup(&mut self);
}
