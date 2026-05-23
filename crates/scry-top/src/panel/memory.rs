use ratatui::layout::Rect;
use ratatui::widgets::{Block, Borders};
use ratatui::Frame;

use scry_chart::prelude::{ChartState, ChartWidget, Charts};

use crate::data::SystemSnapshot;
use crate::panel::Panel;
use crate::ring::RingBuffer;
use crate::theme::{memory_colors, monitor_theme};

const HISTORY: usize = 300;

pub struct MemoryPanel {
    ram: RingBuffer,
    swap: RingBuffer,
    chart: Option<scry_chart::prelude::Chart>,
    state: ChartState,
    ram_pct: f64,
    swap_pct: f64,
}

impl MemoryPanel {
    pub fn new() -> Self {
        Self {
            ram: RingBuffer::new(HISTORY),
            swap: RingBuffer::new(HISTORY),
            chart: None,
            state: ChartState::auto(),
            ram_pct: 0.0,
            swap_pct: 0.0,
        }
    }
}

impl Panel for MemoryPanel {
    fn update(&mut self, snap: &SystemSnapshot) {
        self.ram_pct = if snap.mem_total > 0 {
            snap.mem_used as f64 / snap.mem_total as f64 * 100.0
        } else {
            0.0
        };
        self.swap_pct = if snap.swap_total > 0 {
            snap.swap_used as f64 / snap.swap_total as f64 * 100.0
        } else {
            0.0
        };
        self.ram.push(self.ram_pct);
        self.swap.push(self.swap_pct);

        let ram_data = self.ram.as_vec();
        let swap_data = self.swap.as_vec();
        if ram_data.len() < 2 {
            return;
        }

        let (ram_color, swap_color) = memory_colors();
        self.chart = Some(
            Charts::area(&ram_data)
                .add_named_series("Swap", &swap_data)
                .theme(monitor_theme().with_palette(vec![ram_color, swap_color]))
                .y_range(0.0, 100.0)
                .build(),
        );
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(format!(
                " Mem {:.0}% | Swap {:.0}% ",
                self.ram_pct, self.swap_pct
            ))
            .borders(Borders::ALL)
            .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray));

        if let Some(chart) = &self.chart {
            let inner = block.inner(area);
            frame.render_widget(block, area);
            frame.render_stateful_widget(ChartWidget::new(chart), inner, &mut self.state);
        } else {
            frame.render_widget(block, area);
        }
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.state.flush()?;
        Ok(())
    }

    fn cleanup(&mut self) {
        self.state.cleanup();
    }
}
