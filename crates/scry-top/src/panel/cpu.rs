use ratatui::layout::Rect;
use ratatui::widgets::{Block, Borders};
use ratatui::Frame;

use scry_chart::prelude::{ChartState, ChartWidget, Charts};

use crate::data::SystemSnapshot;
use crate::panel::Panel;
use crate::ring::RingBuffer;
use crate::theme::{cpu_core_colors, monitor_theme};

const HISTORY: usize = 300;

pub struct CpuPanel {
    cores: Vec<RingBuffer>,
    chart: Option<scry_chart::prelude::Chart>,
    state: ChartState,
}

impl CpuPanel {
    pub fn new() -> Self {
        Self {
            cores: Vec::new(),
            chart: None,
            state: ChartState::auto(),
        }
    }
}

impl Panel for CpuPanel {
    fn update(&mut self, snap: &SystemSnapshot) {
        if self.cores.len() != snap.cpu_per_core.len() {
            self.cores = (0..snap.cpu_per_core.len())
                .map(|_| RingBuffer::new(HISTORY))
                .collect();
        }

        for (i, &usage) in snap.cpu_per_core.iter().enumerate() {
            self.cores[i].push(usage as f64);
        }

        // Rebuild chart on new data
        let n_cores = self.cores.len();
        if n_cores == 0 {
            return;
        }
        let max_series = 16.min(n_cores);
        let first_data = self.cores[0].as_vec();
        if first_data.len() < 2 {
            return;
        }

        let colors = cpu_core_colors(max_series);
        let mut builder = Charts::area(&first_data);
        for i in 1..max_series {
            let data = self.cores[i].as_vec();
            builder = builder.add_named_series(format!("Core {i}"), &data);
        }
        self.chart = Some(
            builder
                .theme(monitor_theme().with_palette(colors))
                .y_range(0.0, 100.0)
                .build(),
        );
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        let n_cores = self.cores.len();
        let avg: f64 = if n_cores > 0 {
            self.cores
                .iter()
                .map(|c| c.as_vec().last().copied().unwrap_or(0.0))
                .sum::<f64>()
                / n_cores as f64
        } else {
            0.0
        };

        let block = Block::default()
            .title(format!(" CPU ({n_cores} cores, {avg:.0}%) "))
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
