use ratatui::layout::Rect;
use ratatui::widgets::{Block, Borders};
use ratatui::Frame;

use scry_chart::prelude::{ChartState, ChartWidget, Charts};

use crate::data::{DiskInfo, SystemSnapshot};
use crate::panel::Panel;
use crate::theme::monitor_theme;

pub struct DiskPanel {
    disks: Vec<DiskInfo>,
    chart: Option<scry_chart::prelude::Chart>,
    state: ChartState,
}

impl DiskPanel {
    pub fn new() -> Self {
        Self {
            disks: Vec::new(),
            chart: None,
            state: ChartState::auto(),
        }
    }
}

impl Panel for DiskPanel {
    fn update(&mut self, snap: &SystemSnapshot) {
        self.disks = snap
            .disks
            .iter()
            .map(|d| DiskInfo {
                name: d.name.clone(),
                used_pct: d.used_pct,
            })
            .collect();

        if self.disks.is_empty() {
            return;
        }

        let labels: Vec<String> = self.disks.iter().map(|d| d.name.clone()).collect();
        let values: Vec<f64> = self.disks.iter().map(|d| d.used_pct).collect();
        self.chart = Some(
            Charts::bar(labels, &values)
                .theme(monitor_theme())
                .y_range(0.0, 100.0)
                .build(),
        );
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Disk ")
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
