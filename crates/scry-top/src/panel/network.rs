use ratatui::layout::Rect;
use ratatui::widgets::{Block, Borders};
use ratatui::Frame;

use scry_chart::prelude::{ChartState, ChartWidget, Charts};

use crate::data::SystemSnapshot;
use crate::panel::Panel;
use crate::ring::RingBuffer;
use crate::theme::{monitor_theme, network_colors};

const HISTORY: usize = 300;

pub struct NetworkPanel {
    rx: RingBuffer,
    tx: RingBuffer,
    chart: Option<scry_chart::prelude::Chart>,
    state: ChartState,
    last_rx_rate: f64,
    last_tx_rate: f64,
}

impl NetworkPanel {
    pub fn new() -> Self {
        Self {
            rx: RingBuffer::new(HISTORY),
            tx: RingBuffer::new(HISTORY),
            chart: None,
            state: ChartState::auto(),
            last_rx_rate: 0.0,
            last_tx_rate: 0.0,
        }
    }
}

fn format_bytes(bytes: f64) -> String {
    if bytes >= 1_073_741_824.0 {
        format!("{:.1} GB/s", bytes / 1_073_741_824.0)
    } else if bytes >= 1_048_576.0 {
        format!("{:.1} MB/s", bytes / 1_048_576.0)
    } else if bytes >= 1024.0 {
        format!("{:.1} KB/s", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B/s")
    }
}

impl Panel for NetworkPanel {
    fn update(&mut self, snap: &SystemSnapshot) {
        let total_rx: f64 = snap.networks.iter().map(|n| n.rx_bytes_sec).sum();
        let total_tx: f64 = snap.networks.iter().map(|n| n.tx_bytes_sec).sum();
        self.rx.push(total_rx);
        self.tx.push(total_tx);
        self.last_rx_rate = total_rx;
        self.last_tx_rate = total_tx;

        let rx_data = self.rx.as_vec();
        let tx_data = self.tx.as_vec();
        if rx_data.len() < 2 {
            return;
        }

        let (rx_color, tx_color) = network_colors();
        self.chart = Some(
            Charts::line(&rx_data)
                .add_named_series("TX", &tx_data)
                .smooth()
                .theme(monitor_theme().with_palette(vec![rx_color, tx_color]))
                .build(),
        );
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(format!(
                " Net ↓{} ↑{} ",
                format_bytes(self.last_rx_rate),
                format_bytes(self.last_tx_rate)
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
