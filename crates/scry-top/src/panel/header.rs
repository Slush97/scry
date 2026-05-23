use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::data::SystemSnapshot;
use crate::panel::Panel;

pub struct HeaderPanel {
    hostname: String,
    uptime_secs: u64,
    load_avg: [f64; 3],
    cpu_global: f32,
    mem_pct: f32,
}

impl HeaderPanel {
    pub fn new() -> Self {
        Self {
            hostname: String::new(),
            uptime_secs: 0,
            load_avg: [0.0; 3],
            cpu_global: 0.0,
            mem_pct: 0.0,
        }
    }
}

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}

impl Panel for HeaderPanel {
    fn update(&mut self, snap: &SystemSnapshot) {
        self.hostname = snap.hostname.clone();
        self.uptime_secs = snap.uptime_secs;
        self.load_avg = snap.load_avg;
        self.cpu_global = snap.cpu_global;
        self.mem_pct = if snap.mem_total > 0 {
            snap.mem_used as f32 / snap.mem_total as f32 * 100.0
        } else {
            0.0
        };
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        let dim = Style::default().fg(Color::DarkGray);
        let bright = Style::default().fg(Color::White);
        let accent = Style::default().fg(Color::Cyan);

        let line = Line::from(vec![
            Span::styled(" scry-top", accent),
            Span::styled(" │ ", dim),
            Span::styled(&self.hostname, bright),
            Span::styled(" │ up ", dim),
            Span::styled(format_uptime(self.uptime_secs), bright),
            Span::styled(" │ load ", dim),
            Span::styled(
                format!(
                    "{:.2} {:.2} {:.2}",
                    self.load_avg[0], self.load_avg[1], self.load_avg[2]
                ),
                bright,
            ),
            Span::styled(" │ cpu ", dim),
            Span::styled(
                format!("{:.0}%", self.cpu_global),
                cpu_color(self.cpu_global),
            ),
            Span::styled(" │ mem ", dim),
            Span::styled(format!("{:.0}%", self.mem_pct), cpu_color(self.mem_pct)),
        ]);

        frame.render_widget(Paragraph::new(line), area);
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn cleanup(&mut self) {}
}

fn cpu_color(pct: f32) -> Style {
    let color = if pct > 80.0 {
        Color::Red
    } else if pct > 50.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    Style::default().fg(color)
}
