use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Cell, Row, Table};
use ratatui::Frame;

use crate::data::{ProcessInfo, SystemSnapshot};
use crate::panel::Panel;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortField {
    Cpu,
    Mem,
    Pid,
    Name,
}

impl SortField {
    pub fn next(self) -> Self {
        match self {
            Self::Cpu => Self::Mem,
            Self::Mem => Self::Pid,
            Self::Pid => Self::Name,
            Self::Name => Self::Cpu,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU%",
            Self::Mem => "MEM%",
            Self::Pid => "PID",
            Self::Name => "NAME",
        }
    }
}

pub struct ProcessPanel {
    processes: Vec<ProcessInfo>,
    sort_by: SortField,
}

impl ProcessPanel {
    pub fn new() -> Self {
        Self {
            processes: Vec::new(),
            sort_by: SortField::Cpu,
        }
    }

    pub fn cycle_sort(&mut self) {
        self.sort_by = self.sort_by.next();
    }
}

impl Panel for ProcessPanel {
    fn update(&mut self, snap: &SystemSnapshot) {
        self.processes = snap
            .processes
            .iter()
            .map(|p| ProcessInfo {
                pid: p.pid,
                name: p.name.clone(),
                cpu_pct: p.cpu_pct,
                mem_pct: p.mem_pct,
                status: p.status.clone(),
            })
            .collect();

        match self.sort_by {
            SortField::Cpu => self.processes.sort_by(|a, b| {
                b.cpu_pct
                    .partial_cmp(&a.cpu_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            SortField::Mem => self.processes.sort_by(|a, b| {
                b.mem_pct
                    .partial_cmp(&a.mem_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            SortField::Pid => self.processes.sort_by_key(|p| p.pid),
            SortField::Name => self.processes.sort_by(|a, b| a.name.cmp(&b.name)),
        }
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        let header_cells = ["PID", "NAME", "CPU%", "MEM%", "STATE"].iter().map(|&h| {
            let style = if h == self.sort_by.label() {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Cell::from(h).style(style)
        });
        let header = Row::new(header_cells).height(1);

        let rows = self
            .processes
            .iter()
            .take(area.height.saturating_sub(4) as usize)
            .map(|p| {
                let cpu_color = if p.cpu_pct > 80.0 {
                    Color::Red
                } else if p.cpu_pct > 30.0 {
                    Color::Yellow
                } else {
                    Color::White
                };

                Row::new(vec![
                    Cell::from(format!("{}", p.pid)).style(Style::default().fg(Color::DarkGray)),
                    Cell::from(truncate_name(&p.name, 20)).style(Style::default().fg(Color::White)),
                    Cell::from(format!("{:.1}", p.cpu_pct)).style(Style::default().fg(cpu_color)),
                    Cell::from(format!("{:.1}", p.mem_pct))
                        .style(Style::default().fg(Color::White)),
                    Cell::from(p.status.as_str()).style(Style::default().fg(Color::DarkGray)),
                ])
            });

        let widths = [
            ratatui::layout::Constraint::Length(8),
            ratatui::layout::Constraint::Min(12),
            ratatui::layout::Constraint::Length(8),
            ratatui::layout::Constraint::Length(8),
            ratatui::layout::Constraint::Length(10),
        ];

        let table = Table::new(rows, widths).header(header).block(
            Block::default()
                .title(format!(
                    " Processes (sort: {}) — [s] cycle ",
                    self.sort_by.label()
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        frame.render_widget(table, area);
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn cleanup(&mut self) {}
}

fn truncate_name(name: &str, max: usize) -> String {
    if name.len() <= max {
        name.to_string()
    } else {
        format!("{}…", &name[..max - 1])
    }
}
