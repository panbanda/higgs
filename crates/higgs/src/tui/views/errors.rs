use std::sync::Arc;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use super::format_time_ago;
use crate::metrics::MetricsStore;

pub fn draw(frame: &mut Frame, area: Rect, metrics: &Arc<MetricsStore>, scroll: usize) {
    let snap = metrics.snapshot();

    let now = std::time::Instant::now();
    let mut errors: Vec<_> = snap.iter().filter(|r| r.status >= 400).collect();
    errors.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    let header = Row::new(vec!["Age", "Model", "Provider", "Status", "Error"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = errors
        .iter()
        .skip(scroll)
        .take(100)
        .map(|r| {
            let error_preview = r
                .error_body
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(80)
                .collect::<String>()
                .replace('\n', " ");
            Row::new(vec![
                Cell::from(format_time_ago(now.duration_since(r.timestamp))),
                Cell::from(r.model.as_str()),
                Cell::from(r.provider.as_str()),
                Cell::from(r.status.to_string()).style(Style::default().fg(Color::Red)),
                Cell::from(error_preview),
            ])
        })
        .collect();

    let count = errors.len();
    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Min(20),
            Constraint::Length(12),
            Constraint::Length(6),
            Constraint::Min(30),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Errors ({count}) ")),
    );

    frame.render_widget(table, area);
    super::render_scrollbar(frame, area, count, scroll);
}
