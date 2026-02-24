use std::sync::Arc;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use super::{format_duration, format_tokens};
use crate::metrics::MetricsStore;

pub fn draw(frame: &mut Frame, area: Rect, metrics: &Arc<MetricsStore>, scroll: usize) {
    let snap = metrics.snapshot();
    let groups = MetricsStore::group_by(&snap, |r| r.provider.clone());

    let header = Row::new(vec![
        "Provider", "Reqs", "In", "Out", "Avg/Req", "P50", "P95", "Errs",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let mut names: Vec<&String> = groups.keys().collect();
    names.sort();

    let total = names.len();

    let rows: Vec<Row> = names
        .iter()
        .skip(scroll)
        .filter_map(|name| {
            let records = groups.get(*name)?;
            let count = u64::try_from(records.len()).unwrap_or(0);
            let input: u64 = records.iter().map(|r| r.input_tokens).sum();
            let output: u64 = records.iter().map(|r| r.output_tokens).sum();
            let durations: Vec<_> = records.iter().map(|r| r.duration).collect();
            let p50 = MetricsStore::duration_percentile(&durations, 50);
            let p95 = MetricsStore::duration_percentile(&durations, 95);
            let errors: u64 =
                u64::try_from(records.iter().filter(|r| r.status >= 400).count()).unwrap_or(0);
            let error_style = if errors > 0 {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Some(Row::new(vec![
                Cell::from(name.as_str()).style(Style::default().fg(Color::White)),
                Cell::from(format_tokens(count)),
                Cell::from(format_tokens(input)).style(Style::default().fg(Color::Cyan)),
                Cell::from(format_tokens(output)).style(Style::default().fg(Color::Green)),
                Cell::from(format_tokens((input + output) / count.max(1)))
                    .style(Style::default().fg(Color::White)),
                Cell::from(format_duration(p50)),
                Cell::from(format_duration(p95)),
                Cell::from(format_tokens(errors)).style(error_style),
            ]))
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Min(15),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Providers "));

    frame.render_widget(table, area);
    super::render_scrollbar(frame, area, total, scroll);
}
