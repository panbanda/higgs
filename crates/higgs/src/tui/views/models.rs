use std::sync::Arc;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use super::{format_duration, format_tokens};
use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};

/// Builds model-summary rows from a snapshot. Shared by the Models tab and the
/// overview Token Usage panel.
pub fn model_table(snap: &[RequestRecord], title: String, skip: usize) -> (Table<'static>, usize) {
    let groups = MetricsStore::group_by(snap, |r| r.model.clone());

    let header = Row::new(vec![
        "", "Model", "Reqs", "In", "Out", "Avg/Req", "P50", "P95", "Errs",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let mut model_names: Vec<String> = groups.keys().cloned().collect();
    model_names.sort();
    let total = model_names.len();

    let rows: Vec<Row> = model_names
        .iter()
        .skip(skip)
        .filter_map(|model| {
            let records = groups.get(model)?;
            let count = u64::try_from(records.len()).unwrap_or(0);
            let input: u64 = records.iter().map(|r| r.input_tokens).sum();
            let output: u64 = records.iter().map(|r| r.output_tokens).sum();
            let durations: Vec<_> = records.iter().map(|r| r.duration).collect();
            let p50 = MetricsStore::duration_percentile(&durations, 50);
            let p95 = MetricsStore::duration_percentile(&durations, 95);
            let errors: u64 =
                u64::try_from(records.iter().filter(|r| r.status >= 400).count()).unwrap_or(0);
            let routing_method = if records
                .iter()
                .any(|r| r.routing_method == RoutingMethod::Higgs)
            {
                RoutingMethod::Higgs
            } else if records
                .iter()
                .any(|r| r.routing_method == RoutingMethod::Auto)
            {
                RoutingMethod::Auto
            } else if records
                .iter()
                .any(|r| r.routing_method == RoutingMethod::Pattern)
            {
                RoutingMethod::Pattern
            } else {
                RoutingMethod::Default
            };

            let (indicator, indicator_style) = match routing_method {
                RoutingMethod::Pattern => ("PTN", Style::default().fg(Color::Cyan)),
                RoutingMethod::Auto => ("AUT", Style::default().fg(Color::Yellow)),
                RoutingMethod::Default => ("DEF", Style::default().fg(Color::DarkGray)),
                RoutingMethod::Higgs => ("HGS", Style::default().fg(Color::Magenta)),
            };

            let error_style = if errors > 0 {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            Some(Row::new(vec![
                Cell::from(indicator).style(indicator_style),
                Cell::from(model.clone()).style(Style::default().fg(Color::White)),
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
            Constraint::Length(3),
            Constraint::Min(25),
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
    .block(Block::default().borders(Borders::ALL).title(title));

    (table, total)
}

pub fn draw(frame: &mut Frame, area: Rect, metrics: &Arc<MetricsStore>, scroll: usize) {
    let snap = metrics.snapshot();
    let (table, total) = model_table(&snap, " Models ".to_owned(), scroll);
    frame.render_widget(table, area);
    super::render_scrollbar(frame, area, total, scroll);
}
