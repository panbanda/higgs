use std::sync::Arc;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use super::{format_duration, format_tokens};
use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};

/// Builds model-summary rows from a snapshot. Shared by the Models tab and the
/// overview Token Usage panel.
#[allow(clippy::too_many_lines)]
pub fn model_table(
    snap: &[RequestRecord],
    title: String,
    skip: usize,
    configured_models: Option<&[String]>,
) -> (Table<'static>, usize) {
    let groups = MetricsStore::group_by(snap, |r| r.model.clone());

    let header = Row::new(vec![
        "", "Model", "Reqs", "In", "Out", "Avg/Req", "P50", "P95", "Errs",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let mut model_names: Vec<String> = groups.keys().cloned().collect();
    // Merge in configured names not already present in metrics
    if let Some(configured) = configured_models {
        for name in configured {
            if !model_names.contains(name) {
                model_names.push(name.clone());
            }
        }
    }
    model_names.sort();
    let total = model_names.len();

    let rows: Vec<Row> = model_names
        .iter()
        .skip(skip)
        .map(|model| {
            let Some(records) = groups.get(model) else {
                // Zero-traffic configured model
                let dim = Style::default().fg(Color::DarkGray);
                return Row::new(vec![
                    Cell::from("---").style(dim),
                    Cell::from(model.clone()).style(dim),
                    Cell::from("-").style(dim),
                    Cell::from("-").style(dim),
                    Cell::from("-").style(dim),
                    Cell::from("-").style(dim),
                    Cell::from("-").style(dim),
                    Cell::from("-").style(dim),
                    Cell::from("-").style(dim),
                ]);
            };
            let count = u64::try_from(records.len()).unwrap_or(0);
            let input: u64 = records.iter().map(|r| r.input_tokens).sum();
            let output: u64 = records.iter().map(|r| r.output_tokens).sum();
            let durations: Vec<_> = records
                .iter()
                .filter(|r| !r.is_error())
                .map(|r| r.duration)
                .collect();
            let p50 = MetricsStore::duration_percentile(&durations, 50);
            let p95 = MetricsStore::duration_percentile(&durations, 95);
            let errors: u64 =
                u64::try_from(records.iter().filter(|r| r.is_error()).count()).unwrap_or(0);
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

            Row::new(vec![
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
            ])
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

pub fn draw(
    frame: &mut Frame,
    area: Rect,
    metrics: &Arc<MetricsStore>,
    scroll: usize,
    configured_models: Option<&[String]>,
) {
    let snap = metrics.snapshot();
    let (table, total) = model_table(&snap, " Models ".to_owned(), scroll, configured_models);
    frame.render_widget(table, area);
    super::render_scrollbar(frame, area, total, scroll);
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    use chrono::Utc;

    use crate::metrics::{RequestRecord, RoutingMethod};

    const METRICS_WINDOW_SECS: u64 = 60 * 60;

    fn sample_record() -> RequestRecord {
        RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: Utc::now(),
            model: Some("claude-opus-4-6".to_owned()),
            provider: Some("anthropic".to_owned()),
            routing_method: RoutingMethod::Default,
            status: 200,
            duration: Duration::from_millis(500),
            input_tokens: 100,
            output_tokens: 200,
            error_body: None,
        }
    }

    #[test]
    fn draw_empty_metrics_no_panic() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(METRICS_WINDOW_SECS)));
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), &metrics, 0, None);
            })
            .unwrap();
    }

    #[test]
    fn draw_with_records_contains_model_name() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(METRICS_WINDOW_SECS)));
        metrics.record(sample_record());
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), &metrics, 0, None);
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            content.contains("claude-opus"),
            "buffer should contain model name"
        );
    }

    #[test]
    fn model_table_groups_by_model() {
        let mut records = Vec::new();
        for _ in 0..3 {
            records.push(sample_record());
        }
        let mut other = sample_record();
        other.model = Some("gpt-4".to_owned());
        records.push(other);
        let (_, total) = model_table(&records, " Test ".to_owned(), 0, None);
        assert_eq!(total, 2);
    }

    #[test]
    fn model_table_includes_configured_with_zero_traffic() {
        let records = vec![sample_record()];
        let configured = vec!["claude-opus-4-6".to_owned(), "unused-model".to_owned()];
        let (_, total) = model_table(&records, " Test ".to_owned(), 0, Some(&configured));
        assert_eq!(total, 2);
    }

    #[test]
    fn model_table_excludes_error_latency_from_percentiles() {
        let mut error = sample_record();
        error.status = 404;
        error.duration = Duration::from_millis(500);
        error.input_tokens = 0;
        error.output_tokens = 0;
        let backend = ratatui::backend::TestBackend::new(120, 8);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let (table, _) = model_table(&[error], " Test ".to_owned(), 0, None);
                frame.render_widget(table, frame.area());
            })
            .unwrap();
        let content: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            !content.contains("500ms"),
            "error latency leaked into model percentiles"
        );
    }
}
