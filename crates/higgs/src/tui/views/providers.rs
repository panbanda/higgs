use std::sync::Arc;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use super::{format_duration, format_tokens};
use crate::metrics::MetricsStore;

pub fn draw(
    frame: &mut Frame,
    area: Rect,
    metrics: &Arc<MetricsStore>,
    scroll: usize,
    configured_providers: Option<&[String]>,
) {
    let snap = metrics.snapshot();
    let groups = MetricsStore::group_by(&snap, |r| r.provider.clone());

    let header = Row::new(vec![
        "Provider", "Reqs", "In", "Out", "Avg/Req", "P50", "P95", "Errs",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let mut names: Vec<String> = groups.keys().cloned().collect();
    // Merge in configured provider names not already present in metrics
    if let Some(configured) = configured_providers {
        for name in configured {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    names.sort();

    let total = names.len();

    let rows: Vec<Row> = names
        .iter()
        .skip(scroll)
        .map(|name| {
            let Some(records) = groups.get(name) else {
                // Zero-traffic configured provider
                let dim = Style::default().fg(Color::DarkGray);
                return Row::new(vec![
                    Cell::from(name.as_str()).style(dim),
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
            Row::new(vec![
                Cell::from(name.as_str()).style(Style::default().fg(Color::White)),
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

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    use chrono::Utc;

    use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};

    const METRICS_WINDOW_SECS: u64 = 60 * 60;

    fn sample_record() -> RequestRecord {
        RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: Utc::now(),
            model: "claude-opus-4-6".to_owned(),
            provider: "anthropic".to_owned(),
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
    fn draw_with_records_contains_provider() {
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
            content.contains("anthropic"),
            "buffer should contain provider name"
        );
    }

    #[test]
    fn draw_shows_configured_zero_traffic_providers() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(METRICS_WINDOW_SECS)));
        let configured = vec!["openai".to_owned(), "ollama".to_owned()];
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), &metrics, 0, Some(&configured));
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            content.contains("openai"),
            "buffer should contain configured provider"
        );
    }
}
