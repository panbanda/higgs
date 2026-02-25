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

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    use chrono::Utc;

    use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};

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
    fn draw_no_errors() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(3600)));
        metrics.record(sample_record());
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), &metrics, 0);
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Errors (0)"), "should show zero errors");
    }

    #[test]
    fn draw_with_errors() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(3600)));
        let mut error_rec = sample_record();
        error_rec.status = 500;
        error_rec.error_body = Some("Internal server error".to_owned());
        metrics.record(error_rec);
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), &metrics, 0);
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Errors (1)"), "should show one error");
    }
}
