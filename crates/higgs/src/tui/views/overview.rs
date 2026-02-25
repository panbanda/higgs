use std::sync::Arc;

use ratatui::prelude::*;
use ratatui::symbols::Marker;
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table,
};

use super::{format_duration, format_time_ago, format_tokens};
use crate::metrics::{MetricsStore, RoutingMethod};

fn time_axis_labels(num_buckets: usize) -> Vec<String> {
    vec![
        format!("-{}m", num_buckets),
        format!("-{}m", num_buckets * 3 / 4),
        format!("-{}m", num_buckets / 2),
        format!("-{}m", num_buckets / 4),
        "now".to_owned(),
    ]
}

fn value_axis_labels(max: u64, steps: u64) -> Vec<String> {
    (0..=steps)
        .map(|i| {
            let v = max * i / steps;
            format_tokens(v)
        })
        .collect()
}

#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
fn build_time_chart(
    points: &[(f64, f64)],
    num_buckets: usize,
    title: String,
    color: Color,
    ceil: u64,
) -> Chart<'_> {
    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(points);
    Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title(title))
        .x_axis(
            Axis::default()
                .bounds([0.0, (num_buckets - 1) as f64])
                .labels(time_axis_labels(num_buckets)),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, ceil as f64])
                .labels(value_axis_labels(ceil, 4)),
        )
}

#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
fn to_points(data: &[u64]) -> Vec<(f64, f64)> {
    data.iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect()
}

#[allow(clippy::indexing_slicing)]
fn draw_charts_row(
    frame: &mut Frame,
    area: Rect,
    snap: &[crate::metrics::RequestRecord],
    num_buckets: usize,
) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let rpm_data = MetricsStore::requests_per_minute(snap, num_buckets);
    let total_requests: u64 = rpm_data.iter().sum();
    let rpm_ceil = rpm_data.iter().max().unwrap_or(&1).max(&10).div_ceil(5) * 5;
    let rpm_points = to_points(&rpm_data);
    let rpm_chart = build_time_chart(
        &rpm_points,
        num_buckets,
        format!(" Requests/min (total: {total_requests}) "),
        Color::Cyan,
        rpm_ceil,
    );
    // Layout::split with 2 constraints always returns 2 elements
    frame.render_widget(rpm_chart, cols[0]);

    let tpm_data = MetricsStore::tokens_per_minute(snap, num_buckets);
    let total_tokens: u64 = tpm_data.iter().sum();
    let tpm_ceil = tpm_data.iter().max().unwrap_or(&1).max(&100).div_ceil(100) * 100;
    let tpm_points = to_points(&tpm_data);
    let tpm_chart = build_time_chart(
        &tpm_points,
        num_buckets,
        format!(" Tokens/min (total: {}) ", format_tokens(total_tokens)),
        Color::Green,
        tpm_ceil,
    );
    frame.render_widget(tpm_chart, cols[1]);
}

#[allow(clippy::as_conversions)]
fn draw_latency(frame: &mut Frame, area: Rect, snap: &[crate::metrics::RequestRecord]) {
    let durations: Vec<std::time::Duration> = snap.iter().map(|r| r.duration).collect();
    let p50 = MetricsStore::duration_percentile(&durations, 50);
    let p95 = MetricsStore::duration_percentile(&durations, 95);
    let p99 = MetricsStore::duration_percentile(&durations, 99);
    let avg = if snap.is_empty() {
        std::time::Duration::ZERO
    } else {
        let total: std::time::Duration = snap.iter().map(|r| r.duration).sum();
        let len_u32 = u32::try_from(snap.len()).unwrap_or(u32::MAX);
        total / len_u32
    };
    let lines = vec![
        Line::from(vec![
            Span::raw(" Avg: "),
            Span::styled(format_duration(avg), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::raw(" P50: "),
            Span::styled(format_duration(p50), Style::default().fg(Color::Green)),
            Span::raw("  P95: "),
            Span::styled(format_duration(p95), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw(" P99: "),
            Span::styled(format_duration(p99), Style::default().fg(Color::Red)),
        ]),
    ];
    let widget =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Duration "));
    frame.render_widget(widget, area);
}

const fn status_label(code: u16) -> Option<&'static str> {
    match code {
        200 => Some("200 OK"),
        201 => Some("201 Created"),
        204 => Some("204 No Content"),
        400 => Some("400 Bad Request"),
        401 => Some("401 Unauthorized"),
        403 => Some("403 Forbidden"),
        404 => Some("404 Not Found"),
        429 => Some("429 Rate Limited"),
        500 => Some("500 Internal Error"),
        502 => Some("502 Bad Gateway"),
        503 => Some("503 Unavailable"),
        529 => Some("529 Overloaded"),
        0..=199
        | 202
        | 203
        | 205..=399
        | 402
        | 405..=428
        | 430..=499
        | 501
        | 504..=528
        | 530..=u16::MAX => None,
    }
}

const fn status_color(code: u16) -> Color {
    if code < 300 {
        Color::Green
    } else if code < 500 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn draw_status_codes(frame: &mut Frame, area: Rect, snap: &[crate::metrics::RequestRecord]) {
    let statuses = MetricsStore::status_counts(snap);
    let mut entries: Vec<(u16, u64)> = statuses.into_iter().collect();
    entries.sort_by_key(|(s, _)| *s);
    let lines: Vec<Line> = if entries.is_empty() {
        vec![Line::from(Span::styled(
            " No traffic yet",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        entries
            .iter()
            .map(|(status, count)| {
                status_label(*status).map_or_else(
                    || Line::from(format!(" {status}: {count}")),
                    |label| {
                        Line::from(vec![
                            Span::styled(format!(" {label}: "), status_color(*status)),
                            Span::styled(count.to_string(), Style::default().fg(Color::White)),
                        ])
                    },
                )
            })
            .collect()
    };
    let widget = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Status Codes "),
    );
    frame.render_widget(widget, area);
}

#[allow(clippy::indexing_slicing)]
fn draw_stats_row(frame: &mut Frame, area: Rect, snap: &[crate::metrics::RequestRecord]) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Layout::split with 2 constraints always returns 2 elements
    draw_latency(frame, cols[0], snap);
    draw_status_codes(frame, cols[1], snap);
}

fn draw_token_usage(frame: &mut Frame, area: Rect, snap: &[crate::metrics::RequestRecord]) {
    let (table, _) = super::models::model_table(snap, " Token Usage ".to_owned(), 0);
    frame.render_widget(table, area);
}

fn duration_style(
    dur: std::time::Duration,
    p50: std::time::Duration,
    p95: std::time::Duration,
    p99: std::time::Duration,
) -> Style {
    if dur >= p99 {
        Style::default().fg(Color::Red)
    } else if dur >= p95 {
        Style::default().fg(Color::Yellow)
    } else if dur >= p50 {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn draw_live_log(
    frame: &mut Frame,
    area: Rect,
    snap: &[crate::metrics::RequestRecord],
    scroll: usize,
) {
    let header = Row::new(vec![
        "Age", "Model", "Provider", "Route", "Status", "Duration", "In/Out",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD))
    .bottom_margin(0);

    let now = std::time::Instant::now();
    let durations: Vec<std::time::Duration> = snap.iter().map(|r| r.duration).collect();
    let p50 = MetricsStore::duration_percentile(&durations, 50);
    let p95 = MetricsStore::duration_percentile(&durations, 95);
    let p99 = MetricsStore::duration_percentile(&durations, 99);

    let mut sorted: Vec<_> = snap.iter().collect();
    sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    let total_rows = sorted.len();

    let rows: Vec<Row> = sorted
        .iter()
        .skip(scroll)
        .take(50)
        .map(|r| {
            let status_style = if r.status >= 400 {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Green)
            };
            let age = now.duration_since(r.timestamp);
            let (route_label, route_style) = match r.routing_method {
                RoutingMethod::Pattern => ("PTN", Style::default().fg(Color::Cyan)),
                RoutingMethod::Auto => ("AUT", Style::default().fg(Color::Yellow)),
                RoutingMethod::Default => ("DEF", Style::default().fg(Color::DarkGray)),
                RoutingMethod::Higgs => ("HGS", Style::default().fg(Color::Magenta)),
            };
            Row::new(vec![
                Cell::from(format_time_ago(age)).style(Style::default().fg(Color::DarkGray)),
                Cell::from(r.model.as_str()),
                Cell::from(r.provider.as_str()).style(Style::default().fg(Color::DarkGray)),
                Cell::from(route_label).style(route_style),
                Cell::from(r.status.to_string()).style(status_style),
                Cell::from(format_duration(r.duration))
                    .style(duration_style(r.duration, p50, p95, p99)),
                Cell::from(Line::from(vec![
                    Span::styled(
                        format_tokens(r.input_tokens),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw("/"),
                    Span::styled(
                        format_tokens(r.output_tokens),
                        Style::default().fg(Color::Green),
                    ),
                ])),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Min(20),
            Constraint::Length(12),
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Length(10),
            Constraint::Length(12),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Live Log "));

    frame.render_widget(table, area);
    super::render_scrollbar(frame, area, total_rows, scroll);
}

#[allow(clippy::indexing_slicing)]
pub fn draw(frame: &mut Frame, area: Rect, metrics: &Arc<MetricsStore>, scroll: usize) {
    let snap = metrics.snapshot();
    let num_buckets = usize::try_from(metrics.window_minutes().max(1)).unwrap_or(1);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // charts row
            Constraint::Length(6),  // stats row (duration + status)
            Constraint::Length(6),  // token usage (full width)
            Constraint::Min(0),     // live log
        ])
        .split(area);

    // Layout::split with 4 constraints always returns 4 elements
    draw_charts_row(frame, chunks[0], &snap, num_buckets);
    draw_stats_row(frame, chunks[1], &snap);
    draw_token_usage(frame, chunks[2], &snap);
    draw_live_log(frame, chunks[3], &snap, scroll);
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
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
    fn status_label_known_codes() {
        assert_eq!(status_label(200), Some("200 OK"));
        assert_eq!(status_label(429), Some("429 Rate Limited"));
        assert_eq!(status_label(500), Some("500 Internal Error"));
        assert_eq!(status_label(502), Some("502 Bad Gateway"));
    }

    #[test]
    fn status_label_unknown_code() {
        assert_eq!(status_label(418), None);
        assert_eq!(status_label(999), None);
    }

    #[test]
    fn status_color_ranges() {
        assert_eq!(status_color(200), Color::Green);
        assert_eq!(status_color(299), Color::Green);
        assert_eq!(status_color(400), Color::Yellow);
        assert_eq!(status_color(429), Color::Yellow);
        assert_eq!(status_color(500), Color::Red);
        assert_eq!(status_color(502), Color::Red);
    }

    #[test]
    fn duration_style_below_p50() {
        let p50 = Duration::from_millis(500);
        let p95 = Duration::from_secs(1);
        let p99 = Duration::from_secs(2);
        let style = duration_style(Duration::from_millis(100), p50, p95, p99);
        assert_eq!(style, Style::default().fg(Color::DarkGray));
    }

    #[test]
    fn duration_style_at_p50() {
        let p50 = Duration::from_millis(500);
        let p95 = Duration::from_secs(1);
        let p99 = Duration::from_secs(2);
        let style = duration_style(Duration::from_millis(700), p50, p95, p99);
        assert_eq!(style, Style::default().fg(Color::White));
    }

    #[test]
    fn duration_style_at_p95() {
        let p50 = Duration::from_millis(500);
        let p95 = Duration::from_secs(1);
        let p99 = Duration::from_secs(2);
        let style = duration_style(Duration::from_millis(1500), p50, p95, p99);
        assert_eq!(style, Style::default().fg(Color::Yellow));
    }

    #[test]
    fn duration_style_at_p99() {
        let p50 = Duration::from_millis(500);
        let p95 = Duration::from_secs(1);
        let p99 = Duration::from_secs(2);
        let style = duration_style(Duration::from_secs(3), p50, p95, p99);
        assert_eq!(style, Style::default().fg(Color::Red));
    }

    #[test]
    fn time_axis_labels_count() {
        let labels = time_axis_labels(60);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels.last().unwrap(), "now");
        assert_eq!(labels.first().unwrap(), "-60m");
    }

    #[test]
    fn value_axis_labels_count() {
        let labels = value_axis_labels(1000, 4);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels[0], "0");
    }

    #[test]
    fn draw_empty_metrics_no_panic() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(3600)));
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), &metrics, 0);
            })
            .unwrap();
    }

    #[test]
    fn draw_with_records_contains_expected_text() {
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
        assert!(
            content.contains("Requests/min"),
            "buffer should contain Requests/min"
        );
        assert!(
            content.contains("Duration"),
            "buffer should contain Duration"
        );
        assert!(
            content.contains("Live Log"),
            "buffer should contain Live Log"
        );
    }
}
