use ratatui::prelude::*;
use ratatui::widgets::{Scrollbar, ScrollbarOrientation, ScrollbarState};

pub mod errors;
pub mod models;
pub mod overview;
pub mod providers;

/// Formats a token count for display: raw below 1K, "1.0K" style up to ~1M,
/// "1.5M" style above.
pub fn format_tokens(n: u64) -> String {
    if n >= 999_950 {
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        let val = n as f64 / 1_000_000.0;
        format!("{val:.1}M")
    } else if n >= 1_000 {
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        let val = n as f64 / 1_000.0;
        format!("{val:.1}K")
    } else {
        n.to_string()
    }
}

/// Formats a duration as a human-readable relative time string (e.g. "3s ago",
/// "5m ago", "2h ago", "1d ago").
pub fn format_time_ago(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

/// Formats a duration for display: raw ms below 1s, seconds with decimals
/// below 1m, minutes+seconds above.
pub fn format_duration(dur: std::time::Duration) -> String {
    let total_secs = dur.as_secs_f64();
    if total_secs >= 60.0 {
        let mut minutes = (total_secs / 60.0).floor();
        let mut seconds = total_secs - minutes * 60.0;
        // Clamp to avoid "1m60.0s" from rounding
        if seconds >= 59.95 {
            minutes += 1.0;
            seconds = 0.0;
        }
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let minutes_u64 = minutes as u64;
        format!("{minutes_u64}m{seconds:.1}s")
    } else if total_secs >= 1.0 {
        // Truncate to avoid "60.00s" from rounding
        let secs = (total_secs * 100.0).floor() / 100.0;
        format!("{secs:.2}s")
    } else {
        format!("{}ms", dur.as_millis())
    }
}

/// Renders a subtle vertical scrollbar when `total_rows` exceeds the visible
/// area. Accounts for border (top + bottom) and header row = 3 lines of
/// overhead.
pub fn render_scrollbar(frame: &mut Frame, area: Rect, total_rows: usize, scroll: usize) {
    #[allow(clippy::as_conversions)]
    let visible_rows = area.height.saturating_sub(3) as usize;
    if total_rows > visible_rows {
        let mut state = ScrollbarState::new(total_rows)
            .viewport_content_length(visible_rows)
            .position(scroll);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .thumb_style(Style::default().fg(Color::DarkGray))
            .track_style(Style::default().fg(Color::Black));
        frame.render_stateful_widget(scrollbar, area, &mut state);
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn format_tokens_thresholds() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1000), "1.0K");
        assert_eq!(format_tokens(999_949), "999.9K");
        assert_eq!(format_tokens(999_950), "1.0M");
        assert_eq!(format_tokens(1_500_000), "1.5M");
    }

    #[test]
    fn format_time_ago_seconds() {
        assert_eq!(format_time_ago(std::time::Duration::from_secs(0)), "0s ago");
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(30)),
            "30s ago"
        );
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(59)),
            "59s ago"
        );
    }

    #[test]
    fn format_time_ago_minutes() {
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(60)),
            "1m ago"
        );
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(90)),
            "1m ago"
        );
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(3599)),
            "59m ago"
        );
    }

    #[test]
    fn format_time_ago_hours() {
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(3600)),
            "1h ago"
        );
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(7200)),
            "2h ago"
        );
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(86399)),
            "23h ago"
        );
    }

    #[test]
    fn format_duration_millis() {
        assert_eq!(format_duration(std::time::Duration::from_millis(0)), "0ms");
        assert_eq!(
            format_duration(std::time::Duration::from_millis(500)),
            "500ms"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_millis(999)),
            "999ms"
        );
    }

    #[test]
    fn format_duration_seconds() {
        assert_eq!(
            format_duration(std::time::Duration::from_millis(1000)),
            "1.00s"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_millis(1500)),
            "1.50s"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_millis(59999)),
            "59.99s"
        );
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(
            format_duration(std::time::Duration::from_secs(60)),
            "1m0.0s"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_millis(304_088)),
            "5m4.1s"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_secs(3661)),
            "61m1.0s"
        );
    }

    #[test]
    fn format_time_ago_days() {
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(86400)),
            "1d ago"
        );
        assert_eq!(
            format_time_ago(std::time::Duration::from_secs(172_800)),
            "2d ago"
        );
    }
}
