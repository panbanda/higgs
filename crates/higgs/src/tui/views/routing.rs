use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table};

use crate::tui::TuiConfig;

pub fn draw(frame: &mut Frame, area: Rect, config: Option<&TuiConfig>, sidebar_width_pct: u16) {
    if let Some(c) = config {
        draw_with_config(frame, area, c, sidebar_width_pct);
    } else {
        let msg = Paragraph::new("No configuration available")
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).title(" Routing "));
        frame.render_widget(msg, area);
    }
}

#[allow(clippy::indexing_slicing)]
fn draw_with_config(frame: &mut Frame, area: Rect, config: &TuiConfig, sidebar_width_pct: u16) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(8),
            Constraint::Length(8),
        ])
        .split(area);

    // Layout::split with 3 constraints always returns 3 elements
    draw_auto_router(frame, chunks[0], config);
    draw_routes_table(frame, chunks[1], config);
    draw_bottom_panels(frame, chunks[2], config, sidebar_width_pct);
}

fn draw_auto_router(frame: &mut Frame, area: Rect, config: &TuiConfig) {
    let ar = &config.auto_router;
    let (status_text, status_style) = if ar.enabled {
        if ar.force {
            ("ENABLED (force)", Style::default().fg(Color::Yellow))
        } else {
            ("ENABLED", Style::default().fg(Color::Green))
        }
    } else {
        ("DISABLED", Style::default().fg(Color::DarkGray))
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("  Status: "),
            Span::styled(status_text, status_style),
        ]),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(&ar.model, Style::default().fg(Color::White)),
            Span::raw("  Timeout: "),
            Span::styled(
                format!("{}ms", ar.timeout_ms),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Precedence: "),
            Span::styled(
                "auto -> exact local model -> regex route -> default",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    let widget = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Auto Router "),
    );
    frame.render_widget(widget, area);
}

#[allow(clippy::indexing_slicing)]
fn draw_routes_table(frame: &mut Frame, area: Rect, config: &TuiConfig) {
    let header = Row::new(vec![
        "#",
        "Pattern",
        "Provider",
        "Rewrite",
        "Name",
        "Description",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let mut rows: Vec<Row> = config
        .routes
        .iter()
        .enumerate()
        .map(|(i, route)| {
            let priority = format!("{}", i + 1);
            let pattern = route.pattern.as_deref().unwrap_or("-");
            let rewrite = route.model_rewrite.as_deref().unwrap_or("-");
            let name = route.name.as_deref().unwrap_or("-");
            let description = route.description.as_deref().unwrap_or("-");

            let provider_style = if route.provider == "higgs" {
                Style::default().fg(Color::Magenta)
            } else {
                Style::default().fg(Color::White)
            };

            Row::new(vec![
                Cell::from(priority),
                Cell::from(pattern.to_owned()).style(Style::default().fg(Color::Cyan)),
                Cell::from(route.provider.clone()).style(provider_style),
                Cell::from(rewrite.to_owned()),
                Cell::from(name.to_owned()),
                Cell::from(description.to_owned()).style(Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    // Default route as last row
    let default_style = Style::default().fg(Color::DarkGray);
    let default_provider_style = if config.default_provider == "higgs" {
        Style::default().fg(Color::Magenta)
    } else {
        Style::default().fg(Color::White)
    };
    rows.push(Row::new(vec![
        Cell::from("*").style(default_style),
        Cell::from("(default)").style(default_style),
        Cell::from(config.default_provider.clone()).style(default_provider_style),
        Cell::from("-").style(default_style),
        Cell::from("-").style(default_style),
        Cell::from("-").style(default_style),
    ]));

    let table = Table::new(
        rows,
        [
            Constraint::Length(3),
            Constraint::Min(15),
            Constraint::Length(15),
            Constraint::Length(15),
            Constraint::Length(15),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Routes (local exact match beats regex) "),
    );

    frame.render_widget(table, area);
}

#[allow(clippy::indexing_slicing)]
fn bottom_panel_columns(area: Rect, sidebar_width_pct: u16) -> std::rc::Rc<[Rect]> {
    let providers_width_pct = 100_u16.saturating_sub(sidebar_width_pct);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(sidebar_width_pct),
            Constraint::Percentage(providers_width_pct),
        ])
        .split(area)
}

#[allow(clippy::indexing_slicing)]
fn draw_bottom_panels(frame: &mut Frame, area: Rect, config: &TuiConfig, sidebar_width_pct: u16) {
    let cols = bottom_panel_columns(area, sidebar_width_pct);

    // Local models list
    let model_items: Vec<ListItem> = if config.model_names.is_empty() {
        vec![ListItem::new(Span::styled(
            "  (none)",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        config
            .model_names
            .iter()
            .map(|name| {
                ListItem::new(Span::styled(
                    format!("  {name}"),
                    Style::default().fg(Color::White),
                ))
            })
            .collect()
    };
    let models_list = List::new(model_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Local Models "),
    );
    // Layout::split with 2 constraints always returns 2 elements
    frame.render_widget(models_list, cols[0]);

    // Providers list
    let provider_items: Vec<ListItem> = if config.provider_names.is_empty() {
        vec![ListItem::new(Span::styled(
            "  (none)",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        config
            .provider_names
            .iter()
            .map(|name| {
                let style = if name == "higgs" {
                    Style::default().fg(Color::Magenta)
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(Span::styled(format!("  {name}"), style))
            })
            .collect()
    };
    let providers_list = List::new(provider_items)
        .block(Block::default().borders(Borders::ALL).title(" Providers "));
    frame.render_widget(providers_list, cols[1]);
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{TuiAutoRouter, TuiConfig, TuiRoute};

    fn sample_config() -> TuiConfig {
        TuiConfig {
            profile: None,
            model_names: vec!["Llama-3.2-1B".to_owned()],
            provider_names: vec!["anthropic".to_owned(), "higgs".to_owned()],
            routes: vec![TuiRoute {
                pattern: Some("claude-.*".to_owned()),
                provider: "anthropic".to_owned(),
                model_rewrite: None,
                name: None,
                description: None,
            }],
            default_provider: "higgs".to_owned(),
            auto_router: TuiAutoRouter {
                enabled: false,
                force: false,
                model: "katanemo/Arch-Router-1.5B".to_owned(),
                timeout_ms: 2000,
            },
        }
    }

    #[test]
    fn draw_no_config_no_panic() {
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), None, 50);
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("No configuration available"));
    }

    #[test]
    fn draw_with_config_no_panic() {
        let config = sample_config();
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), Some(&config), 50);
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Auto Router"));
        assert!(content.contains("Routes"));
        assert!(content.contains("anthropic"));
    }

    #[test]
    fn draw_auto_router_enabled() {
        let mut config = sample_config();
        config.auto_router.enabled = true;
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw(f, f.area(), Some(&config), 50);
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("ENABLED"));
    }

    #[test]
    fn bottom_panel_columns_follow_sidebar_width() {
        let area = Rect::new(0, 0, 100, 8);

        let default_cols = bottom_panel_columns(area, 50);
        match default_cols.as_ref() {
            [default_left, default_right] => {
                assert_eq!(default_left.width, 50);
                assert_eq!(default_right.width, 50);
            }
            _ => panic!("bottom panel columns should produce exactly two columns"),
        }

        let wide_models_cols = bottom_panel_columns(area, 65);
        match wide_models_cols.as_ref() {
            [wide_left, wide_right] => {
                assert_eq!(wide_left.width, 65);
                assert_eq!(wide_right.width, 35);
            }
            _ => panic!("bottom panel columns should produce exactly two columns"),
        }
    }
}
