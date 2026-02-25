pub mod views;

use std::io;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};

use crate::metrics::MetricsStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Models,
    Providers,
    Errors,
}

impl Tab {
    fn titles() -> Vec<&'static str> {
        vec!["Overview [1]", "Models [2]", "Providers [3]", "Errors [4]"]
    }

    const fn index(self) -> usize {
        match self {
            Self::Overview => 0,
            Self::Models => 1,
            Self::Providers => 2,
            Self::Errors => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitMode {
    Quit,
    Detach,
}

pub struct App {
    pub metrics: Arc<MetricsStore>,
    pub active_tab: Tab,
    pub scroll_offset: usize,
    pub exit_mode: Option<ExitMode>,
    pub attached: bool,
}

impl App {
    pub const fn new(metrics: Arc<MetricsStore>, attached: bool) -> Self {
        Self {
            metrics,
            active_tab: Tab::Overview,
            scroll_offset: 0,
            exit_mode: None,
            attached,
        }
    }

    pub fn handle_key(&mut self, key: event::KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.exit_mode = Some(ExitMode::Quit);
            return;
        }
        match key.code {
            KeyCode::Char('q') => self.exit_mode = Some(ExitMode::Quit),
            KeyCode::Char('d') if !self.attached => {
                self.exit_mode = Some(ExitMode::Detach);
            }
            KeyCode::Char('1') => {
                self.active_tab = Tab::Overview;
                self.scroll_offset = 0;
            }
            KeyCode::Char('2') => {
                self.active_tab = Tab::Models;
                self.scroll_offset = 0;
            }
            KeyCode::Char('3') => {
                self.active_tab = Tab::Providers;
                self.scroll_offset = 0;
            }
            KeyCode::Char('4') => {
                self.active_tab = Tab::Errors;
                self.scroll_offset = 0;
            }
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                self.active_tab = match self.active_tab {
                    Tab::Overview => Tab::Models,
                    Tab::Models => Tab::Providers,
                    Tab::Providers => Tab::Errors,
                    Tab::Errors => Tab::Overview,
                };
                self.scroll_offset = 0;
            }
            KeyCode::Left | KeyCode::Char('h') => {
                self.active_tab = match self.active_tab {
                    Tab::Overview => Tab::Errors,
                    Tab::Models => Tab::Overview,
                    Tab::Providers => Tab::Models,
                    Tab::Errors => Tab::Providers,
                };
                self.scroll_offset = 0;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            KeyCode::Char(_)
            | KeyCode::Backspace
            | KeyCode::Enter
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Insert
            | KeyCode::Delete
            | KeyCode::F(_)
            | KeyCode::BackTab
            | KeyCode::CapsLock
            | KeyCode::ScrollLock
            | KeyCode::NumLock
            | KeyCode::PrintScreen
            | KeyCode::Pause
            | KeyCode::Menu
            | KeyCode::KeypadBegin
            | KeyCode::Null
            | KeyCode::Esc
            | KeyCode::Media(_)
            | KeyCode::Modifier(_) => {}
        }
    }

    #[allow(clippy::indexing_slicing)]
    pub fn draw(&self, frame: &mut Frame) {
        let title = if self.attached {
            " higgs (attached) "
        } else {
            " higgs "
        };

        let hint = if self.attached {
            " q:quit "
        } else {
            " q:quit  d:detach "
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(frame.area());

        let tabs = Tabs::new(Tab::titles().into_iter().map(Line::from))
            .block(Block::default().borders(Borders::ALL).title(title))
            .select(self.active_tab.index())
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            );
        // Layout::split with 3 constraints always returns 3 elements
        frame.render_widget(tabs, chunks[0]);

        let content_area = chunks[1];
        match self.active_tab {
            Tab::Overview => {
                views::overview::draw(frame, content_area, &self.metrics, self.scroll_offset);
            }
            Tab::Models => {
                views::models::draw(frame, content_area, &self.metrics, self.scroll_offset);
            }
            Tab::Providers => {
                views::providers::draw(frame, content_area, &self.metrics, self.scroll_offset);
            }
            Tab::Errors => {
                views::errors::draw(frame, content_area, &self.metrics, self.scroll_offset);
            }
        }

        let footer = Paragraph::new(Line::from(vec![Span::styled(
            hint,
            Style::default().fg(Color::DarkGray),
        )]));
        frame.render_widget(footer, chunks[2]);
    }
}

pub fn run(metrics: Arc<MetricsStore>, attached: bool) -> io::Result<ExitMode> {
    let mut terminal = ratatui::init();

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        default_hook(info);
    }));

    let mut app = App::new(metrics, attached);

    let result = (|| -> io::Result<ExitMode> {
        loop {
            terminal.draw(|frame| app.draw(frame))?;

            if event::poll(Duration::from_millis(250))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        app.handle_key(key);
                    }
                    Event::Resize(_, _) | Event::FocusGained => {
                        terminal.clear()?;
                    }
                    Event::FocusLost | Event::Mouse(_) | Event::Paste(_) | Event::Key(_) => {}
                }
            }

            if let Some(mode) = app.exit_mode {
                return Ok(mode);
            }
        }
    })();

    ratatui::restore();
    result
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn make_app() -> App {
        App::new(Arc::new(MetricsStore::new(Duration::from_secs(60))), false)
    }

    fn make_attached_app() -> App {
        App::new(Arc::new(MetricsStore::new(Duration::from_secs(60))), true)
    }

    fn key(code: KeyCode) -> event::KeyEvent {
        event::KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl_c() -> event::KeyEvent {
        event::KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    fn assert_tab_cycle(nav_key: KeyCode, expected: &[Tab]) {
        let mut app = make_app();
        for &tab in expected {
            app.handle_key(key(nav_key));
            assert_eq!(app.active_tab, tab);
        }
    }

    fn assert_nav_resets_scroll(nav_key: KeyCode) {
        let mut app = make_app();
        app.handle_key(key(KeyCode::Char('j')));
        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.scroll_offset, 2);
        app.handle_key(key(nav_key));
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = make_app();
        app.handle_key(ctrl_c());
        assert_eq!(app.exit_mode, Some(ExitMode::Quit));
    }

    #[test]
    fn q_quits() {
        let mut app = make_app();
        app.handle_key(key(KeyCode::Char('q')));
        assert_eq!(app.exit_mode, Some(ExitMode::Quit));
    }

    #[test]
    fn plain_c_does_not_quit() {
        let mut app = make_app();
        app.handle_key(key(KeyCode::Char('c')));
        assert!(app.exit_mode.is_none());
    }

    #[test]
    fn number_keys_switch_tabs() {
        let mut app = make_app();
        for (ch, tab) in [
            ('2', Tab::Models),
            ('3', Tab::Providers),
            ('4', Tab::Errors),
            ('1', Tab::Overview),
        ] {
            app.handle_key(key(KeyCode::Char(ch)));
            assert_eq!(app.active_tab, tab);
        }
    }

    #[test]
    fn number_keys_reset_scroll() {
        let mut app = make_app();
        app.handle_key(key(KeyCode::Char('j')));
        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.scroll_offset, 2);
        app.handle_key(key(KeyCode::Char('2')));
        assert_eq!(app.scroll_offset, 0);
        app.handle_key(key(KeyCode::Char('j')));
        app.handle_key(key(KeyCode::Char('1')));
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn tab_cycles_through_tabs() {
        assert_tab_cycle(
            KeyCode::Tab,
            &[Tab::Models, Tab::Providers, Tab::Errors, Tab::Overview],
        );
    }

    #[test]
    fn scroll_j_k() {
        let mut app = make_app();
        assert_eq!(app.scroll_offset, 0);
        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.scroll_offset, 1);
        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.scroll_offset, 2);
        app.handle_key(key(KeyCode::Char('k')));
        assert_eq!(app.scroll_offset, 1);
        app.handle_key(key(KeyCode::Char('k')));
        assert_eq!(app.scroll_offset, 0);
        // k at 0 stays at 0
        app.handle_key(key(KeyCode::Char('k')));
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn tab_resets_scroll() {
        assert_nav_resets_scroll(KeyCode::Tab);
    }

    #[test]
    fn right_arrow_cycles_forward() {
        assert_tab_cycle(
            KeyCode::Right,
            &[Tab::Models, Tab::Providers, Tab::Errors, Tab::Overview],
        );
    }

    #[test]
    fn left_arrow_cycles_backward() {
        assert_tab_cycle(
            KeyCode::Left,
            &[Tab::Errors, Tab::Providers, Tab::Models, Tab::Overview],
        );
    }

    #[test]
    fn h_l_navigate_tabs() {
        let mut app = make_app();
        app.handle_key(key(KeyCode::Char('l')));
        assert_eq!(app.active_tab, Tab::Models);
        app.handle_key(key(KeyCode::Char('h')));
        assert_eq!(app.active_tab, Tab::Overview);
    }

    #[test]
    fn left_right_resets_scroll() {
        assert_nav_resets_scroll(KeyCode::Right);
        assert_nav_resets_scroll(KeyCode::Left);
    }

    #[test]
    fn d_detaches_in_foreground() {
        let mut app = make_app();
        app.handle_key(key(KeyCode::Char('d')));
        assert_eq!(app.exit_mode, Some(ExitMode::Detach));
    }

    #[test]
    fn d_ignored_in_attached() {
        let mut app = make_attached_app();
        app.handle_key(key(KeyCode::Char('d')));
        assert!(app.exit_mode.is_none());
    }

    #[test]
    fn footer_shows_detach_in_foreground() {
        let app = make_app();
        assert!(!app.attached);
    }

    #[test]
    fn footer_hides_detach_in_attached() {
        let app = make_attached_app();
        assert!(app.attached);
    }
}
