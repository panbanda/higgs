pub mod views;

use std::io;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};

use crate::config::HiggsConfig;
use crate::metrics::MetricsStore;

// ---------------------------------------------------------------------------
// TuiConfig -- config data passed to the TUI for display
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TuiConfig {
    pub profile: Option<String>,
    pub model_names: Vec<String>,
    pub provider_names: Vec<String>,
    pub routes: Vec<TuiRoute>,
    pub default_provider: String,
    pub auto_router: TuiAutoRouter,
}

#[derive(Debug, Clone)]
pub struct TuiRoute {
    pub pattern: Option<String>,
    pub provider: String,
    pub model_rewrite: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TuiAutoRouter {
    pub enabled: bool,
    pub force: bool,
    pub model: String,
    pub timeout_ms: u64,
}

impl TuiConfig {
    pub fn from_higgs_config(config: &HiggsConfig, profile: Option<&str>) -> Self {
        let model_names: Vec<String> = config
            .models
            .iter()
            .map(|m| {
                m.name
                    .clone()
                    .unwrap_or_else(|| m.path.rsplit('/').next().unwrap_or(&m.path).to_owned())
            })
            .collect();

        let mut provider_names: Vec<String> = config.providers.keys().cloned().collect();
        if !config.models.is_empty() && !provider_names.contains(&"higgs".to_owned()) {
            provider_names.push("higgs".to_owned());
        }
        provider_names.sort();

        let routes: Vec<TuiRoute> = config
            .routes
            .iter()
            .map(|r| TuiRoute {
                pattern: r.pattern.clone(),
                provider: r.provider.clone(),
                model_rewrite: r.model.clone(),
                name: r.name.clone(),
                description: r.description.clone(),
            })
            .collect();

        Self {
            profile: profile.map(String::from),
            model_names,
            provider_names,
            routes,
            default_provider: config.default.provider.clone(),
            auto_router: TuiAutoRouter {
                enabled: config.auto_router.enabled,
                force: config.auto_router.force,
                model: config.auto_router.model.clone(),
                timeout_ms: config.auto_router.timeout_ms,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Models,
    Providers,
    Errors,
    Routing,
}

impl Tab {
    fn titles() -> Vec<&'static str> {
        vec![
            "Overview [1]",
            "Models [2]",
            "Providers [3]",
            "Errors [4]",
            "Routing [5]",
        ]
    }

    const fn index(self) -> usize {
        match self {
            Self::Overview => 0,
            Self::Models => 1,
            Self::Providers => 2,
            Self::Errors => 3,
            Self::Routing => 4,
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
    pub config: Option<TuiConfig>,
}

impl App {
    pub const fn new(
        metrics: Arc<MetricsStore>,
        attached: bool,
        config: Option<TuiConfig>,
    ) -> Self {
        Self {
            metrics,
            active_tab: Tab::Overview,
            scroll_offset: 0,
            exit_mode: None,
            attached,
            config,
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
            KeyCode::Char('5') => {
                self.active_tab = Tab::Routing;
                self.scroll_offset = 0;
            }
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                self.active_tab = match self.active_tab {
                    Tab::Overview => Tab::Models,
                    Tab::Models => Tab::Providers,
                    Tab::Providers => Tab::Errors,
                    Tab::Errors => Tab::Routing,
                    Tab::Routing => Tab::Overview,
                };
                self.scroll_offset = 0;
            }
            KeyCode::Left | KeyCode::Char('h') => {
                self.active_tab = match self.active_tab {
                    Tab::Overview => Tab::Routing,
                    Tab::Models => Tab::Overview,
                    Tab::Providers => Tab::Models,
                    Tab::Errors => Tab::Providers,
                    Tab::Routing => Tab::Errors,
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
        let profile_tag = self
            .config
            .as_ref()
            .and_then(|c| c.profile.as_ref())
            .map(|p| format!(" [{p}]"))
            .unwrap_or_default();

        let title = if self.attached {
            format!(" higgs{profile_tag} (attached) ")
        } else {
            format!(" higgs{profile_tag} ")
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
                let names = self.config.as_ref().map(|c| c.model_names.as_slice());
                views::models::draw(
                    frame,
                    content_area,
                    &self.metrics,
                    self.scroll_offset,
                    names,
                );
            }
            Tab::Providers => {
                let names = self.config.as_ref().map(|c| c.provider_names.as_slice());
                views::providers::draw(
                    frame,
                    content_area,
                    &self.metrics,
                    self.scroll_offset,
                    names,
                );
            }
            Tab::Errors => {
                views::errors::draw(frame, content_area, &self.metrics, self.scroll_offset);
            }
            Tab::Routing => {
                views::routing::draw(frame, content_area, self.config.as_ref());
            }
        }

        let footer = Paragraph::new(Line::from(vec![Span::styled(
            hint,
            Style::default().fg(Color::DarkGray),
        )]));
        frame.render_widget(footer, chunks[2]);
    }
}

pub fn run(
    metrics: Arc<MetricsStore>,
    attached: bool,
    config: Option<TuiConfig>,
) -> io::Result<ExitMode> {
    let mut terminal = ratatui::init();

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        default_hook(info);
    }));

    let mut app = App::new(metrics, attached, config);

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
        App::new(
            Arc::new(MetricsStore::new(Duration::from_secs(60))),
            false,
            None,
        )
    }

    fn make_attached_app() -> App {
        App::new(
            Arc::new(MetricsStore::new(Duration::from_secs(60))),
            true,
            None,
        )
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
            ('5', Tab::Routing),
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
            &[
                Tab::Models,
                Tab::Providers,
                Tab::Errors,
                Tab::Routing,
                Tab::Overview,
            ],
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
            &[
                Tab::Models,
                Tab::Providers,
                Tab::Errors,
                Tab::Routing,
                Tab::Overview,
            ],
        );
    }

    #[test]
    fn left_arrow_cycles_backward() {
        assert_tab_cycle(
            KeyCode::Left,
            &[
                Tab::Routing,
                Tab::Errors,
                Tab::Providers,
                Tab::Models,
                Tab::Overview,
            ],
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

    // -- TuiConfig::from_higgs_config tests -----------------------------------

    use crate::config::{
        AutoRouterConfig, DefaultRoute, HiggsConfig, ModelConfig, ProviderConfig, RouteConfig,
    };
    use std::collections::HashMap;

    fn minimal_config() -> HiggsConfig {
        HiggsConfig {
            models: vec![ModelConfig {
                path: "mlx-community/Llama-3.2-1B-Instruct-4bit".to_owned(),
                name: None,
                batch: false,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
            }],
            ..HiggsConfig::default()
        }
    }

    #[test]
    fn from_config_extracts_model_name_from_path() {
        let config = minimal_config();
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert_eq!(tui.model_names, vec!["Llama-3.2-1B-Instruct-4bit"]);
    }

    #[test]
    fn from_config_uses_explicit_model_name() {
        let mut config = minimal_config();
        config.models[0].name = Some("llama".to_owned());
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert_eq!(tui.model_names, vec!["llama"]);
    }

    #[test]
    fn from_config_injects_higgs_provider_when_models_exist() {
        let config = minimal_config();
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert!(tui.provider_names.contains(&"higgs".to_owned()));
    }

    #[test]
    fn from_config_no_higgs_provider_without_models() {
        let config = HiggsConfig::default();
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert!(!tui.provider_names.contains(&"higgs".to_owned()));
    }

    #[test]
    fn from_config_includes_remote_providers() {
        let mut config = minimal_config();
        config.providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert!(tui.provider_names.contains(&"anthropic".to_owned()));
        assert!(tui.provider_names.contains(&"higgs".to_owned()));
    }

    #[test]
    fn from_config_providers_are_sorted() {
        let mut config = minimal_config();
        let mut providers = HashMap::new();
        providers.insert(
            "openai".to_owned(),
            ProviderConfig {
                url: "https://api.openai.com".to_owned(),
                format: crate::config::ApiFormat::OpenAi,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        config.providers = providers;
        let tui = TuiConfig::from_higgs_config(&config, None);
        let expected = vec!["anthropic", "higgs", "openai"];
        assert_eq!(tui.provider_names, expected);
    }

    #[test]
    fn from_config_maps_routes() {
        let mut config = minimal_config();
        config.routes = vec![RouteConfig {
            pattern: Some("claude-.*".to_owned()),
            provider: "anthropic".to_owned(),
            model: Some("claude-sonnet-4-6".to_owned()),
            name: Some("Claude".to_owned()),
            description: Some("Anthropic models".to_owned()),
        }];
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert_eq!(tui.routes.len(), 1);
        let r = &tui.routes[0];
        assert_eq!(r.pattern.as_deref(), Some("claude-.*"));
        assert_eq!(r.provider, "anthropic");
        assert_eq!(r.model_rewrite.as_deref(), Some("claude-sonnet-4-6"));
        assert_eq!(r.name.as_deref(), Some("Claude"));
        assert_eq!(r.description.as_deref(), Some("Anthropic models"));
    }

    #[test]
    fn from_config_copies_default_provider() {
        let mut config = minimal_config();
        config.default = DefaultRoute {
            provider: "openai".to_owned(),
        };
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert_eq!(tui.default_provider, "openai");
    }

    #[test]
    fn from_config_copies_auto_router() {
        let mut config = minimal_config();
        config.auto_router = AutoRouterConfig {
            enabled: true,
            force: true,
            model: "my-router".to_owned(),
            timeout_ms: 5000,
        };
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert!(tui.auto_router.enabled);
        assert!(tui.auto_router.force);
        assert_eq!(tui.auto_router.model, "my-router");
        assert_eq!(tui.auto_router.timeout_ms, 5000);
    }

    #[test]
    fn from_config_passes_profile() {
        let config = minimal_config();
        let tui = TuiConfig::from_higgs_config(&config, Some("dev"));
        assert_eq!(tui.profile.as_deref(), Some("dev"));
    }

    #[test]
    fn from_config_none_profile() {
        let config = minimal_config();
        let tui = TuiConfig::from_higgs_config(&config, None);
        assert!(tui.profile.is_none());
    }

    #[test]
    fn from_config_no_duplicate_higgs_when_already_a_provider() {
        let mut config = minimal_config();
        config.providers.insert(
            "higgs".to_owned(),
            ProviderConfig {
                url: "http://localhost:8000".to_owned(),
                format: crate::config::ApiFormat::OpenAi,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let tui = TuiConfig::from_higgs_config(&config, None);
        let higgs_count = tui
            .provider_names
            .iter()
            .filter(|n| n.as_str() == "higgs")
            .count();
        assert_eq!(higgs_count, 1);
    }
}
