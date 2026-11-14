//! The set of tools Higgs can execute on a model's behalf: those provided by
//! configured MCP servers.

use serde_json::{Value, json};

use crate::config::HiggsConfig;
use crate::mcp::config::process_env;
use crate::mcp::manager::{McpManager, ToolCallError, ToolInfo, ToolOutput};
use crate::web::WebService;

pub struct ToolHub {
    pub mcp: McpManager,
    pub web: WebService,
}

impl ToolHub {
    /// No MCP servers and no web access; starts nothing.
    pub fn disabled() -> Self {
        Self::new(McpManager::empty(), WebService::disabled())
    }

    pub const fn new(mcp: McpManager, web: WebService) -> Self {
        Self { mcp, web }
    }

    /// Start background connections for everything configured. Must run
    /// inside a Tokio runtime. Configuration problems are returned as errors
    /// so the server refuses to start with them.
    pub fn start(config: &HiggsConfig) -> Result<Self, String> {
        let mut problems = config.mcp.problems(&process_env);
        problems.extend(config.web.problems());
        if !problems.is_empty() {
            return Err(problems.join("; "));
        }
        Ok(Self::new(
            McpManager::start(&config.mcp, &process_env)?,
            WebService::new(&config.web)?,
        ))
    }

    /// Every tool that can currently be called: ready MCP servers first, then
    /// the built-in web tools when they are exposed.
    pub fn list(&self) -> Vec<ToolInfo> {
        let mut tools = self.mcp.tools();
        tools.extend(self.web.tools());
        tools
    }

    pub fn owns(&self, name: &str) -> bool {
        self.mcp.owns(name) || self.web.owns(name)
    }

    pub async fn call(&self, name: &str, arguments: Value) -> Result<ToolOutput, ToolCallError> {
        if self.web.owns(name) {
            return self.web.call_tool(name, &arguments).await;
        }
        self.mcp.call(name, arguments).await
    }

    pub async fn shutdown(&self) {
        self.mcp.shutdown().await;
    }
}

/// `OpenAI` function-tool form of a tool definition.
pub fn openai_tool(tool: &ToolInfo) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    })
}
