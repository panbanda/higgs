//! The set of tools Higgs can execute on a model's behalf: those provided by
//! configured MCP servers.

use serde_json::{Value, json};

use crate::config::HiggsConfig;
use crate::mcp::config::process_env;
use crate::mcp::manager::{McpManager, ToolCallError, ToolInfo, ToolOutput};

pub struct ToolHub {
    pub mcp: McpManager,
}

impl ToolHub {
    /// No MCP servers; starts nothing.
    pub fn disabled() -> Self {
        Self::new(McpManager::empty())
    }

    pub const fn new(mcp: McpManager) -> Self {
        Self { mcp }
    }

    /// Start background connections for everything configured. Must run
    /// inside a Tokio runtime. Configuration problems are returned as errors
    /// so the server refuses to start with them.
    pub fn start(config: &HiggsConfig) -> Result<Self, String> {
        let problems = config.mcp.problems(&process_env);
        if !problems.is_empty() {
            return Err(problems.join("; "));
        }
        Ok(Self::new(McpManager::start(&config.mcp, &process_env)?))
    }

    pub fn list(&self) -> Vec<ToolInfo> {
        self.mcp.tools()
    }

    pub fn owns(&self, name: &str) -> bool {
        self.mcp.owns(name)
    }

    pub async fn call(&self, name: &str, arguments: Value) -> Result<ToolOutput, ToolCallError> {
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
