pub mod anthropic;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod mcp;
pub mod metrics;
pub mod models;
pub mod responses;
pub mod system;
pub mod tokenize;
pub mod web;

#[cfg(test)]
mod tool_choice_tests;
