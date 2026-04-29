//! Deferred-loading support for *native* (built-in Rust) tools.
//!
//! This mirrors [`crate::mcp_deferred::DeferredMcpToolSet`] for tools
//! that live inside the zeroclaw binary — `shell`, `web_search`,
//! `http_request`, the integrations gated by `[xxx] enabled = true`,
//! etc. Stubs (name + description only) ship in the system prompt;
//! full JSON parameter schemas only enter the LLM's context window
//! after the agent calls `tool_search` to activate them.
//!
//! Native stubs use *bare* names (`shell`, `web_search`); MCP stubs use
//! `server__tool` prefixes. The two namespaces don't collide in
//! practice and both flow through the same `tool_search` and
//! [`crate::mcp_deferred::ActivatedToolSet`] machinery.

use std::sync::Arc;

use zeroclaw_api::tool::{Tool, ToolSpec};

/// A native built-in tool registered as a stub. Holds an `Arc<dyn Tool>`
/// directly so activation is a cheap clone rather than a re-construct.
#[derive(Clone)]
pub struct DeferredNativeToolStub {
    /// Bare tool name (matches what the LLM will call once activated).
    pub name: String,
    /// Human-readable description (extracted from the tool's
    /// [`Tool::description`] at stub-construction time).
    pub description: String,
    /// The live tool, kept warm so activation is `Arc::clone`.
    tool: Arc<dyn Tool>,
}

impl DeferredNativeToolStub {
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        let name = tool.name().to_string();
        let description = tool.description().to_string();
        Self {
            name,
            description,
            tool,
        }
    }

    /// Return the underlying [`Arc<dyn Tool>`] — used by `tool_search`
    /// to insert into [`crate::mcp_deferred::ActivatedToolSet`].
    pub fn activate(&self) -> Arc<dyn Tool> {
        Arc::clone(&self.tool)
    }
}

impl std::fmt::Debug for DeferredNativeToolStub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeferredNativeToolStub")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Collection of native tool stubs discovered at agent startup.
/// Provides keyword search and exact-name lookup parallel to
/// [`crate::mcp_deferred::DeferredMcpToolSet`].
#[derive(Default, Clone)]
pub struct DeferredNativeToolSet {
    pub stubs: Vec<DeferredNativeToolStub>,
}

impl DeferredNativeToolSet {
    /// Build a set from a list of live tools. The tools' names and
    /// descriptions are captured for the stub list.
    pub fn from_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let stubs = tools.into_iter().map(DeferredNativeToolStub::new).collect();
        Self { stubs }
    }

    /// All stub names — used by the system-prompt section builder.
    pub fn stub_names(&self) -> Vec<&str> {
        self.stubs.iter().map(|s| s.name.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.stubs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stubs.is_empty()
    }

    /// Look up a stub by exact name. Used for `select:name1,name2`.
    pub fn get_by_name(&self, name: &str) -> Option<&DeferredNativeToolStub> {
        self.stubs.iter().find(|s| s.name == name)
    }

    /// Keyword search — case-insensitive AND across name + description,
    /// ranked by number of matching terms. Mirrors the semantics of
    /// [`crate::mcp_deferred::DeferredMcpToolSet::search`].
    pub fn search(&self, query: &str, max_results: usize) -> Vec<&DeferredNativeToolStub> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        if terms.is_empty() {
            return self.stubs.iter().take(max_results).collect();
        }

        let mut scored: Vec<(&DeferredNativeToolStub, usize)> = self
            .stubs
            .iter()
            .filter_map(|stub| {
                let haystack = format!(
                    "{} {}",
                    stub.name.to_ascii_lowercase(),
                    stub.description.to_ascii_lowercase()
                );
                let hits = terms
                    .iter()
                    .filter(|t| haystack.contains(t.as_str()))
                    .count();
                if hits > 0 { Some((stub, hits)) } else { None }
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored
            .into_iter()
            .take(max_results)
            .map(|(s, _)| s)
            .collect()
    }

    /// Materialize a stub as a live [`Arc<dyn Tool>`].
    pub fn activate(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.get_by_name(name).map(|stub| stub.activate())
    }

    /// Return the full [`ToolSpec`] for a stub. Used by `tool_search`
    /// to emit schemas to the LLM after activation.
    pub fn tool_spec(&self, name: &str) -> Option<ToolSpec> {
        self.get_by_name(name).map(|stub| stub.tool.spec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use zeroclaw_api::tool::ToolResult;

    struct StubTool {
        name: String,
        description: String,
    }

    #[async_trait]
    impl Tool for StubTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.description
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                success: true,
                output: String::new(),
                error: None,
            })
        }
    }

    fn make_tool(name: &str, desc: &str) -> Arc<dyn Tool> {
        Arc::new(StubTool {
            name: name.into(),
            description: desc.into(),
        })
    }

    #[test]
    fn stub_captures_name_and_description() {
        let stub = DeferredNativeToolStub::new(make_tool("shell", "Run shell commands"));
        assert_eq!(stub.name, "shell");
        assert_eq!(stub.description, "Run shell commands");
    }

    #[test]
    fn from_tools_builds_set_with_all_stubs() {
        let set = DeferredNativeToolSet::from_tools(vec![
            make_tool("shell", "Run shell commands"),
            make_tool("web_search", "Search the web"),
        ]);
        assert_eq!(set.len(), 2);
        let names = set.stub_names();
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"web_search"));
    }

    #[test]
    fn get_by_name_finds_exact_match() {
        let set = DeferredNativeToolSet::from_tools(vec![make_tool("shell", "Run shell")]);
        assert!(set.get_by_name("shell").is_some());
        assert!(set.get_by_name("nope").is_none());
    }

    #[test]
    fn search_finds_by_description_keyword() {
        let set = DeferredNativeToolSet::from_tools(vec![
            make_tool("shell", "Run shell commands"),
            make_tool("web_search", "Search the web for results"),
        ]);
        let hits = set.search("search", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "web_search");
    }

    #[test]
    fn search_ranks_by_match_count() {
        let set = DeferredNativeToolSet::from_tools(vec![
            make_tool("shell", "Run a shell command"),
            make_tool("web_search", "Search the web shell-like"),
        ]);
        let hits = set.search("shell command", 5);
        // shell matches both "shell" and "command" → 2 hits
        // web_search matches only "shell" → 1 hit
        assert_eq!(hits[0].name, "shell");
    }

    #[test]
    fn activate_returns_clone_of_arc() {
        let set = DeferredNativeToolSet::from_tools(vec![make_tool("shell", "Run shell")]);
        let live = set.activate("shell").expect("activates");
        assert_eq!(live.name(), "shell");
    }

    #[test]
    fn tool_spec_returns_full_schema() {
        let set = DeferredNativeToolSet::from_tools(vec![make_tool("shell", "Run shell")]);
        let spec = set.tool_spec("shell").expect("has spec");
        assert_eq!(spec.name, "shell");
        assert!(spec.parameters.is_object());
    }
}
