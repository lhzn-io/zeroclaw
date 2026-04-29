//! Built-in `tool_search` tool for on-demand tool schema loading.
//!
//! When deferred loading is enabled (either MCP via
//! `[mcp] deferred_loading = true` or native via
//! `[agent] lazy_load_native_tools = true`), this tool lets the LLM
//! discover and activate stubs whose full JSON schemas have been kept
//! out of the system prompt to save tokens. Supports two query modes:
//! - `select:name1,name2` — fetch exact tools by name. Native tools use
//!   bare names (`shell`); MCP tools use `server__tool` prefixes.
//! - Free-text keyword search — returns the best-matching stubs across
//!   both deferred sets, ranked by description-match count.

use std::fmt::Write;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::mcp_deferred::{ActivatedToolSet, DeferredMcpToolSet};
use crate::native_deferred::DeferredNativeToolSet;
use zeroclaw_api::tool::{Tool, ToolResult, ToolSpec};

/// Default maximum number of search results.
const DEFAULT_MAX_RESULTS: usize = 5;

/// Built-in tool that fetches full schemas for deferred tools.
/// Both MCP and native deferred sets are optional and orthogonal —
/// `tool_search` is useful as long as at least one is attached.
pub struct ToolSearchTool {
    mcp_deferred: Option<DeferredMcpToolSet>,
    native_deferred: Option<DeferredNativeToolSet>,
    activated: Arc<Mutex<ActivatedToolSet>>,
}

impl ToolSearchTool {
    /// Construct an empty `tool_search`. Use the builders below to
    /// attach deferred sets before registering this tool.
    pub fn new(activated: Arc<Mutex<ActivatedToolSet>>) -> Self {
        Self {
            mcp_deferred: None,
            native_deferred: None,
            activated,
        }
    }

    /// Builder: attach an MCP deferred set.
    pub fn with_mcp_deferred(mut self, mcp: DeferredMcpToolSet) -> Self {
        self.mcp_deferred = Some(mcp);
        self
    }

    /// Builder: attach a native-tool deferred set so `tool_search` can
    /// surface and activate native stubs as well as MCP ones.
    pub fn with_native_deferred(mut self, native: DeferredNativeToolSet) -> Self {
        self.native_deferred = Some(native);
        self
    }
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "tool_search"
    }

    fn description(&self) -> &str {
        "Fetch full schema definitions for deferred tools so they can be called. \
         Use \"select:name1,name2\" for direct selection, or keywords to search."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "description": "Query to find deferred tools. Use \"select:<tool_name>\" for direct selection, or keywords to search.",
                    "type": "string"
                },
                "max_results": {
                    "description": "Maximum number of results to return (default: 5)",
                    "type": "number",
                    "default": DEFAULT_MAX_RESULTS
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim();

        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|v| usize::try_from(v).unwrap_or(DEFAULT_MAX_RESULTS))
            .unwrap_or(DEFAULT_MAX_RESULTS);

        if query.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("query parameter is required".into()),
            });
        }

        // Parse query mode
        if let Some(names_str) = query.strip_prefix("select:") {
            // Exact selection mode
            let names: Vec<&str> = names_str.split(',').map(str::trim).collect();
            return self.select_tools(&names);
        }

        // Keyword search mode — search BOTH deferred sets and merge.
        let mcp_hits: Vec<&str> = self
            .mcp_deferred
            .as_ref()
            .map(|m| m.search(query, max_results))
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.prefixed_name.as_str())
            .collect();
        let native_hits: Vec<&str> = self
            .native_deferred
            .as_ref()
            .map(|n| n.search(query, max_results))
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.name.as_str())
            .collect();

        // Interleave MCP and native results, then trim to max_results.
        // Per-set ordering preserves each set's rank-by-match-count.
        let mut merged: Vec<&str> = Vec::with_capacity(mcp_hits.len() + native_hits.len());
        let (mut i, mut j) = (0, 0);
        while merged.len() < max_results && (i < mcp_hits.len() || j < native_hits.len()) {
            if i < mcp_hits.len() {
                merged.push(mcp_hits[i]);
                i += 1;
                if merged.len() >= max_results {
                    break;
                }
            }
            if j < native_hits.len() {
                merged.push(native_hits[j]);
                j += 1;
            }
        }

        if merged.is_empty() {
            return Ok(ToolResult {
                success: true,
                output: "No matching deferred tools found.".into(),
                error: None,
            });
        }

        let mut output = String::from("<functions>\n");
        let mut activated_count = 0;
        let mut guard = self.activated.lock().unwrap();
        for name in &merged {
            if self.activate_and_emit(name, &mut guard, &mut output) {
                activated_count += 1;
            }
        }
        output.push_str("</functions>\n");
        drop(guard);

        tracing::debug!(
            "tool_search: query={query:?}, matched={}, activated={activated_count}",
            merged.len()
        );

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }
}

impl ToolSearchTool {
    fn select_tools(&self, names: &[&str]) -> anyhow::Result<ToolResult> {
        let mut output = String::from("<functions>\n");
        let mut not_found = Vec::new();
        let mut activated_count = 0;
        let mut guard = self.activated.lock().unwrap();

        for name in names {
            if name.is_empty() {
                continue;
            }
            if self.activate_and_emit(name, &mut guard, &mut output) {
                activated_count += 1;
            } else if !guard.is_activated(name) {
                // Tool wasn't found in either deferred set AND wasn't
                // already activated — report as not found.
                not_found.push(*name);
            }
        }

        output.push_str("</functions>\n");
        drop(guard);

        if !not_found.is_empty() {
            let _ = write!(output, "\nNot found: {}", not_found.join(", "));
        }

        tracing::debug!(
            "tool_search select: requested={}, activated={activated_count}, not_found={}",
            names.len(),
            not_found.len()
        );

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }

    /// Resolve `name` against both deferred sets (native first, then
    /// MCP), activate if not already, and emit its `<function>` block
    /// to `output`. Returns `true` when a fresh activation happened.
    /// Returns `false` if the name wasn't found in either set; the
    /// caller decides whether that's an error (select mode) or
    /// silently skipped (search mode hits already filter to known).
    fn activate_and_emit(
        &self,
        name: &str,
        guard: &mut std::sync::MutexGuard<'_, ActivatedToolSet>,
        output: &mut String,
    ) -> bool {
        // Native first — bare names land here.
        if let Some(native) = &self.native_deferred {
            if let Some(spec) = native.tool_spec(name) {
                let mut activated = false;
                if !guard.is_activated(name) {
                    if let Some(tool_arc) = native.activate(name) {
                        guard.activate(name.to_string(), tool_arc);
                        activated = true;
                    }
                }
                emit_function_xml(output, &spec);
                return activated;
            }
        }
        // MCP — prefixed names land here.
        if let Some(mcp) = &self.mcp_deferred {
            if let Some(spec) = mcp.tool_spec(name) {
                let mut activated = false;
                if !guard.is_activated(name) {
                    if let Some(tool) = mcp.activate(name) {
                        guard.activate(name.to_string(), Arc::from(tool));
                        activated = true;
                    }
                }
                emit_function_xml(output, &spec);
                return activated;
            }
        }
        false
    }
}

/// Render a `ToolSpec` into the `<function>{...}</function>` XML form
/// the LLM consumes after activation. Extracted so both the keyword-
/// search and `select:` paths emit identical formatting.
fn emit_function_xml(output: &mut String, spec: &ToolSpec) {
    let _ = writeln!(
        output,
        "<function>{{\"name\": \"{}\", \"description\": \"{}\", \"parameters\": {}}}</function>",
        spec.name,
        spec.description.replace('"', "\\\""),
        spec.parameters
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_client::McpRegistry;
    use crate::mcp_deferred::DeferredMcpToolStub;
    use crate::mcp_protocol::McpToolDef;

    async fn make_deferred_set(stubs: Vec<DeferredMcpToolStub>) -> DeferredMcpToolSet {
        let registry = Arc::new(McpRegistry::connect_all(&[]).await.unwrap());
        DeferredMcpToolSet { stubs, registry }
    }

    fn make_stub(name: &str, desc: &str) -> DeferredMcpToolStub {
        let def = McpToolDef {
            name: name.to_string(),
            description: Some(desc.to_string()),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };
        DeferredMcpToolStub::new(name.to_string(), def)
    }

    #[tokio::test]
    async fn tool_metadata() {
        let tool = ToolSearchTool::new(Arc::new(Mutex::new(ActivatedToolSet::new())))
            .with_mcp_deferred(make_deferred_set(vec![]).await);
        assert_eq!(tool.name(), "tool_search");
        assert!(!tool.description().is_empty());
        assert!(tool.parameters_schema()["properties"]["query"].is_object());
    }

    #[tokio::test]
    async fn empty_query_returns_error() {
        let tool = ToolSearchTool::new(Arc::new(Mutex::new(ActivatedToolSet::new())))
            .with_mcp_deferred(make_deferred_set(vec![]).await);
        let result = tool
            .execute(serde_json::json!({"query": ""}))
            .await
            .unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn select_nonexistent_tool_reports_not_found() {
        let tool = ToolSearchTool::new(Arc::new(Mutex::new(ActivatedToolSet::new())))
            .with_mcp_deferred(make_deferred_set(vec![]).await);
        let result = tool
            .execute(serde_json::json!({"query": "select:nonexistent"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("Not found"));
    }

    #[tokio::test]
    async fn keyword_search_no_matches() {
        let tool = ToolSearchTool::new(Arc::new(Mutex::new(ActivatedToolSet::new())))
            .with_mcp_deferred(make_deferred_set(vec![make_stub("fs__read", "Read file")]).await);
        let result = tool
            .execute(serde_json::json!({"query": "zzzzz_nonexistent"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("No matching"));
    }

    #[tokio::test]
    async fn keyword_search_finds_match() {
        let activated = Arc::new(Mutex::new(ActivatedToolSet::new()));
        let tool = ToolSearchTool::new(Arc::clone(&activated))
            .with_mcp_deferred(make_deferred_set(vec![make_stub("fs__read", "Read a file from disk")]).await);
        let result = tool
            .execute(serde_json::json!({"query": "read file"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("<function>"));
        assert!(result.output.contains("fs__read"));
        // Tool should now be activated
        assert!(activated.lock().unwrap().is_activated("fs__read"));
    }

    /// Verify tool_search works with stubs from multiple MCP servers,
    /// simulating a daemon-mode setup where several servers are deferred.
    #[tokio::test]
    async fn multiple_servers_stubs_all_searchable() {
        let activated = Arc::new(Mutex::new(ActivatedToolSet::new()));
        let stubs = vec![
            make_stub("server_a__list_files", "List files on server A"),
            make_stub("server_a__read_file", "Read file on server A"),
            make_stub("server_b__query_db", "Query database on server B"),
            make_stub("server_b__insert_row", "Insert row on server B"),
        ];
        let tool = ToolSearchTool::new(Arc::clone(&activated))
            .with_mcp_deferred(make_deferred_set(stubs).await);

        // Search should find tools across both servers
        let result = tool
            .execute(serde_json::json!({"query": "file"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("server_a__list_files"));
        assert!(result.output.contains("server_a__read_file"));

        // Server B tools should also be searchable
        let result = tool
            .execute(serde_json::json!({"query": "database query"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("server_b__query_db"));
    }

    /// Verify select mode activates tools and they stay activated across calls,
    /// matching the daemon-mode pattern where a single ActivatedToolSet persists.
    #[tokio::test]
    async fn select_activates_and_persists_across_calls() {
        let activated = Arc::new(Mutex::new(ActivatedToolSet::new()));
        let stubs = vec![
            make_stub("srv__tool_a", "Tool A"),
            make_stub("srv__tool_b", "Tool B"),
        ];
        let tool = ToolSearchTool::new(Arc::clone(&activated))
            .with_mcp_deferred(make_deferred_set(stubs).await);

        // Activate tool_a
        let result = tool
            .execute(serde_json::json!({"query": "select:srv__tool_a"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(activated.lock().unwrap().is_activated("srv__tool_a"));
        assert!(!activated.lock().unwrap().is_activated("srv__tool_b"));

        // Activate tool_b in a separate call
        let result = tool
            .execute(serde_json::json!({"query": "select:srv__tool_b"}))
            .await
            .unwrap();
        assert!(result.success);

        // Both should remain activated
        let guard = activated.lock().unwrap();
        assert!(guard.is_activated("srv__tool_a"));
        assert!(guard.is_activated("srv__tool_b"));
        assert_eq!(guard.tool_specs().len(), 2);
    }

    /// Verify re-activating an already-activated tool does not duplicate it.
    #[tokio::test]
    async fn reactivation_is_idempotent() {
        let activated = Arc::new(Mutex::new(ActivatedToolSet::new()));
        let tool = ToolSearchTool::new(Arc::clone(&activated))
            .with_mcp_deferred(make_deferred_set(vec![make_stub("srv__tool", "A tool")]).await);

        tool.execute(serde_json::json!({"query": "select:srv__tool"}))
            .await
            .unwrap();
        tool.execute(serde_json::json!({"query": "select:srv__tool"}))
            .await
            .unwrap();

        assert_eq!(activated.lock().unwrap().tool_specs().len(), 1);
    }
}
