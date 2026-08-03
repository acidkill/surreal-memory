//! MCP server exposing the isolated spike database over stdio via rmcp.

use crate::surreal_engine::SurrealProcess;
use rmcp::Json;
use rmcp::{tool, tool_router, ServiceExt};
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize, JsonSchema)]
pub struct PingOutput {
    pub neuron_count: u64,
    pub source: String,
    pub version: String,
}

pub struct SmemServer {
    pub engine: Arc<SurrealProcess>,
    pub ns: String,
    pub db: String,
}

#[tool_router(server_handler)]
impl SmemServer {
    #[tool(
        name = "smem_ping",
        description = "Returns the neuron count in the isolated spike SurrealDB. Takes no parameters."
    )]
    pub async fn smem_ping(&self) -> Json<PingOutput> {
        let version = env!("CARGO_PKG_VERSION").to_string();
        let endpoint = self.engine.endpoint().to_string();
        let query_result = self
            .engine
            .run_query(
                &self.ns,
                &self.db,
                "RETURN array::len((SELECT VALUE id FROM neuron));",
            )
            .await;
        match query_result {
            Ok(value) => {
                let count = first_u64(&value).unwrap_or(0);
                Json(PingOutput {
                    neuron_count: count,
                    source: format!("isolated surreal @ {endpoint}"),
                    version,
                })
            }
            Err(error) => Json(PingOutput {
                neuron_count: 0,
                source: format!("surreal query error at {endpoint}: {error}"),
                version,
            }),
        }
    }
}

fn first_u64(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Number(number) => number.as_u64(),
        serde_json::Value::Array(items) => items.iter().find_map(first_u64),
        serde_json::Value::Object(map) => map.values().find_map(first_u64),
        _ => None,
    }
}

pub async fn serve(server: SmemServer) -> Result<(), Box<dyn std::error::Error>> {
    let (stdin, stdout) = rmcp::transport::io::stdio();
    let running = server.serve((stdin, stdout)).await?;
    running.waiting().await?;
    Ok(())
}
