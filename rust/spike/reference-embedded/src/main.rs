//! Reference in-process spike using the official `surrealdb` Rust crate.
//!
//! Goal of the rewrite feasibility spike: embed SurrealDB (SurrealKV engine)
//! directly inside the process via `surrealdb::engine::any::connect` with the
//! `kv-surrealkv` feature, run the same vector KNN + graph traversal as the
//! shell-driven spike, and expose `smem_ping` over rmcp stdio.
//!
//! It mirrors the SurrealQL proven against SurrealDB v3.2 by the offline
//! spike in this directory. Build it in an environment with crates.io access:
//!     cd rust/spike/reference-embedded && cargo run

use rmcp::Json;
use rmcp::{tool, tool_router, ServiceExt};
use schemars::JsonSchema;
use serde::Serialize;
use surrealdb::engine::any::connect;
use surrealdb::Surreal;

const NS: &str = "smem";
const DB: &str = "spike";
const SEED_SQL: &str = "\
DEFINE TABLE neuron SCHEMALESS;\
DEFINE INDEX neuron_embedding_idx ON neuron FIELDS embedding_vec HNSW DIMENSION 4 DIST COSINE TYPE F32;\
DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron;\
CREATE neuron:a SET embedding_vec = [1.0, 0.0, 0.0, 0.0];\
CREATE neuron:b SET embedding_vec = [0.0, 1.0, 0.0, 0.0];\
CREATE neuron:c SET embedding_vec = [0.7, 0.7, 0.0, 0.0];\
RELATE neuron:a->synapse->neuron:b SET weight = 0.9;\
RELATE neuron:b->synapse->neuron:c SET weight = 0.8;\
";

#[derive(Serialize, JsonSchema)]
pub struct PingOutput {
    pub neuron_count: u64,
    pub source: String,
}

pub struct SmemServer {
    pub db: Surreal<surrealdb::engine::any::Any>,
}

#[tool_router(server_handler)]
impl SmemServer {
    #[tool(name = "smem_ping", description = "Returns the neuron count in the embedded spike database.")]
    pub async fn smem_ping(&self) -> Json<PingOutput> {
        let result: Result<Option<u64>, _> = self
            .db
            .query("RETURN array::len((SELECT VALUE id FROM neuron));")
            .await
            .and_then(|mut response| response.take::<Option<u64>>(0));
        match result {
            Ok(count) => Json(PingOutput {
                neuron_count: count.unwrap_or(0),
                source: "embedded surrealdb (surrealkv)".to_string(),
            }),
            Err(error) => Json(PingOutput {
                neuron_count: 0,
                source: format!("embedded surrealdb error: {error}"),
            }),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // In-process embedding (context7-verified for surrealdb crate v2):
    // canonical v2 API is `engine::any::connect("surrealkv://<dir>")`, which
    // routes by URL scheme to the SurrealKV engine (cargo feature
    // `kv-surrealkv`). The goal's `Surreal::new::<SurrealKv>(path)` is the
    // v1.x typed-engine form; v2 unified every backend behind `connect`.
    // Embedded engines (memory, surrealkv, rocksdb) need no root signin by
    // default. Use a fresh dir each run so the seed is deterministic.
    let db: Surreal<surrealdb::engine::any::Any> = connect("surrealkv://./spike.db").await?;
    db.use_ns(NS).use_db(DB).await?;
    db.query(SEED_SQL).await?;

    let knn: Vec<(String, f64)> = db
        .query(
            "SELECT id, vector::distance::knn() AS s FROM neuron WHERE embedding_vec <|2,COSINE|> [0.9,0.1,0.0,0.0] ORDER BY s;",
        )
        .await?
        .take(0)?;
    println!("[reference] KNN nearest-first: {knn:?}");

    let two_hop: Vec<String> = db
        .query("SELECT VALUE ->synapse->neuron->synapse->neuron.id FROM neuron:a;")
        .await?
        .take(0)?;
    println!("[reference] 2-hop reach from neuron:a: {two_hop:?}");

    let server = SmemServer { db };
    let (stdin, stdout) = rmcp::transport::io::stdio();
    server.serve((stdin, stdout)).await?.waiting().await?;
    Ok(())
}
