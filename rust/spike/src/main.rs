//! smem-rust-spike — Faza-0 spike.
//!
//! Proves, against an isolated SurrealDB v3.2 engine (SurrealKV backend):
//!   * GREEN A — vector KNN nearest-first ordering via an HNSW index.
//!   * GREEN B — multi-hop graph traversal a->b->c over `synapse` edges.
//!   * GREEN C — an MCP server (rmcp, stdio) exposing `smem_ping` returning
//!     the live neuron count of the isolated database.
//!
//! Modes:
//!   `smem-spike`          seed + run GREEN A/B, print evidence, exit.
//!   `smem-spike --mcp`    seed + serve the MCP server over stdio (GREEN C).
//!
//! The spike binary manages its own isolated `surreal` child process
//! (loopback bind, ephemeral SurrealKV file). It does NOT contact the
//! production SurrealDB instance or the default brain.

mod mcp;
mod surreal_engine;

use std::sync::Arc;
use surreal_engine::SurrealProcess;

const NS: &str = "smem";
const DB: &str = "spike";
const PORT: u16 = 61801;

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

fn spike_db_path() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push("smem-rust-spike/spike.skv");
    path
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp_mode = std::env::args().any(|arg| arg == "--mcp");

    let db_path = spike_db_path();
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_dir_all(&db_path);

    let engine = SurrealProcess::spawn(PORT, &db_path).await?;
    let endpoint = engine.endpoint().to_string();
    eprintln!(
        "[spike] isolated surreal ready at {endpoint} (ns={NS} db={DB}, file={})",
        db_path.display()
    );

    engine.run_script(NS, DB, SEED_SQL).await?;
    eprintln!("[spike] seeded: 3 neurons (a,b,c) + 2 synapse edges (a->b, b->c)");

    if mcp_mode {
        mcp::serve(mcp::SmemServer {
            engine: Arc::new(engine),
            ns: NS.to_string(),
            db: DB.to_string(),
        })
        .await
    } else {
        run_verify(&engine).await
    }
}

async fn run_verify(engine: &SurrealProcess) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== GREEN A: vector KNN nearest-first (HNSW index, <|2,COSINE|>) ===");
    let knn = engine
        .run_query(
            NS,
            DB,
            "SELECT id, vector::distance::knn() AS s FROM neuron WHERE embedding_vec <|2,COSINE|> [0.9,0.1,0.0,0.0] ORDER BY s;",
        )
        .await?;
    println!("[GREEN A] raw KNN result: {knn}");
    let mut ordered_ids = Vec::new();
    collect_neuron_ids(&knn, &mut ordered_ids);
    println!("[GREEN A] ordered nearest-first ids: {ordered_ids:?}");
    assert!(ordered_ids.len() >= 2, "KNN must return at least 2 results");
    assert_eq!(
        ordered_ids[0], "neuron:a",
        "nearest neighbor must be neuron:a"
    );
    assert_eq!(
        ordered_ids[1], "neuron:c",
        "second neighbor must be neuron:c"
    );
    println!("[GREEN A] PASS — nearest-first order asserted: neuron:a then neuron:c\n");

    println!("=== GREEN B: graph path a->b->c via synapse edges (arrow traversal) ===");
    let two_hop = engine
        .run_query(
            NS,
            DB,
            "SELECT ->synapse->neuron->synapse->neuron.id AS two_hop FROM neuron:a;",
        )
        .await?;
    println!("[GREEN B] raw 2-hop traversal: {two_hop}");
    let mut reached = Vec::new();
    collect_neuron_ids(&two_hop, &mut reached);
    println!("[GREEN B] ids reachable from neuron:a within 2 synapse hops: {reached:?}");
    assert!(
        reached.iter().any(|id| id == "neuron:c"),
        "neuron:c must be reachable from neuron:a within 2 hops"
    );
    println!("[GREEN B] PASS — graph path neuron:a -> neuron:b -> neuron:c confirmed\n");

    println!("=== SPIKE VERIFY COMPLETE: GREEN A + GREEN B PASSED ===");
    Ok(())
}

fn collect_neuron_ids(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) => {
            if text.starts_with("neuron:") {
                out.push(text.clone());
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_neuron_ids(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                collect_neuron_ids(value, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn knn_returns_nearest_first_ordering() {
        let db_path = spike_db_path();
        let _ = std::fs::remove_dir_all(&db_path);
        let engine = SurrealProcess::spawn(PORT, &db_path)
            .await
            .expect("isolated surreal must spawn");
        engine
            .run_script(NS, DB, SEED_SQL)
            .await
            .expect("seed must succeed");

        let knn = engine
            .run_query(
                NS,
                DB,
                "SELECT id, vector::distance::knn() AS s FROM neuron WHERE embedding_vec <|2,COSINE|> [0.9,0.1,0.0,0.0] ORDER BY s;",
            )
            .await
            .expect("knn query must succeed");

        let mut ordered_ids = Vec::new();
        collect_neuron_ids(&knn, &mut ordered_ids);
        assert!(
            ordered_ids.len() >= 2,
            "expected >=2 knn results, got {ordered_ids:?}"
        );
        assert_eq!(ordered_ids[0], "neuron:a");
        assert_eq!(ordered_ids[1], "neuron:c");
    }
}
