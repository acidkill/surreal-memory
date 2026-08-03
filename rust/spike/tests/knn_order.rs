//! Integration assertion: a live isolated SurrealDB v3.2 returns KNN results
//! nearest-first. Self-contained (spawns its own surreal process) so it does
//! not depend on the binary's internal modules.

use std::io::Write;
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 61811;
const NS: &str = "smem";
const DB: &str = "spike";

fn run_sql(endpoint: &str, sql: &str) -> String {
    let full = format!("USE NS {NS} DB {DB};\n{sql}");
    let mut child = Command::new("surreal")
        .args([
            "sql", "-e", endpoint, "-u", "root", "-p", "root", "--json", "--multi",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn surreal sql");
    {
        let mut stdin = child.stdin.take().expect("stdin pipe");
        stdin.write_all(full.as_bytes()).expect("write sql");
    }
    let output = child.wait_with_output().expect("wait surreal sql");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn with_isolated_surreal<R>(routine: impl FnOnce(&str) -> R) -> R {
    let dir = std::env::temp_dir().join("smem-rust-spike-it");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let db_path = dir.join("it.skv");
    let bind = format!("127.0.0.1:{PORT}");
    let datastore = format!("surrealkv://{}", db_path.display());
    let mut server = Command::new("surreal")
        .args([
            "start",
            "--bind",
            &bind,
            "--user",
            "root",
            "--pass",
            "root",
            "--no-banner",
            &datastore,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn surreal start");
    let endpoint = format!("http://127.0.0.1:{PORT}");
    let deadline = Instant::now() + Duration::from_secs(15);
    while TcpStream::connect(&bind).is_err() {
        if Instant::now() > deadline {
            let _ = server.kill();
            panic!("isolated surreal did not become ready on {bind}");
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    let result = routine(&endpoint);
    let _ = server.kill();
    let _ = server.wait();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn collect_neuron_ids(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) if text.starts_with("neuron:") => out.push(text.clone()),
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

#[test]
fn green_a_knn_returns_nearest_first() {
    let seed = "\
DEFINE TABLE neuron SCHEMALESS;\
DEFINE INDEX neuron_embedding_idx ON neuron FIELDS embedding_vec HNSW DIMENSION 4 DIST COSINE TYPE F32;\
CREATE neuron:a SET embedding_vec = [1.0,0.0,0.0,0.0];\
CREATE neuron:b SET embedding_vec = [0.0,1.0,0.0,0.0];\
CREATE neuron:c SET embedding_vec = [0.7,0.7,0.0,0.0];";
    with_isolated_surreal(|endpoint| {
        run_sql(endpoint, seed);
        let output = run_sql(
            endpoint,
            "SELECT id, vector::distance::knn() AS s FROM neuron WHERE embedding_vec <|2,COSINE|> [0.9,0.1,0.0,0.0] ORDER BY s;",
        );
        let mut ordered_ids = Vec::new();
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    collect_neuron_ids(&value, &mut ordered_ids);
                    if ordered_ids.len() >= 2 {
                        break;
                    }
                }
            }
        }
        assert!(
            ordered_ids.len() >= 2,
            "expected >=2 knn ids, got {ordered_ids:?}\nraw: {output}"
        );
        assert_eq!(ordered_ids[0], "neuron:a");
        assert_eq!(ordered_ids[1], "neuron:c");
    });
}
