//! Isolated SurrealDB process driver.
//!
//! Spawns a dedicated `surreal` server (SurrealKV file backend) on a loopback
//! port and runs SurrealQL by shelling out to `surreal sql --json`. The server
//! is fully isolated: loopback bind, dedicated ephemeral file, dedicated
//! namespace/database. It never touches the production SurrealDB instance.

use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("surreal binary spawn failed: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("isolated surreal server did not become ready on {endpoint} within {timeout_ms}ms")]
    NotReady { endpoint: String, timeout_ms: u128 },
    #[error("surreal sql subprocess failed (exit={exit:?}): {stderr}")]
    SqlFailed { exit: Option<i32>, stderr: String },
    #[error("no JSON result line found in surreal sql stdout (raw: {0})")]
    NoJsonResult(String),
    #[error("JSON parse of surreal result failed: {0}")]
    Parse(#[from] serde_json::Error),
}

pub struct SurrealProcess {
    child: Child,
    endpoint: String,
}

impl SurrealProcess {
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub async fn spawn(port: u16, db_path: &Path) -> Result<Self, EngineError> {
        let endpoint = format!("http://127.0.0.1:{port}");
        let datastore = format!("surrealkv://{}", db_path.display());
        let bind = format!("127.0.0.1:{port}");
        let mut cmd = Command::new("surreal");
        cmd.args([
            "start",
            "--bind",
            &bind,
            "--user",
            "root",
            "--pass",
            "root",
            "--no-banner",
            &datastore,
        ]);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        cmd.kill_on_drop(true);
        let child = cmd.spawn()?;

        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if tokio::net::TcpStream::connect(&bind).await.is_ok() {
                break;
            }
            if Instant::now() > deadline {
                return Err(EngineError::NotReady {
                    endpoint: endpoint.clone(),
                    timeout_ms: 15000,
                });
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        Ok(Self { child, endpoint })
    }

    async fn run_sql_raw(&self, ns: &str, db: &str, sql: &str) -> Result<String, EngineError> {
        let full = format!("USE NS {ns} DB {db};\n{sql}");
        let mut cmd = Command::new("surreal");
        cmd.args([
            "sql",
            "-e",
            &self.endpoint,
            "-u",
            "root",
            "-p",
            "root",
            "--json",
            "--multi",
        ]);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let mut child = cmd.spawn()?;
        {
            let mut stdin = child.stdin.take().ok_or(EngineError::SqlFailed {
                exit: None,
                stderr: "stdin pipe not captured".into(),
            })?;
            stdin.write_all(full.as_bytes()).await?;
        }
        let output = child.wait_with_output().await?;
        if !output.status.success() {
            return Err(EngineError::SqlFailed {
                exit: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    pub async fn run_script(&self, ns: &str, db: &str, sql: &str) -> Result<(), EngineError> {
        self.run_sql_raw(ns, db, sql).await?;
        Ok(())
    }

    pub async fn run_query(
        &self,
        ns: &str,
        db: &str,
        sql: &str,
    ) -> Result<serde_json::Value, EngineError> {
        let raw = self.run_sql_raw(ns, db, sql).await?;
        let mut last: Option<serde_json::Value> = None;
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') || trimmed.starts_with('{') {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    last = Some(value);
                }
            }
        }
        last.ok_or(EngineError::NoJsonResult(raw))
    }
}

impl Drop for SurrealProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}
