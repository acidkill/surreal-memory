# smem Rust rewrite — Faza-0 spike result

Branch: `rust-rewrite-dev` (from `origin/main` @ `87bea15f`)
Date: 2026-08-03
Engine proven against: **SurrealDB v3.2.0** (SurrealKV backend), isolated instance
MCP SDK: **rmcp 1.5.0** (official Rust MCP SDK), compiled offline from local cargo cache

## Verdict: CONDITIONAL GO

The rewrite is **not blocked by the Surreal layer**. Vector KNN (HNSW), graph
traversal over relation edges, and an rmcp MCP server over stdio are all proven
end-to-end against real SurrealDB v3.2 + the real rmcp 1.5.0 crate. The only
unproven dimension is the **in-process embedding via the `surrealdb` Rust
crate** specifically, because that crate is unavailable in this air-gapped
sandbox and the network is blocked. A complete, context7-verified reference
implementation is provided under `reference-embedded/` for compilation in a
networked environment.

## Criteria

| Criterion | Status | Evidence |
|---|---|---|
| GREEN A — vector KNN nearest-first (HNSW, `<\|2,COSINE\|>`) | PASS | ordered `[neuron:a (s=0.0061), neuron:c (s=0.2191)]`, asserted |
| GREEN B — graph path a->b->c over `synapse` edges | PASS | 2-hop arrow traversal reaches `neuron:c` from `neuron:a` |
| GREEN C — rmcp MCP server, `smem_ping` over stdio | PASS | real JSON-RPC handshake, `tools/list=[smem_ping]`, `tools/call` returns `neuron_count=3` |
| `cargo test --offline` | PASS | 1 unit + 1 integration, both green |
| `cargo clippy --offline --all-targets -- -D warnings` | PASS | rc=0 |
| `cargo check --offline --all-targets` | PASS | rc=0 |
| `cargo fmt -- --check` | PASS | rc=0 |
| grep-ban (forbidden-token scan (per goal spec)) | PASS | ZERO matches under `rust/spike` |
| Rule 10 — isolated + default brain untouched | PASS | spike uses port 61801 only; prod is `localhost:8001`; `smem_stats` shows brain=`default`, no test brains; ephemeral DB removed |

## Evidence (stdout excerpts)

GREEN A (`cargo run --offline`, default mode):
```
[GREEN A] raw KNN result: [[{"id":"neuron:a","s":0.006116265326381098},{"id":"neuron:c","s":0.21913119055696972}]]
[GREEN A] ordered nearest-first ids: ["neuron:a", "neuron:c"]
[GREEN A] PASS — nearest-first order asserted: neuron:a then neuron:c
```

GREEN B:
```
[GREEN B] raw 2-hop traversal: [[{"two_hop":["neuron:c"]}]]
[GREEN B] ids reachable from neuron:a within 2 synapse hops: ["neuron:c"]
[GREEN B] PASS — graph path neuron:a -> neuron:b -> neuron:c confirmed
```

GREEN C (python3 stdio probe, real JSON-RPC over the rmcp stdio transport):
```
INIT_OK: True | {"protocolVersion":"2025-06-18","serverInfo":{"name":"rmcp","version":"1.5.0"}, ...}
TOOLS_LIST: ['smem_ping']
CALL_RESULT: {"content":[{"text":"{\"neuron_count\":3,...}"}], "structuredContent":{"neuron_count":3,...}, "isError":false}
NEURON_COUNT: 3
GREEN_C_PASS
```

## Proven SurrealQL (v3.2.0)

```sql
DEFINE TABLE neuron SCHEMALESS;
DEFINE INDEX neuron_embedding_idx ON neuron FIELDS embedding_vec
    HNSW DIMENSION 4 DIST COSINE TYPE F32;
DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron;
CREATE neuron:a SET embedding_vec = [1.0, 0.0, 0.0, 0.0];
CREATE neuron:b SET embedding_vec = [0.0, 1.0, 0.0, 0.0];
CREATE neuron:c SET embedding_vec = [0.7, 0.7, 0.0, 0.0];
RELATE neuron:a->synapse->neuron:b SET weight = 0.9;
RELATE neuron:b->synapse->neuron:c SET weight = 0.8;

-- GREEN A: vector KNN
SELECT id, vector::distance::knn() AS s FROM neuron
WHERE embedding_vec <|2,COSINE|> [0.9,0.1,0.0,0.0] ORDER BY s;

-- GREEN B: graph path (SurrealDB-native arrow traversal)
SELECT ->synapse->neuron->synapse->neuron.id AS two_hop FROM neuron:a;
```

## Deviation from the literal goal spec (honest)

1. **In-process `surrealdb` crate could not be compiled here.** Definitive
   proof: `cargo add surrealdb --offline` -> `error: the crate surrealdb could
   not be found in registry index`. The crate is absent from the local cargo
   cache and the sandbox blocks crates.io (CONNECT 403 / 000). Per the goal's
   block-handling clause ("try a different approach ... alt engine"), the
   runnable spike drives the locally-installed `surreal` v3.2.0 binary as an
   **isolated child process** (loopback bind `127.0.0.1:61801`, ephemeral
   SurrealKV file) via `surreal sql --json`. This still proves the SurrealQL
   semantics on real SurrealDB v3.2 and the rmcp Rust integration on the real
   rmcp 1.5.0 crate (compiled offline). The `reference-embedded/` package
   contains the intended in-process implementation for a networked environment.

2. **GREEN B query shape differs.** The goal's `MATCH p = SHORTEST 1 (...)`
   is openCypher syntax; SurrealQL v3.2 rejects it at parse time and there is
   no `graph::shortest_path()` function in v3.2. The native SurrealDB graph
   traversal is the record-arrow form `->synapse->neuron`, which is what the
   spike uses and asserts.

3. **Stretch D (fastembed) not attempted.** The `fastembed` crate is not
   cached and the network is blocked; this was an explicit stretch bonus, not
   a blocker. The spike uses static 4-dim vectors to exercise the HNSW path.

## Exhaustive search for the `surrealdb` crate (why in-process is impossible here)

Before concluding, the host was searched exhaustively for the crate + its closure:
- cargo registry cache + src: no `surreal*` crate (surrealdb, surrealkv, store,
  rebalanced) — the dependency closure is absent.
- cargo-git checkouts: only `cosmic-protocols`; no surrealdb.
- `~/.cargo/config{,.toml}` / `/etc/cargo`: no source replacement or mirror.
- Filesystem grep for `^name = "surrealdb"` in Cargo.toml across
  /home /opt /srv /usr/local/src: no Rust crate source (only Python `uv.lock`
  mentions of the PyPI `surrealdb` wheel).
- uv wheel/archive cache: no `.rs` / `Cargo.toml` under surrealdb paths (the
  PyPI wheel ships a compiled extension, not crate source).
- yay cache: only `surrealdb-bin` (precompiled CLI release, no source).
- loopback ports: no local crates mirror.
- Network: `cargo add surrealdb --offline` -> "could not be found in registry
  index"; crates.io and github blocked (proxy CONNECT 403 / curl 000).

Conclusion: the `surrealdb` crate and its transitive closure are unobtainable
in this sandbox, so the literal in-process embedding (IMPLEMENT step 1,
`Surreal::new::<SurrealKv>(...)`) cannot be compiled here by any approach.
The `reference-embedded/` artifact is the deliverable for a networked host.
This is an environmental hard-blocker, not a task that more code can resolve.

## Reproduce (offline-capable host)

```bash
cd rust/spike
cargo run --offline                       # GREEN A + B evidence on stdout
cargo test --offline                      # unit + integration KNN-order
cargo clippy --offline --all-targets -- -D warnings
cargo fmt -- --check
cargo run --offline -- --mcp              # then talk JSON-RPC over stdio
```

## Next step

Compile-verify `reference-embedded/` in an environment with crates.io access;
that closes the only remaining dimension (in-process embedding) and unblocks a
full GO for the Python-to-Rust rewrite of the storage + MCP layers.
