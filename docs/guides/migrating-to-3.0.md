# Migrating off the SQLite backend

The SQLite storage backend is deprecated as of 2.21.0 and **removed in 3.0.0**.
SurrealDB has been the production backend since 2.0.0; SQLite remained the
default only because it needed no server.

This guide moves an existing SQLite brain to SurrealDB. It takes a few minutes
and never touches your `.db` files, so you can go back at any time.

## Am I affected?

```bash
smem doctor
```

The **Storage backend** check reports which backend resolves. If it says
`could not resolve backend: The SQLite storage backend was removed in 3.0.0...`,
this guide is for you. If it says `surrealdb`, you are already migrated and
3.0.0 changes nothing for you.

Anything that sets `SURREAL_MEMORY_STORAGE=surrealdb` — the Docker compose
files, `smem setup`, and the MCP configs written by `smem mcp-config` — is
already on the new backend.

## Migrate

### 1. Back up the current brain

```bash
smem export backup.json
```

Repeat per brain if you have several (`smem brain list` shows them):

```bash
smem export work.json --brain work
```

Keep these files until step 4 confirms the data arrived.

### 2. Start SurrealDB

```bash
docker compose -f docker-compose.surrealdb.yml up -d
```

### 3. Point surreal-memory at it

Set these in your environment (or `.env`):

```bash
SURREAL_MEMORY_STORAGE=surrealdb
SURREALDB_URL=http://localhost:8000
SURREALDB_PASS=<your password>
```

Equivalent in `~/.surrealmemory/config.toml`:

```toml
storage_backend = "surrealdb"
```

Restart anything long-running — the MCP server, `smem serve`, your editor
extension. A running process keeps the backend it started with.

### 4. Import and verify

```bash
smem import backup.json
smem stats
smem recall "something you know is in there"
```

Compare `smem stats` against what the SQLite brain reported before the switch.

## What a snapshot carries

| Carried | Not carried |
|---|---|
| Neurons, synapses, fibers | Document-training progress (the first `smem train` after migrating re-scans) |
| Typed memories — type, priority, tags, trust score, expiry, tier, validity window, supersession | Sync cursors and device registrations |
| Projects | Change-log history |
| Brain configuration | |
| Pinned status of trained memories | |

Everything in the right column is derived state: consolidation, recall and the
next training run rebuild it. Nothing you explicitly remembered is in it.

Training progress is the one entry worth a caveat. It is tracked per brain, so
the first `smem train` after you migrate re-encodes the corpus once and records
it; runs after that skip files whose contents have not changed. Before 2.21.0
the tracking existed only on the SQLite backend, so a SurrealDB brain re-scanned
on *every* run and duplicated the corpus each time — if you migrated earlier and
trained repeatedly, `smem consolidate` will merge the duplicates.

!!! note "Typed memories and projects need 2.21.0 or newer"

    Older versions did not put typed memories or projects into a SurrealDB
    snapshot, so a snapshot **exported** by 2.20.x or earlier from a SurrealDB
    brain carries only the graph. Export with 2.21.0+ on both ends.

## Trying it without a database

If you only want to evaluate surreal-memory, skip SurrealDB entirely:

```bash
SURREAL_MEMORY_STORAGE=memory smem remember "test note"
```

Nothing is written to disk and everything is discarded when the process exits.
This is for experiments, not for keeping memories.

## Going back

3.0.0 never reads, writes or deletes `~/.surrealmemory/brains/*.db`. If you need
the old backend, install the 2.x line again and it picks up right where it left
off:

```bash
pipx install 'surreal-memory==2.*' --force
```

Anything you wrote to SurrealDB in the meantime stays there — export it first if
you want to bring it back with you.

## Troubleshooting

**`smem` reports the backend was removed.** You are on 3.0.0 with
`storage_backend = "sqlite"` still configured — every command that touches
storage fails with the same message, `smem doctor` included. Set
`SURREAL_MEMORY_STORAGE=surrealdb` (or `memory`), or downgrade to 2.x to reach
your existing `.db` file.

**Connection refused.** SurrealDB is not running or is on a different port —
check `docker compose -f docker-compose.surrealdb.yml ps` and that
`SURREALDB_URL` matches.

**Authentication failed.** `SURREALDB_PASS` does not match the password the
container was started with. `smem doctor --fix` writes a consistent set.

**The dashboard and the CLI disagree.** One of them is still on the old backend.
This is the failure the deprecation warning calls out: check `smem doctor` in
the same environment as each process.

**`/health` reports `schema_version` dropped from 40 to 9.** That is not a
regression — `schema_version` is the *active backend's* schema version, and
2.x always reported the SQLite constant `40` even on a SurrealDB install.
SurrealDB's own schema is versioned separately and starts at a much lower
number. If you monitor this field, watch `version` (the product release, e.g.
`3.0.0`) for upgrades instead — `schema_version` moving is expected the moment
the active backend changes.
