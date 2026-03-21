# Phase 1: Schema + Storage Layer

## Goal
Add `ephemeral` column to neurons table, write migration v33, and update storage layer to read/write the flag.

## Tasks
- [ ] 1.1 Add migration (32, 33) in `sqlite_schema.py` — `ALTER TABLE neurons ADD COLUMN ephemeral INTEGER DEFAULT 0` + index
- [ ] 1.2 Update `SCHEMA_VERSION` to 33
- [ ] 1.3 Update base schema DDL string to include `ephemeral INTEGER DEFAULT 0` in neurons CREATE TABLE
- [ ] 1.4 Add `ephemeral` field to `Neuron` frozen dataclass in `core/neuron.py` (default `False`)
- [ ] 1.5 Update `Neuron.create()` factory to accept `ephemeral` param
- [ ] 1.6 Update `add_neuron()` in `sqlite_neurons.py` — persist `ephemeral` column (1/0)
- [ ] 1.7 Update `_row_to_neuron()` in `sqlite_neurons.py` — read `ephemeral` column from row
- [ ] 1.8 Update `find_neurons()` in `sqlite_neurons.py` — add optional `ephemeral` filter param
- [ ] 1.9 Update `seed_change_log()` in `sqlite_change_log.py` — add `AND n.ephemeral = 0` to exclude ephemeral neurons from sync seeding
- [ ] 1.10 Update `record_change()` — accept `ephemeral` flag, skip recording if True

## Acceptance Criteria
- [ ] `SCHEMA_VERSION == 33` and migration runs without error on existing DBs
- [ ] `Neuron.create(ephemeral=True)` produces neuron with `ephemeral=True`
- [ ] `add_neuron()` persists ephemeral=1 in SQLite for ephemeral neurons
- [ ] `find_neurons(ephemeral=False)` excludes ephemeral neurons
- [ ] `seed_change_log()` never includes ephemeral neurons
- [ ] Fresh install creates neurons table with ephemeral column

## Files Touched
- `src/neural_memory/storage/sqlite_schema.py` — modify (migration + DDL)
- `src/neural_memory/core/neuron.py` — modify (add field)
- `src/neural_memory/storage/sqlite_neurons.py` — modify (CRUD)
- `src/neural_memory/storage/sqlite_change_log.py` — modify (sync exclusion)
- `src/neural_memory/storage/base.py` — modify (add ephemeral param to find_neurons signature)

## Dependencies
- None (first phase)

## Notes
- The `ephemeral` column is INTEGER (0/1) not BOOLEAN — SQLite has no native bool.
- Must update FTS triggers? No — ephemeral neurons should still be FTS-indexed for recall within session. The filtering happens at query time, not index time.
- `_row_to_neuron()` must handle old rows where column doesn't exist yet (migration adds DEFAULT 0).
