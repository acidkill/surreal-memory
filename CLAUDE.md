# NeuralMemory — AI Coding Standards

Project-level rules that Claude Code reads automatically.

## Architecture

```
src/neural_memory/
  core/       — Frozen dataclasses (Neuron, Synapse, Fiber, Brain)
  engine/     — Encoding, retrieval, consolidation, diagnostics
  storage/    — SQLite persistence (async via aiosqlite)
  mcp/        — MCP server (stdio transport for Claude Code)
  server/     — FastAPI REST API + dashboard
  integration/— External source adapters (Mem0, ChromaDB, Graphiti, …)
  safety/     — Sensitive content detection, freshness evaluation
  utils/      — Config, time utilities, simhash
```

## Immutability Rules

- **Never mutate function parameters.** Create new objects with `replace()` or spread (`{**d, key: val}`).
- **Core models are frozen dataclasses.** Use `replace()` to derive new instances.
- **No mutable default arguments** on frozen dataclasses (use `field(default_factory=...)`).

## Datetime Rules

- Use `utcnow()` from `neural_memory.utils.timeutils` — never `datetime.now()`.
- Store **naive UTC** datetimes for SQLite (no tzinfo).
- Never mix naive and timezone-aware datetimes.

## Security Rules

- **Parameterized SQL only.** Never f-string or `.format()` into SQL queries.
- **Validate all paths** with `Path.resolve()` + `is_relative_to()` before file access.
- **Never expose internal errors** to clients. Use generic messages in HTTP/MCP responses.
- **Never include available brain names, stack traces, or exception types** in error responses.
- **Bind to `127.0.0.1` by default**, not `0.0.0.0`.
- **CORS defaults** to localhost origins, not `["*"]`.

## Bounds Rules

- **Always cap server-side limits.** Use `min(user_limit, MAX)` for any user-provided limit.
- MCP context limit: max 200. Habits fiber fetch: max 1000.
- REST neuron list: max 1000. Encode content: max 100,000 chars.

## Testing Rules

- Minimum coverage: **67%** (enforced by CI via `fail_under = 67` in pyproject.toml).
- Test immutability: verify that functions don't mutate their inputs.
- Use `pytest-asyncio` with `asyncio_mode = "auto"`.

## Error Handling

- Never bare `except: pass`. Always log or re-raise.
- In MCP handlers: always `logger.error(...)` before returning error dicts.
- Migration errors: halt on non-benign errors (don't advance schema version).

## Type Safety Rules

- **Always use generic type params**: `dict[str, Any]` not bare `dict`, `list[str]` not bare `list`.
- **Mixin classes must declare protocol stubs** under `if TYPE_CHECKING` for all attributes/methods used from the composing class. Use `raise NotImplementedError` for stubs with non-`None` return types.
- **Narrow Optional types before use**: `assert x is not None` or `x = x or "default"` before passing `str | None` to a parameter typed `str`.
- **No stale `# type: ignore`**: remove when the underlying issue is fixed. Always use specific error codes (`# type: ignore[attr-defined]`), never bare `# type: ignore`.
- **CI must pass `mypy src/ --ignore-missing-imports` with 0 errors.** Never merge code that adds new mypy errors.
- **Avoid variable name reuse** across different types in the same scope — rename to avoid type conflicts (e.g. `storage` / `sqlite_storage`).

## Naming Conventions

- `type` parameter is accepted in **public API** (FastAPI query params, MCP tool args).
- Use `neuron_type` in **new internal code** to avoid shadowing the builtin.
- `snake_case` for functions/variables, `PascalCase` for classes, `SCREAMING_SNAKE` for constants.

## Migration Rules

When changing config formats, storage paths, or schema:

- **Test the upgrade path**: existing data (old format) → new code must work seamlessly.
- **Test fresh install**: no prior data → new code creates correct defaults.
- **Test mixed state**: partial migration (e.g. `config.json` exists but `config.toml` doesn't) must resolve correctly.
- **Never silently discard user state.** If a legacy file contains `current_brain = "work"`, the migration must carry it forward — not reset to `"default"`.
- **Write migration tests before merging.** Every `load()` / `migrate()` function needs tests for: old→new, fresh, already-migrated, corrupt input, and invalid values.
- **Log migrations.** Use `logger.info()` when migrating data so users can diagnose issues.

## Pre-release Smoke Test

Before tagging a release, verify these scenarios manually or via integration tests:

1. **Fresh install**: delete data dir, run MCP server, confirm default brain created.
2. **Upgrade from previous version**: keep old data dir intact, run new code, confirm brain name and memories preserved.
3. **Brain switch round-trip**: switch brain via CLI → confirm MCP reads the new brain → switch back → confirm again.
4. **Config file conflicts**: both `config.json` and `config.toml` exist → confirm `config.toml` wins.
5. **Recall after upgrade**: store a memory, upgrade, recall it — confirm it's still there with correct brain context.

## API Pitfalls

| API | Gotcha |
|-----|--------|
| `Neuron.create()` | Uses `type=` not `neuron_type=` |
| `Synapse.create()` | Uses `type=` not `synapse_type=` |
| `find_neurons()` | Uses `content_exact=` param. NO `brain_id` param |
| `get_synapses()` | NOT `find_synapses()`. Uses `source_id=`, `target_id=`, `type=` |
| `add_neuron()`, `add_synapse()` | NO `brain_id` param — uses `set_brain` context |
| `tool_events` FK | Test fixture must insert brain with ALL required columns |

## Version Bump Checklist

ALL files must be updated when bumping version:

1. `pyproject.toml` → `version = "x.y.z"`
2. `src/neural_memory/__init__.py` → `__version__ = "x.y.z"`
3. `.claude-plugin/plugin.json` → `"version": "x.y.z"`
4. `.claude-plugin/marketplace.json` → `"version": "x.y.z"` (TWO occurrences)
5. `tests/unit/test_health_fixes.py` → `assert neural_memory.__version__ == "x.y.z"`
6. `tests/unit/test_markdown_export.py` → `"version": "x.y.z"` in fixture

## Pre-Ship Verification (MANDATORY)

Before EVERY release, run:

```bash
python scripts/pre_ship.py        # Verify all checks pass
python scripts/pre_ship.py --fix  # Auto-fix ruff issues first
```

This checks: version consistency (6 files), ruff lint+format, mypy, import smoke test, fast unit tests, CHANGELOG entry, OpenClaw plugin consistency.

**Do NOT tag or release if pre_ship.py fails.**

## Commit Messages

Format: `<type>: <description>` — types: feat, fix, refactor, docs, test, chore, perf, ci

## Neurodungeon (Dashboard Game Feature)

Roguelike dungeon crawler inside the NM dashboard. Map generated from real brain data.

### Architecture

```
dashboard/src/features/neurodungeon/
  engine/
    types.ts          — All game types (as const objects, NOT enums — TS 5.9 erasableSyntaxOnly)
    dungeon-gen.ts    — Graph → dungeon map (rooms from neurons, corridors from synapses)
    game-loop.ts      — Pure turn-based state machine: (state, action) → state
    combat.ts         — Type advantages, damage formulas, crit/flee
    items.ts          — Item generation per room type, useItem()
    events.ts         — World events from health data (6 types)
    achievements.ts   — 13 achievements, localStorage persistence
  renderer/
    GameCanvas.tsx    — Canvas 2D with fog of war, camera follow, effects
    tiles.ts          — Dark theme color palette
    minimap.tsx       — 160px minimap overlay
    hud.tsx           — HP bar, floor info, score, turn log
    effects.ts        — Particles, screen shake, flash (stateless)
  ui/
    StartScreen.tsx   — Brain stats, difficulty, high score, Enter to start
    GameOverScreen.tsx— Score summary, achievements, share card buttons
    DialogueModal.tsx — Per-room-type styled dialogue
    CombatOverlay.tsx — Turn-based combat UI (A/D/F keys)
    QuizModal.tsx     — Fill-the-blank memory recall challenge
    AchievementToast.tsx — Gold toast popup on unlock
  hooks/
    useGameState.ts   — Zustand store (startGame, dispatch, updateState)
    useKeyboard.ts    — WASD/Arrow movement, E interact, 1-5 items
    useDungeonData.ts — API graph → DungeonGenInput transformer
  utils/
    pathfinding.ts    — A* for corridor generation
    noise.ts          — Mulberry32 seeded PRNG
    share-card.ts     — Canvas PNG generation (600x340)
  NeurodungeonPage.tsx — Main page composing all components
```

### Key Conventions

- **TS 5.9 compat**: Use `as const` objects + derived types, NOT TypeScript enums
- **Immutability**: `processTurn(state, action) → newState` — never mutate DungeonState
- **0 new dependencies**: Pure Canvas 2D + React + Zustand (already in project)
- **Data mapping**: neuron type → room type, synapse weight → corridor width, fiber → floor
- **Combat stats from data**: `HP = content.length / 10`, `ATK = activation * 10`, `DEF = synapse_count`
- **World events from health**: orphans → zombies, low connectivity → collapsed corridors
- **Persistence**: High scores + achievements in localStorage
- **API reuse**: Uses existing `/api/graph`, `/api/dashboard/health`, `/api/dashboard/fibers`

### Item System Design

Items create meaningful decisions — keep vs use vs recycle:

| Item | Source Room | Effect | Strategy |
|------|-----------|--------|----------|
| Memory Essence | Treasure | Heal HP | Save for low HP moments |
| Memory Barrier | Treasure (high activation) | Block 1 hit completely | Save for boss fights |
| Scroll of Resolve | Fork (priority >= 6) | ATK +N for 10 turns | Time before combat |
| Scroll of Caution | Fork (priority < 6) | DEF +N for 10 turns | Time before combat |
| Scholar's Map | Library (25% chance) | Reveal all rooms on floor | Use early on new floor |
| Tome of Restoration | Library (25% chance) | Large HP heal | Save for emergencies |
| Treatise on War | Library (25% chance) | ATK buff 10 turns | Same as ATK scroll |
| Manual of Defense | Library (25% chance) | DEF buff 10 turns | Same as DEF scroll |
| Forgotten Ward | Secret room | Block 1 hit | Rare, save for bosses |

- **Buff scrolls are temporary** (10 turns) — creates timing decisions
- **Shield** blocks exactly 1 hit then shatters — clutch saves
- **Drop = Recycle** gives +15 score — inventory management matters
- **Max 5 items** — forces prioritization
- **Library variety**: deterministic hash-based, not random — same room always gives same item type
- **Keybinds**: `1-5` use, `Shift+1-5` drop/recycle

### Engagement Systems (engine/engagement.ts)

| System | Mechanic | Player Impact |
|--------|----------|---------------|
| Memory Chain | Visit synapse-connected rooms in sequence | Score multiplier x1.5, x2, x2.5... |
| Danger Level | +1 every 15 turns on floor, max 10 | Enemies +10% stats/level, forces pacing |
| Floor Rating | S/A/B/C/D from explore%, kills%, chain, speed | Score multiplied: S=3x, A=2x, B=1.5x |
| Kill Streak | Consecutive kills without taking damage | +25 score per streak kill |

- Chain uses REAL corridor/synapse data → each brain has unique optimal paths
- Danger creates tension: explore everything (loot) vs speedrun (safety)
- Rating drives replayability: "I got B, let me try for S"
- FLOOR_COMPLETE phase pauses between floors to show rating overlay

### Plan & Status

Full plan with task breakdown: `.rune/neurodungeon-plan.md`
Phases 1-3 complete, Phase 4 partial, Item Redesign + Engagement Systems done.
