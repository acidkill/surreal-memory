# REST Endpoints

Every HTTP endpoint the server exposes, generated from its OpenAPI schema.

A running server serves the same schema interactively at `/docs` (Swagger) and
`/redoc`, including request and response bodies. This page is the flat index.

!!! note "Two paths reach most handlers"

    The memory, brain, sync, consolidation and hub routers are mounted twice:
    once under `/api/v1` and once unprefixed for backward compatibility.

## Brains

| Method | Path | Summary |
|---|---|---|
| `POST` | `/api/v1/brain/create` | Create a new brain |
| `DELETE` | `/api/v1/brain/{brain_id}` | Delete brain |
| `GET` | `/api/v1/brain/{brain_id}` | Get brain details |
| `GET` | `/api/v1/brain/{brain_id}/export` | Export brain |
| `POST` | `/api/v1/brain/{brain_id}/import` | Import brain |
| `POST` | `/api/v1/brain/{brain_id}/merge` | Merge snapshot into brain |
| `GET` | `/api/v1/brain/{brain_id}/stats` | Get brain statistics |
| `POST` | `/brain/create` | Create a new brain |
| `DELETE` | `/brain/{brain_id}` | Delete brain |
| `GET` | `/brain/{brain_id}` | Get brain details |
| `GET` | `/brain/{brain_id}/export` | Export brain |
| `POST` | `/brain/{brain_id}/import` | Import brain |
| `POST` | `/brain/{brain_id}/merge` | Merge snapshot into brain |
| `GET` | `/brain/{brain_id}/stats` | Get brain statistics |

## Consolidation

| Method | Path | Summary |
|---|---|---|
| `POST` | `/api/v1/brain/{brain_id}/consolidate` | Consolidate brain memories |
| `POST` | `/brain/{brain_id}/consolidate` | Consolidate brain memories |

## Dashboard

| Method | Path | Summary |
|---|---|---|
| `GET` | `/` | Root |
| `GET` | `/api/dashboard/activity` | Get integration activity metrics and log |
| `GET` | `/api/dashboard/brain-files` | Get brain file paths and sizes |
| `GET` | `/api/dashboard/brains` | List all brains |
| `POST` | `/api/dashboard/brains/switch` | Switch active brain |
| `PUT` | `/api/dashboard/config` | Update Config |
| `GET` | `/api/dashboard/config-status` | Get configuration status and actionable items |
| `GET` | `/api/dashboard/config/embedding` | Get Embedding Config |
| `POST` | `/api/dashboard/config/embedding/test` | Test Embedding Connection |
| `GET` | `/api/dashboard/evolution` | Get brain evolution metrics |
| `GET` | `/api/dashboard/fiber/{fiber_id}/diagram` | Get fiber structure for diagram |
| `GET` | `/api/dashboard/fibers` | List fibers for dropdown |
| `GET` | `/api/dashboard/health` | Get active brain health report |
| `GET` | `/api/dashboard/license` | Current license tier |
| `POST` | `/api/dashboard/license/activate` | Activate a license key |
| `GET` | `/api/dashboard/stats` | Get dashboard overview stats |
| `GET` | `/api/dashboard/storage/status` | Get SurrealDB storage status |
| `POST` | `/api/dashboard/sync-config` | Update sync configuration |
| `GET` | `/api/dashboard/sync-status` | Cloud sync status for dashboard |
| `POST` | `/api/dashboard/telegram/backup` | Send brain backup to Telegram |
| `GET` | `/api/dashboard/telegram/status` | Get Telegram integration status |
| `POST` | `/api/dashboard/telegram/test` | Send test message to Telegram |
| `GET` | `/api/dashboard/tier-stats` | Get memory tier distribution |
| `GET` | `/api/dashboard/timeline` | Get chronological memory timeline |
| `GET` | `/api/dashboard/timeline/daily-stats` | Get daily activity stats for timeline charts |
| `GET` | `/api/dashboard/tool-stats` | Tool Stats |
| `GET` | `/api/dashboard/uncertainty` | Brain-wide uncertainty overview (contradictions, low-trust, superseded, expiring, drift) |
| `POST` | `/api/dashboard/visualize` | Visualize Memory |
| `GET` | `/api/dashboard/watcher/status` | Get Watcher Status |
| `GET` | `/dashboard` | Dashboard |
| `GET` | `/dashboard/{path}` | Dashboard Spa Catchall |
| `GET` | `/ui` | Ui |
| `GET` | `/ui/{path}` | Ui Spa Catchall |

## Health

| Method | Path | Summary |
|---|---|---|
| `GET` | `/health` | Health Check |
| `GET` | `/ready` | Ready Check |

## Sync hub

| Method | Path | Summary |
|---|---|---|
| `GET` | `/api/v1/hub/devices/{brain_id}` | List registered devices for a brain |
| `POST` | `/api/v1/hub/register` | Register a device for a brain |
| `GET` | `/api/v1/hub/status/{brain_id}` | Get sync status for a brain |
| `POST` | `/api/v1/hub/sync` | Push/pull incremental changes |
| `POST` | `/api/v1/hub/sync/merkle` | Merkle delta sync (Pro) |
| `GET` | `/hub/devices/{brain_id}` | List registered devices for a brain |
| `POST` | `/hub/register` | Register a device for a brain |
| `GET` | `/hub/status/{brain_id}` | Get sync status for a brain |
| `POST` | `/hub/sync` | Push/pull incremental changes |
| `POST` | `/hub/sync/merkle` | Merkle delta sync (Pro) |

## Memory

| Method | Path | Summary |
|---|---|---|
| `POST` | `/api/v1/memory/encode` | Encode a new memory |
| `GET` | `/api/v1/memory/fiber/{fiber_id}` | Get a specific fiber |
| `POST` | `/api/v1/memory/index` | Index codebase |
| `GET` | `/api/v1/memory/neurons` | List neurons |
| `POST` | `/api/v1/memory/neurons` | Create a neuron |
| `DELETE` | `/api/v1/memory/neurons/{neuron_id}` | Delete a neuron |
| `GET` | `/api/v1/memory/neurons/{neuron_id}` | Get a neuron by ID |
| `PUT` | `/api/v1/memory/neurons/{neuron_id}` | Update a neuron |
| `GET` | `/api/v1/memory/neurons/{neuron_id}/neighbors` | Get neighboring neurons |
| `GET` | `/api/v1/memory/neurons/{neuron_id}/state` | Get neuron activation state |
| `PUT` | `/api/v1/memory/neurons/{neuron_id}/state` | Update neuron activation state |
| `GET` | `/api/v1/memory/neurons/{source_id}/path` | Find shortest path between neurons |
| `POST` | `/api/v1/memory/query` | Query memories |
| `GET` | `/api/v1/memory/show/{memory_id}` | Get full memory detail |
| `GET` | `/api/v1/memory/sources` | List sources |
| `GET` | `/api/v1/memory/sources/{source_id}` | Get source detail |
| `GET` | `/api/v1/memory/suggest` | Neuron suggestions |
| `GET` | `/api/v1/memory/synapses` | List synapses |
| `POST` | `/api/v1/memory/synapses` | Create a synapse |
| `DELETE` | `/api/v1/memory/synapses/{synapse_id}` | Delete a synapse |
| `GET` | `/api/v1/memory/synapses/{synapse_id}` | Get a synapse by ID |
| `PUT` | `/api/v1/memory/synapses/{synapse_id}` | Update a synapse |
| `POST` | `/memory/encode` | Encode a new memory |
| `GET` | `/memory/fiber/{fiber_id}` | Get a specific fiber |
| `POST` | `/memory/index` | Index codebase |
| `GET` | `/memory/neurons` | List neurons |
| `POST` | `/memory/neurons` | Create a neuron |
| `DELETE` | `/memory/neurons/{neuron_id}` | Delete a neuron |
| `GET` | `/memory/neurons/{neuron_id}` | Get a neuron by ID |
| `PUT` | `/memory/neurons/{neuron_id}` | Update a neuron |
| `GET` | `/memory/neurons/{neuron_id}/neighbors` | Get neighboring neurons |
| `GET` | `/memory/neurons/{neuron_id}/state` | Get neuron activation state |
| `PUT` | `/memory/neurons/{neuron_id}/state` | Update neuron activation state |
| `GET` | `/memory/neurons/{source_id}/path` | Find shortest path between neurons |
| `POST` | `/memory/query` | Query memories |
| `GET` | `/memory/show/{memory_id}` | Get full memory detail |
| `GET` | `/memory/sources` | List sources |
| `GET` | `/memory/sources/{source_id}` | Get source detail |
| `GET` | `/memory/suggest` | Neuron suggestions |
| `GET` | `/memory/synapses` | List synapses |
| `POST` | `/memory/synapses` | Create a synapse |
| `DELETE` | `/memory/synapses/{synapse_id}` | Delete a synapse |
| `GET` | `/memory/synapses/{synapse_id}` | Get a synapse by ID |
| `PUT` | `/memory/synapses/{synapse_id}` | Update a synapse |

## OAuth

| Method | Path | Summary |
|---|---|---|
| `POST` | `/api/oauth/callback` | OAuth callback handler |
| `POST` | `/api/oauth/initiate` | Start OAuth session |
| `GET` | `/api/oauth/providers` | List supported OAuth providers |
| `GET` | `/api/oauth/status/{provider}` | Check provider auth status |

## OpenClaw

| Method | Path | Summary |
|---|---|---|
| `POST` | `/api/openclaw/apikeys` | Add or update API key |
| `DELETE` | `/api/openclaw/apikeys/{provider}` | Remove API key |
| `GET` | `/api/openclaw/config` | Get OpenClaw configuration |
| `POST` | `/api/openclaw/config` | Save full OpenClaw configuration |
| `GET` | `/api/openclaw/discord` | Get Discord config |
| `POST` | `/api/openclaw/discord` | Update Discord config |
| `GET` | `/api/openclaw/functions` | List available functions |
| `POST` | `/api/openclaw/functions/{name}` | Toggle or configure function |
| `GET` | `/api/openclaw/security` | Get security config |
| `POST` | `/api/openclaw/security` | Update security config |
| `GET` | `/api/openclaw/telegram` | Get Telegram config |
| `POST` | `/api/openclaw/telegram` | Update Telegram config |

## Reasoning training

| Method | Path | Summary |
|---|---|---|
| `PUT` | `/api/dashboard/reasoning/config` | Update reasoning-training config |
| `POST` | `/api/dashboard/reasoning/mine` | Trigger a one-off mining run |
| `DELETE` | `/api/dashboard/reasoning/patterns` | Delete all patterns for a model |
| `GET` | `/api/dashboard/reasoning/patterns` | List learned patterns |
| `DELETE` | `/api/dashboard/reasoning/patterns/{pattern_id}` | Delete one learned pattern |
| `GET` | `/api/dashboard/reasoning/patterns/{pattern_id}` | Get one pattern's detail |
| `GET` | `/api/dashboard/reasoning/status` | Reasoning-training status |
| `DELETE` | `/api/dashboard/reasoning/traces` | Wipe staged traces for a model |

## Sync

| Method | Path | Summary |
|---|---|---|
| `GET` | `/api/v1/sync/stats` | Get Sync Stats |
| `GET` | `/sync/stats` | Get Sync Stats |

## Visualization

| Method | Path | Summary |
|---|---|---|
| `GET` | `/api/graph` | Get Graph Data |

---

*Auto-generated by `scripts/gen_api_docs.py` from the FastAPI OpenAPI schema — 132 operations across 11 groups.*
