# Cloud Sync Setup

Sync your memories across devices using your own Cloudflare Worker. Your data stays on **your** Cloudflare account — Neural Memory never touches it.

## Free vs Pro Sync

Cloud sync works on both tiers. Pro makes it faster and automatic.

| | Free | Pro |
|--|------|-----|
| **Sync method** | Manual push/pull | **Merkle delta** (only diffs travel) |
| **Auto-sync** | No | Yes — after every remember/recall |
| **Sync speed (10K neurons)** | ~5s (full snapshot) | **<1s** (incremental diff) |
| **Conflict resolution** | `prefer_recent` only | 4 strategies (recent, local, remote, stronger) |
| **Multi-device** | Yes | Yes |

All users get the same privacy model — your data, your infrastructure, your encryption keys. Pro just makes the sync smarter.

> **Upgrading?** See the [Pro Quickstart →](pro-quickstart.md)

## Privacy Model

| Aspect | Detail |
|--------|--------|
| **Where is data stored?** | Your own Cloudflare D1 database, on your own account |
| **Who can access it?** | Only you (via your API key) |
| **Encrypted in transit?** | Yes — HTTPS enforced for all cloud connections |
| **Encrypted at rest?** | Memories with `encrypted=true` use Fernet encryption (key stays local). Default memories are plaintext in D1 |
| **Is the hub open source?** | Yes — full source in `sync-hub/` directory |
| **Shared infrastructure?** | None — you deploy your own Worker |

For sensitive memories, enable encryption before syncing:

```python
nmem_remember("sensitive content", encrypted=True)
```

The Fernet key stays at `~/.neuralmemory/keys/{brain_id}.key` — never uploaded to the cloud.

---

## Quick Start (4 steps)

### 1. Deploy Your Sync Hub

You need a free [Cloudflare account](https://dash.cloudflare.com/sign-up).

```bash
cd sync-hub
npm install
npx wrangler login           # Login to your CF account
npx wrangler d1 create nmem  # Create your D1 database
npx wrangler deploy           # Deploy the Worker
```

After deploy, you'll see your Worker URL: `https://your-worker-name.your-subdomain.workers.dev`

Update `wrangler.toml` with the D1 database ID from the `d1 create` output.

### 2. Register

```bash
curl -X POST https://YOUR-WORKER.workers.dev/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

You'll receive an API key starting with `nmk_`. **Save it immediately** — it's shown only once.

### 3. Connect

```python
nmem_sync_config(
    action="set",
    hub_url="https://YOUR-WORKER.workers.dev",
    api_key="nmk_YOUR_KEY"
)
```

Sync is automatically enabled when both `hub_url` and `api_key` are set.

### 4. Sync

```python
# First time: prepare existing memories for sync
nmem_sync(action="seed")

# Push to cloud
nmem_sync(action="push")

# Pull from cloud (on another device)
nmem_sync(action="pull")

# Bidirectional sync
nmem_sync(action="full")
```

That's it. Your memories are now synced across devices — on your own infrastructure.

---

## Guided Setup (MCP)

If you're using Neural Memory as an MCP tool in Claude Code or Cursor, just run:

```python
nmem_sync_config(action="setup")
```

This returns step-by-step instructions tailored to your current configuration.

## Check Sync Status

```python
nmem_sync_status()
```

Shows:

- Sync enabled/disabled
- Connected hub URL
- API key (masked: `nmk_a1b2c3d4****`)
- Registered devices
- Pending changes
- Cloud tier and usage (when connected)

## Conflict Resolution

When the same memory is modified on two devices, Neural Memory resolves conflicts using a configurable strategy:

| Strategy | Behavior |
|----------|----------|
| `prefer_recent` (default) | Most recent change wins |
| `prefer_local` | Local device always wins |
| `prefer_remote` | Remote/cloud always wins |
| `prefer_stronger` | Higher activation score wins |

Change the strategy:

```python
nmem_sync_config(action="set", conflict_strategy="prefer_local")
```

## Multiple Devices

Each device is automatically registered on first sync. View all devices:

```python
nmem_sync_status()  # Shows device list with last sync time
```

## Security

- **Self-hosted** — your data stays on your Cloudflare account
- **API keys** are SHA-256 hashed in D1 (raw key never stored server-side)
- **API keys** are masked in all client outputs (`nmk_a1b2c3d4****`) and never logged
- **HTTPS enforced** for cloud connections (HTTP only allowed for localhost)
- **Brain ownership** — the first device to sync a brain claims it; other devices must use the same account
- **API key stored** locally in `config.toml` — treat it like any other API key

## Managing API Keys

Create additional keys (e.g., one per device):

```bash
curl -X POST https://YOUR-WORKER.workers.dev/v1/auth/keys \
  -H "Authorization: Bearer nmk_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "work-laptop"}'
```

List your keys:

```bash
curl https://YOUR-WORKER.workers.dev/v1/auth/keys \
  -H "Authorization: Bearer nmk_YOUR_KEY"
```

Revoke a compromised key:

```bash
curl -X DELETE https://YOUR-WORKER.workers.dev/v1/auth/keys/KEY_ID \
  -H "Authorization: Bearer nmk_YOUR_KEY"
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "Invalid or expired API key" | Wrong or revoked key | Re-register or create a new key |
| "Access denied" | Brain owned by another account | Use the same account that first synced this brain |
| "Payload too large" | Too many changes at once | Sync more frequently |
| "Rate limited" | Too many requests | Wait a few seconds and retry |
| "Cloud hub requires HTTPS" | Using `http://` for cloud | Change to `https://` in hub_url |

## Local Hub (LAN Sync)

For syncing between machines on the same network without any cloud:

```bash
# Run a local hub
nmem serve  # localhost:8000

# Connect to local hub (no API key needed)
nmem_sync_config(action="set", hub_url="http://localhost:8000")
nmem_sync(action="full")
```

Local hubs don't require API keys or HTTPS.

## Cloudflare Free Tier Limits

The sync hub runs well within Cloudflare's free tier:

| Resource | Free Limit | Typical Usage |
|----------|-----------|---------------|
| Worker requests | 100K/day | ~100-500/day for personal use |
| D1 storage | 5 GB | Brain with 10K neurons ≈ 50 MB |
| D1 reads | 5M/day | Well within limits |
| D1 writes | 100K/day | Well within limits |
