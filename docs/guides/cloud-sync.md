# Cloud Sync Setup

Sync your memories across devices with Neural Memory's cloud hub. Your memories stay encrypted in transit and only you can access them with your API key.

## Quick Start (3 steps)

### 1. Register

```bash
curl -X POST https://neural-memory-sync-hub.vietnam11399.workers.dev/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

You'll receive an API key starting with `nmk_`. **Save it immediately** — it's shown only once.

### 2. Connect

```python
nmem_sync_config(
    action="set",
    hub_url="https://neural-memory-sync-hub.vietnam11399.workers.dev",
    api_key="nmk_YOUR_KEY"
)
```

Sync is automatically enabled when both `hub_url` and `api_key` are set.

### 3. Sync

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

That's it. Your memories are now synced across devices.

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

- **API keys** are masked in all outputs (`nmk_a1b2c3d4****`) and never logged
- **HTTPS enforced** for cloud connections (HTTP only allowed for localhost)
- **Brain ownership** — the first device to sync a brain claims it; other devices must use the same account
- **API key stored** in `config.toml` — treat it like any other API key in your environment

## Managing API Keys

Create additional keys (e.g., one per device):

```bash
curl -X POST https://neural-memory-sync-hub.vietnam11399.workers.dev/v1/auth/keys \
  -H "Authorization: Bearer nmk_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "work-laptop"}'
```

List your keys:

```bash
curl https://neural-memory-sync-hub.vietnam11399.workers.dev/v1/auth/keys \
  -H "Authorization: Bearer nmk_YOUR_KEY"
```

Revoke a compromised key:

```bash
curl -X DELETE https://neural-memory-sync-hub.vietnam11399.workers.dev/v1/auth/keys/KEY_ID \
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

## Local Hub (Advanced)

For self-hosted sync without the cloud:

```bash
# Run a local hub
nmem serve  # localhost:8000

# Connect to local hub (no API key needed)
nmem_sync_config(action="set", hub_url="http://localhost:8000")
nmem_sync(action="full")
```

Local hubs don't require API keys or HTTPS.
