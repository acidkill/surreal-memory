# Vibe Companion Setup — PC to Mobile via Cloudflare Tunnel

Full workflow to run Claude Code from any browser (PC, Mobile, iPad) using
[The Vibe Companion](https://github.com/The-Vibe-Company/companion) and Cloudflare Named Tunnel.

**Stable domain:** `https://cloude.theio.vn`

## Prerequisites

| Tool | Install |
|------|---------|
| Node.js 18+ | `winget install OpenJS.NodeJS.LTS` |
| Bun | `powershell -c "irm bun.sh/install.ps1 \| iex"` |
| Claude Code CLI | `npm install -g @anthropic-ai/claude-code` |
| Cloudflared | `winget install Cloudflare.cloudflared` |

Verify all tools:

```powershell
node --version        # v18+
bun --version         # 1.x
claude --version      # 2.x
cloudflared --version # 2025.x
```

## Step 1 — Start Companion

```powershell
npx the-vibe-companion
```

Expected output:

```
Server running on http://localhost:3456
  CLI WebSocket:     ws://localhost:3456/ws/cli/:sessionId
  Browser WebSocket: ws://localhost:3456/ws/browser/:sessionId
```

Open `http://localhost:3456` in your browser and create a session to verify it works.

## Step 2 — Fix Windows ENOENT Bug (Required)

> **Problem:** Companion uses `which` (Unix) to resolve the `claude` binary.
> On Windows, this fails with `ENOENT: no such file or directory, uv_spawn 'claude'`.
> The fix replaces `which` with `where.exe` and resolves `.cmd` wrappers.

Find the cached cli-launcher file:

```powershell
# Find the npx cache location
dir "$env:LOCALAPPDATA\npm-cache\_npx" -Recurse -Filter "cli-launcher.ts" |
  Where-Object { $_.FullName -like "*the-vibe-companion*" }
```

Open the file and find this block (~line 183):

```typescript
// BEFORE (broken on Windows)
let binary = options.claudeBinary || "claude";
if (!binary.startsWith("/")) {
  try {
    binary = execSync(`which ${binary}`, { encoding: "utf-8" }).trim();
  } catch {
    // fall through, hope it's in PATH
  }
}
```

Replace with:

```typescript
// AFTER (cross-platform fix)
let binary = options.claudeBinary || "claude";
if (!binary.startsWith("/") && !binary.match(/^[A-Za-z]:\\/)) {
  try {
    const whichCmd = process.platform === "win32" ? "where.exe" : "which";
    const resolved = execSync(`${whichCmd} ${binary}`, { encoding: "utf-8" }).trim();
    // where.exe may return multiple lines — take the first .cmd or .exe match
    if (process.platform === "win32") {
      const lines = resolved.split(/\r?\n/);
      binary = lines.find(l => l.endsWith(".cmd") || l.endsWith(".exe")) || lines[0];
    } else {
      binary = resolved;
    }
  } catch {
    // fall through, hope it's in PATH
  }
}
```

After patching, restart the Companion (kill the old process first).

## Step 3 — Expose via Cloudflare Named Tunnel

Named Tunnel gives you a **permanent HTTPS URL** on your own domain — no more random URLs that change on restart.

### One-time setup

```powershell
# 1. Login to Cloudflare (opens browser for auth)
cloudflared tunnel login

# 2. Create a named tunnel
cloudflared tunnel create companion

# 3. Route DNS — points cloude.theio.vn to the tunnel
cloudflared tunnel route dns companion cloude.theio.vn
```

### Config file

Create `~/.cloudflared/config.yml`:

```yaml
# ~/.cloudflared/config.yml
tunnel: companion
credentials-file: C:\Users\YOU\.cloudflared\TUNNEL_ID.json

ingress:
  - hostname: cloude.theio.vn
    service: http://localhost:3456
    originRequest:
      noTLSVerify: true
  - service: http_status:404
```

> Replace `TUNNEL_ID` with the UUID shown during `tunnel create`.

### Run the tunnel

```powershell
cloudflared tunnel run companion
```

Now open `https://cloude.theio.vn` on any device — phone, tablet, another PC.

## Step 4 — Run as Background Services (Optional)

### PowerShell (keep running in background)

```powershell
# Terminal 1: Companion
Start-Process -NoNewWindow npx -ArgumentList "the-vibe-companion"

# Terminal 2: Named Tunnel
Start-Process -NoNewWindow cloudflared -ArgumentList "tunnel","run","companion"
```

### Windows Task Scheduler (auto-start on boot)

Create a batch file `start-companion.bat`:

```batch
@echo off
start /B npx the-vibe-companion
timeout /t 5
start /B cloudflared tunnel run companion
```

Add to Task Scheduler with trigger "At log on".

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ENOENT: uv_spawn 'claude'` | Apply the Windows patch in Step 2 |
| `EADDRINUSE` port 3456 | `netstat -ano \| findstr :3456` then `taskkill /F /PID <pid>` |
| Tunnel URL shows Error 1033 | Cloudflared process died — restart it |
| `cloudflared: command not found` | Use full path: `& "C:\Program Files (x86)\cloudflared\cloudflared.exe"` |
| Session creates but no response | Check Claude CLI auth: run `claude` in terminal first |
| `failed to connect to origin` | Ensure Companion is running on port 3456 before starting tunnel |

## Architecture

```
Mobile/iPad  ──HTTPS──▶  Cloudflare Edge (cloude.theio.vn)
                              │
                              ▼ (QUIC tunnel)
                         cloudflared (Named Tunnel: companion)
                              │
                              ▼ HTTP
                     Vibe Companion :3456
                        │           │
                   WebSocket     WebSocket
                   (browser)      (CLI)
                        │           │
                        ▼           ▼
                    React UI    Claude Code CLI
                              (spawns on session create)
```

## Security Notes

- Companion grants **full CLI access** — treat `cloude.theio.vn` like a password
- For extra security: add [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/) (Zero Trust) for auth
- Never run with `--permission-mode bypassPermissions` on a public tunnel

## Alternative: Quick Tunnel (no domain needed)

If you don't have a domain, you can use Cloudflare Quick Tunnel for a temporary URL:

```powershell
cloudflared tunnel --url http://localhost:3456
```

Look for the tunnel URL in the output:

```
INF |  https://random-words-here.trycloudflare.com  |
```

**Caveats:** URL changes every restart, no authentication, not suitable for regular use.
