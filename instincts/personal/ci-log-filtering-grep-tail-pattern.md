---
id: ci-log-filtering-grep-tail-pattern
trigger: when diagnosing GitHub Actions job output or filtering CI logs for specific jobs
confidence: 0.7
domain: debugging
source: session-observation
scope: project
project_id: 59ace3bb6ce0
project_name: surreal-memory
---

# CI Log Filtering with grep + tail

## Action
Use `gh run view --log | grep -F <job> | grep -viE <exclude> | tail -N` to extract and filter GitHub Actions job logs for specific jobs, removing noise before inspection.

## Evidence
- Observed 4 times in session f93d04a9-2c15-4557-a5ac-411ae06afc66
- Rows: 14, 16, 24, 25 of analysis file
- Pattern: Extract logs from specific publish/CI jobs, filter deprecation warnings and git config noise, show final lines for diagnosis
- Used for: Publish to PyPI, Publish TypeScript SDK, Publish VS Code Extension, Publish to ClawHub jobs
- Last observed: 2026-08-30T11:44:02Z

## Example Usage
```bash
# Filter a single job's logs (e.g. "Publish to PyPI"), remove verbose output, show last 30 lines
gh run view 33309382209 --log 2>/dev/null | grep -F "Publish to PyPI" | grep -viE "node|deprecat|git config|checkout|cleanup" | tail -30

# Filter multiple jobs in sequence
for j in "Publish to PyPI" "Publish TypeScript SDK"; do
  echo "############ $j ############"
  gh run view 33309382209 --log 2>/dev/null | grep -F "$j" | tail -30
done
```

## Why This Pattern Works
- `grep -F` matches fixed strings (job names) exactly, avoiding regex interpretation
- `grep -viE` excludes verbose noise (Node.js warnings, git config output, cleanup logs)
- `tail -N` shows only final N lines, focusing on actual work and errors
- Separating filter stages keeps pipeline readable and easy to adjust exclusions
