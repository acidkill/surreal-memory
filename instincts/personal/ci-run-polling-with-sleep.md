---
id: ci-run-polling-with-sleep
trigger: when waiting for GitHub Actions run to complete and checking status at intervals
confidence: 0.7
domain: workflow
source: session-observation
scope: project
project_id: 59ace3bb6ce0
project_name: surreal-memory
---

# CI Run Polling with gh run + sleep

## Action
Use `sleep N; gh run view <run-id> --json status,conclusion` pattern to poll CI run status at intervals, waiting for completion before proceeding with verification or next steps.

## Evidence
- Observed 3 times in session f93d04a9-2c15-4557-a5ac-411ae06afc66
- Rows: 2, 7, 10 of analysis file
- Pattern: Sleep interval (100-115 seconds), then query run status and job status in sequence
- Used for: Release workflow CI run 33309382209, waiting for publish jobs to complete
- Last observed: 2026-08-30T11:41:50Z

## Example Usage
```bash
# Poll after CI push, check both run status and individual job status
sleep 115
gh run view 33309382209 --json status,conclusion -q '"\\(.status) \\(.conclusion)"'
gh run view 33309382209 --json jobs -q '.jobs[] | "\\(.conclusion // .status)\\t\\(.name)"'

# Alternative: Combined wait + status check
sleep 100
gh run view 33309382209 --json status,conclusion -q '"RUN: \\(.status) \\(.conclusion)"'
gh run view 33309382209 --json jobs -q '.jobs[] | "\\(.conclusion // .status)\\t\\(.name)"'
```

## Why This Pattern Works
- Sleep duration (100-115s) aligns with typical GHA job execution times for this project
- Polling avoids constant API calls while ensuring reasonably fresh status
- Checking both run-level and job-level status provides comprehensive visibility
- `--json` output is parsed quickly without manual log inspection
- `.conclusion // .status` fallback handles in-progress jobs (no conclusion yet)
- Combining status check with job enumeration shows which specific jobs passed/failed
