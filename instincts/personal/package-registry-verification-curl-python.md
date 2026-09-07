---
id: package-registry-verification-curl-python
trigger: when verifying package publication to npm, PyPI, or other registries after a release
confidence: 0.7
domain: workflow
source: session-observation
scope: project
project_id: 59ace3bb6ce0
project_name: surreal-memory
---

# Package Registry Verification via curl + python

## Action
Use `curl -s https://registry.<host>/json | python3 -c "import json,sys; d=json.load(sys.stdin); print(...)"` to verify package versions are available after publishing to registries (PyPI, npm, etc.).

## Evidence
- Observed 5 times in session f93d04a9-2c15-4557-a5ac-411ae06afc66
- Rows: 12, 18, 19, 20, 21 of analysis file
- Pattern: Post-publish verification against multiple registries (PyPI, npm surrealmemory, npm @acidkill/surreal-memory-client)
- Checks: latest version tag, availability of specific version, presence of artifacts
- Last observed: 2026-08-30T11:43:38Z

## Example Usage
```bash
# Verify PyPI package
curl -s https://pypi.org/pypi/surreal-memory/json | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('latest:', d['info']['version'])
print('3.8.1 files:', [f['filename'] for f in d['releases'].get('3.8.1',[])])
"

# Verify npm package
curl -s https://registry.npmjs.org/surrealmemory | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('latest:', d['dist-tags']['latest'])
print('has 3.8.1:', '3.8.1' in d['versions'])
"

# Verify scoped npm package with detailed query
curl -s https://registry.npmjs.org/@acidkill%2Fsurreal-memory-client | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('dist-tags:', d['dist-tags'])
print('time 3.8.1:', d.get('time',{}).get('3.8.1'))
"
```

## Why This Pattern Works
- `curl -s` fetches registry JSON without unnecessary output
- Python inline parsing allows flexible queries without installing jq
- Separates registry queries (PyPI vs npm) for independent verification
- Handles URL encoding for scoped packages (`%2F` for `/`)
- Can be combined with `sleep` between checks to wait for registry propagation
