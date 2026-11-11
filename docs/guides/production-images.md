# Production images

The `CI` workflow builds one Linux amd64 image for the API and CLI. The hourly
consolidation Job can use the same image with its existing ConfigMap-mounted
supervisor; the supervisor, RBAC, receipts and runtime configuration are managed
separately. The image does not contain an unreviewed local runner script.

Image builds wait for version consistency, lint, type checks, both Python test
jobs, docs, security, both real-SurrealDB integration jobs and package/dashboard
checks. The Dockerfile rebuilds the dashboard using `npm ci` and the committed
lockfile. It installs the existing `server`, `surrealdb` and `embeddings-openai`
extras, including the vectorized consolidation dependency.

`Production Image Smoke` runs the image as UID 1000 with no network and a
read-only root filesystem. It checks runtime imports, `smem consolidate --help`,
and HTTP responses for the dashboard and its entry assets without starting the
database lifespan. This is image validation, not a live database or production
acceptance test.

Pull requests build and smoke test only, with read-only repository permissions.
A successful push to `main`, or a manual `CI` dispatch on `main`, publishes the
same tested archive to `ghcr.io/acidkill/surreal-memory`. Publication is never
allowed from a pull request or another branch. The publishing job uses the
repository's `GITHUB_TOKEN` with `packages: write`; no external registry secret
is required. Package access must separately allow the deployment operator to
pull it; the workflow does not change package visibility or cluster credentials.

Each tag includes the full source SHA, run ID and attempt. The OCI revision label
records that source SHA. `production-image-receipt` records the immutable digest,
image ID, tag and workflow URL. Use `image_digest` from this receipt for deployment;
a source SHA alone does not identify resolved Python dependencies or base layers.
Dashboard dependencies are locked; the Python and base-image inputs retain the
repository's existing version ranges, so this is not a byte-reproducible build.

## Deployment and rollback

Publication does not deploy. Before a rollout, preserve the installed Helm values,
rendered manifests, current app/Job image references, runner ConfigMap bytes and
consolidation receipt/checkpoint identity. Respect active worker leases and the
existing database capacity and backup gates; do not reset an unfinished run.

Set both API and hourly Job image references to the receipt's exact digest. For a
cluster using node-local images and `imagePullPolicy: Never`, first pull/export
that digest and import the same image into every eligible node's containerd; do
not rebuild it locally. For registry pulls, configure existing pull access and a
compatible pull policy explicitly. Render and review the chart before applying.

Verify the running image IDs and installed OCI revision, API readiness and UI,
runtime provider configuration, and the mounted runner hash. Observe a supervised
hourly tick and its durable receipt before declaring consolidation accepted.
Rollback restores the recorded app/Job image references and configuration; it
does not rewind database contents, checkpoint progress or receipts. Never restore
an older runner against newer recovery metadata without verifying compatibility.

To rebuild after this workflow is merged, run `gh workflow run ci.yml --ref main`.
Inspect existing runs for the target SHA before dispatching another build. Wait
for every required job and the publication receipt; an uploaded package or a
green PR test run does not establish that an image was published or deployed.
