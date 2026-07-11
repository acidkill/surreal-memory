# Acceptance stories — RUN-006 "Spectron trust ecosystem" (v2.9.0 + v2.10.0)
#
# SOURCE OF TRUTH for the Stage-2 testers (tonis-api-tester, tonis-browser-qa-tester,
# real-db-test-runner). These stories describe observable behaviour, not implementation.
# Each Feature maps to one backlog unit (U1–U10). TEST stories come from PLAN
# "Weryfikacja end-to-end" 1–7; DAILY-USE stories are the Spectron-demo journeys.
#
# Neutral defaults (binding, PLAN "Decyzje przekrojowe" #3):
#   trust_weight=0.0, recency_weight=1.0, trace.enabled=false,
#   include_superseded=false, include_uncertainty=false.
# The ONLY intended default-behaviour change: facts with valid_until set are hard-filtered
# out of recall (env/config escape hatch + 0.25x demotion as second line of defence).

Feature: U1 — Schema v9 fundament (zero behaviour change)
  As a maintainer upgrading an existing brain
  I want all v2.9.0 DDL to land in one additive schema bump
  So that recall/remember keep working and old code still reads the new DB

  Background:
    Given a SurrealDB v3.2.0 datastore and the SQLite backend
    And the shipped schema versions are SurrealDB 9 and SQLite 39

  Scenario: Migration 8->9 is idempotent
    Given a brain seeded at SurrealDB schema version 8
    When the 8->9 migration runs
    And the 8->9 migration runs a second time
    Then the schema version is 9
    And no error is raised
    And typed_memory rows are unchanged in count

  Scenario: SQLite migration 38->39 is idempotent
    Given a brain seeded at SQLite schema version 38
    When the 38->39 migration runs twice
    Then the schema version is 39
    And the sources table has a trust column
    And a retrieval_traces table exists

  Scenario: Fresh schema equals migrated schema
    Given a freshly bootstrapped brain via ensure_schema
    And a separate brain migrated from version 8 to 9
    Then both expose identical typed_memory, source and retrieval_trace definitions

  Scenario: Old code reads a v9 database without errors (additive rollback)
    Given a database at schema version 9
    When v2.8.x-style code performs a recall
    Then the recall returns results without raising
    Because the v9 DDL is purely additive (new option fields + new table)

  Scenario: TypedMemory round-trips validity fields on all three backends
    Given a TypedMemory with valid_from, valid_until and superseded_by set
    When it is stored and re-read on InMemory, SQLite and SurrealDB
    Then the validity fields survive the round-trip
    And a TypedMemory missing those keys reads back with None defaults

Feature: U2 — Trust/recency calibration (opt-in, golden-ranking stable)
  As an operator who has not enabled trust weighting
  I want the default ranking to stay bit-for-bit identical
  So that turning the knob is a conscious choice, never a silent drift

  Scenario: Golden ranking is unchanged on defaults
    Given ~20 fibers with recorded (fiber_id, score) on default config
    When recall runs on defaults after the trust/recency feature ships
    Then the ranking is bit-for-bit identical to the recorded snapshot

  Scenario: Zero storage reads when trust weighting is off
    Given trust_weight = 0.0
    When a recall executes
    Then no trust-map storage reads occur (spy asserts zero calls)

  Scenario: score_breakdown exposes trust and recency factors over MCP
    Given a source registered with an explicit trust value via smem_source
    And a brain configured with trust_weight > 0
    When smem_recall runs
    Then each result's score_breakdown includes trust_factor and recency_factor
    And the sources map in the response carries the trust value

Feature: U3 — Per-fact supersession (Emma Oslo -> Bergen)
  As a user whose facts change over time
  I want superseded facts hidden by default but reachable on demand
  So that "where does Emma live" answers Bergen, not both

  Background:
    Given smem_remember "Emma mieszka w Oslo"
    And smem_remember "Emma przeprowadziła się do Bergen" triggers auto-conflict supersession

  Scenario: Default recall returns only the current fact
    When smem_recall "gdzie mieszka Emma"
    Then the answer is Bergen
    And the Oslo fact is excluded
    And superseded_excluded_count is at least 1

  Scenario: Point-in-time recall returns the historical fact
    When smem_recall "gdzie mieszka Emma" with valid_at set to before the move
    Then the answer includes Oslo

  Scenario: include_superseded returns both facts flagged
    When smem_recall "gdzie mieszka Emma" with include_superseded = true
    Then both Oslo and Bergen are returned
    And each item carries valid_from, valid_until and superseded_by fields

  Scenario: Emergency escape hatch disables the hard filter
    Given the env/config flag that disables the superseded hard filter is OFF (filter disabled)
    When smem_recall "gdzie mieszka Emma" runs without include_superseded
    Then superseded facts are returned with the 0.25x demotion applied
    And with the flag ON (filter enabled) the same recall excludes them

  Scenario: Backfill supersession is idempotent
    Given legacy neurons marked _superseded=true
    When smem_lifecycle action="backfill_supersession" runs
    Then it reports backfilled and skipped_ambiguous counts
    And running it again changes nothing

  Scenario: Provenance lineage walks SUPERSEDES both directions
    When smem_provenance traces the lineage of the superseded fact
    Then the SUPERSEDES relation to the new fact is visible with cycle-guard

Feature: U4 — Queryable retrieval traces (opt-in telemetry)
  As an operator debugging why an answer was produced
  I want to query which recalls used a memory and what fed an answer
  So that provenance is inspectable without slowing recall

  Scenario: Per-call trace returns a trace_id without changing global config
    Given trace.enabled = false globally
    When smem_recall runs with trace = true
    Then the response includes a trace_id
    And the global trace config remains disabled

  Scenario: smem_provenance exposes traces and trace actions
    Given a persisted retrieval trace for memory X
    When smem_provenance action="traces" fiber_id=X
    Then the recalls that used memory X are listed
    When smem_provenance action="trace" trace_id set
    Then the full trace record is returned (fiber_ids, depth_used, confidence, latency_ms)

  Scenario: Trace-on vs trace-off recall latency delta is under 2 percent
    When the recall benchmark runs with tracing on and off
    Then the latency delta is below 2 percent

  Scenario: Trace pruning respects retention and max_traces
    Given more traces than max_traces or older than retention_days
    When scheduled consolidation runs
    Then oldest/expired traces are pruned

Feature: U5 — Uncertainty surfacing (smem_uncertainty)
  As a user who wants to know what the memory is unsure about
  I want an uncertainty overview aggregated from cheap signals
  So that contradictions, drift, expiring and low-evidence facts are visible

  Scenario: include_uncertainty attaches an uncertainty block to recall
    Given include_uncertainty = false by default
    When smem_recall runs with include_uncertainty = true
    Then the response carries an uncertainty block with a deterministic level low|medium|high

  Scenario: smem_uncertainty overview reports counts and contradiction_rate
    When smem_uncertainty action="overview"
    Then it returns counts, a contradiction_rate and top-10 items per category

  Scenario Outline: smem_uncertainty category actions return bounded lists
    When smem_uncertainty action="<action>"
    Then a list of at most 200 items is returned
    Examples:
      | action        |
      | contradictions|
      | drift         |
      | expiring      |
      | low_evidence  |

Feature: U6 — Health fields + dashboard UncertaintyPage
  As a dashboard user
  I want an Uncertainty page and a contradiction-rate tile
  So that brain health surfaces uncertainty at a glance

  Scenario: Uncertainty page is reachable from the sidebar
    Given the dashboard is served by make serve on port 8000
    When I open the dashboard and click the Uncertainty nav entry
    Then the Uncertainty page renders with a contradiction_rate tile
    And four cards: Contradictions, Drift, Expiring, Low-trust

  Scenario: dashboard uncertainty endpoint serves second hit from cache
    When GET /api/dashboard/uncertainty is called twice for the same brain
    Then the second hit is served from the TTL cache without new storage queries

  Scenario: Health grade formula is unchanged
    Given the existing purity penalty of 10 points
    When a brain health report is generated
    Then the grade matches the pre-change formula (no re-grading)

Feature: U7 — Release prep v2.9.0
  As a maintainer preparing a release
  I want version parity across all nine pinned files plus a CHANGELOG entry
  So that the release looks professional and CI version checks pass

  Scenario: Version parity across nine files
    Given the release version 2.9.0
    Then pyproject.toml, src/surreal_memory/__init__.py, the TestVersionBump pin,
      and the three package.json + three package-lock.json roots all read 2.9.0
    And CHANGELOG.md has a new "## [2.9.0]" section

  Scenario: Full quality gate is green
    When uv run make verify runs
    Then lint, format-check, mypy, coverage >= 67% and security all pass
    And the golden ranking test is green

  Scenario: Push, PR and tag are human-gated
    Then the exact push/PR/tag commands are prepared as NEEDS-HUMAN
    And none of them are executed by the agent

Feature: U8 — Geospatial recall (metadata-only, no schema change)
  As a user recalling memories near a place
  I want a near filter over explicit coordinates
  So that recall can be geo-scoped without a schema migration

  Scenario: Haversine Oslo to Bergen is about 305 km
    When haversine_m is computed between Oslo and Bergen coordinates
    Then the distance is ~305 km within 1 percent

  Scenario: near filter scopes recall on a live brain
    Given fibers with metadata.location coordinates
    When smem_recall runs with near {lat, lon, radius_m}
    Then only fibers within the radius are returned
    And early-return recall paths are bypassed when near is set

  Scenario: SurrealDB geo::distance uses lon,lat order and metres
    Given a live SurrealDB v3.2.0
    When find_fibers pushes down geo::distance with type::point([lon, lat])
    Then the metres unit and [lon, lat] order are confirmed by integration test

Feature: U9 — LangChain adapter (optional extra)
  As a LangChain user
  I want a retriever and chat-message-history backed by surreal-memory
  So that I can plug the memory into an LCEL RAG chain

  Scenario: Retriever round-trip maps fibers to Documents
    Given an InMemory storage wired via SurrealMemoryRetriever.from_storage
    When _aget_relevant_documents runs for a query
    Then Documents are returned with page_content and metadata (fiber_id, memory_type, tags)
    And k caps the number of Documents

  Scenario: Chat history isolates sessions
    Given two SurrealMemoryChatMessageHistory instances with different session ids
    When messages are added to each
    Then reading one session returns only its own Human/AI/System messages

  Scenario: Base env stays green without langchain installed
    When the unit suite runs without langchain-core installed
    Then the langchain adapter tests are skipped via importorskip
    And installing .[langchain,dev] makes test_langchain_adapter.py pass

Feature: U10 — Release prep v2.10.0
  As a maintainer preparing the ecosystem release
  I want parity, changelog and a regression re-run of the perf/geo battery
  So that the second release is as clean as the first

  Scenario: Version parity across nine files for 2.10.0
    Given the release version 2.10.0
    Then all nine pinned files and CHANGELOG.md read 2.10.0

  Scenario: Perf regression battery holds after geo touches the hot-path
    When the trace-on/off benchmark, dashboard uncertainty cache and /stats are re-run
    Then trace delta stays under 2 percent
    And the dashboard uncertainty second hit is cache-served
    And /stats renders under 3 seconds
