import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import type { ConsolidationStatus } from "@/api/types"
import ConsolidationWorkPanel from "./ConsolidationWorkPanel"

const UNKNOWN: ConsolidationStatus = {
  brain: "default", state: "unknown", checkpoint_available: false,
  observation_available: false, observation_stale: null, run_id: null,
  checkpoint_status: null, current_strategy: null, phase: null,
  completed_strategies: null, requested_strategies: null, progress_known: false,
  work_progress_seq: null, last_work_progress_at: null, last_observed_at: null,
  last_confirmed_success_at: null, eligible_work: null, reason: "scheduler_unavailable",
}

describe("saved consolidation work", () => {
  it("keeps old backend and unavailable status unknown without invented zeros", () => {
    for (const status of [undefined, UNKNOWN]) {
      const html = renderToStaticMarkup(<ConsolidationWorkPanel status={status} />)
      expect(html).toContain("Unknown")
      expect(html).not.toContain("0 / 0")
      expect(html).not.toContain(">Completed<")
    }
  })

  it("shows saved work, strategy count, scheduler freshness and confirmed success separately", () => {
    const html = renderToStaticMarkup(<ConsolidationWorkPanel status={{
      ...UNKNOWN, state: "paused", checkpoint_available: true, observation_available: true,
      observation_stale: false, run_id: "synthetic-run", current_strategy: "semantic_link",
      completed_strategies: 19, requested_strategies: 20, progress_known: true,
      work_progress_seq: 7, last_work_progress_at: "2026-11-11T12:00:00Z",
      last_observed_at: "2026-11-11T13:00:00Z",
    }} />)
    expect(html).toContain("Paused")
    expect(html).toContain("19 / 20")
    expect(html).toContain("synthetic-run")
    expect(html).toContain("semantic_link")
    expect(html).toContain("2026-11-11 12:00:00 UTC")
    expect(html).toContain("2026-11-11 13:00:00 UTC")
    expect(html).toContain("Last confirmed success")
  })

  it("does not show cached completed data as green when a refresh is unavailable", () => {
    const html = renderToStaticMarkup(<ConsolidationWorkPanel unavailable status={{
      ...UNKNOWN, state: "completed", run_id: "old-success",
    }} />)
    expect(html).toContain("status is unavailable")
    expect(html).not.toContain("old-success")
    expect(html).not.toContain(">Completed<")
  })

  it("renders stale and malformed timestamp evidence honestly", () => {
    const html = renderToStaticMarkup(<ConsolidationWorkPanel status={{
      ...UNKNOWN, observation_stale: true, last_work_progress_at: "bad-date",
    }} />)
    expect(html).toContain("observation is stale")
    expect(html).not.toContain("Invalid Date")
  })
})
