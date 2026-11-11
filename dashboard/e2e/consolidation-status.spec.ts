import { test, expect, type Page } from "@playwright/test"

const HEALTH = {
  grade: "A", purity_score: 0.9, connectivity: 0.8, diversity: 0.7, freshness: 0.85,
  consolidation_ratio: 0.5, orphan_rate: 0.1, activation_efficiency: 0.6,
  recall_confidence: 0.75, neuron_count: 100, synapse_count: 200, fiber_count: 50,
  contradiction_count: 0, conflict_rate: 0, warnings: [], recommendations: [], top_penalties: [],
}
const STATUS = {
  brain: "default", state: "paused", checkpoint_available: true, observation_available: true,
  observation_stale: false, run_id: "synthetic-browser-run", checkpoint_status: "paused",
  current_strategy: "semantic_link", phase: "semantic_link_discovery_neurons",
  completed_strategies: 19, requested_strategies: 20, progress_known: true,
  work_progress_seq: 17, last_work_progress_at: "2026-11-11T12:00:00Z",
  last_observed_at: "2026-11-11T13:00:00Z", last_confirmed_success_at: null,
  eligible_work: true, reason: null,
}

async function installMocks(page: Page, status: object | null) {
  let healthCalls = 0
  const json = (body: unknown) => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) })
  await page.route("**/api/dashboard/**", route => route.fulfill(json({})))
  await page.route("**/api/dashboard/stats", route => route.fulfill(json({ active_brain: "default" })))
  await page.route("**/api/dashboard/brains", route => route.fulfill(json([])))
  await page.route("**/api/dashboard/fibers", route => route.fulfill(json({ fibers: [] })))
  await page.route(url => url.pathname === "/health", route => route.fulfill(json({ status: "ok", version: "fixture" })))
  await page.route("**/api/dashboard/health", route => {
    healthCalls += 1
    return route.fulfill(json(HEALTH))
  })
  await page.route("**/api/dashboard/consolidation-status", route => route.fulfill(
    status === null ? { status: 404, body: "Missing legacy endpoint" } : json(status),
  ))
  return () => healthCalls
}

test("saved work remains separate from health and does not add a diagnostics query", async ({ page }) => {
  const calls = await installMocks(page, STATUS)
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto("/ui/health")
  const panel = page.locator('[aria-labelledby="consolidation-work-title"]')
  await expect(panel.getByText("19 / 20")).toBeVisible({ timeout: 20_000 })
  await expect(panel.getByText("Paused", { exact: true })).toBeVisible()
  await expect(panel.getByText("2026-11-11 12:00:00 UTC")).toBeVisible()
  await expect(panel.getByText("Last confirmed success")).toBeVisible()
  expect(calls()).toBe(1)
  await panel.screenshot({ path: "output/playwright/consolidation-paused-desktop.png" })
})

test("a missing old-backend endpoint is visibly unknown without zero counts", async ({ page }) => {
  await installMocks(page, null)
  await page.goto("/ui/health")
  const panel = page.locator('[aria-labelledby="consolidation-work-title"]')
  await expect(panel.getByText("Saved consolidation status is unavailable.")).toBeVisible({ timeout: 20_000 })
  await expect(panel.getByText("0 / 0")).toHaveCount(0)
  await expect(panel.getByText("Completed", { exact: true })).toHaveCount(0)
  await panel.screenshot({ path: "output/playwright/consolidation-unknown.png" })
})

test("a blocked legacy checkpoint stays readable on a narrow screen", async ({ page }) => {
  await installMocks(page, { ...STATUS, state: "blocked", progress_known: false,
    work_progress_seq: null, last_work_progress_at: null, observation_stale: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto("/ui/health")
  const panel = page.locator('[aria-labelledby="consolidation-work-title"]')
  await expect(panel.getByText("Blocked", { exact: true })).toBeVisible({ timeout: 20_000 })
  await expect(panel.getByText("Useful-work history is unknown for this checkpoint.")).toBeVisible()
  await expect(panel.getByText("Scheduler observation is stale.")).toBeVisible()
  expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true)
  await page.getByRole("button", { name: /Collapse sidebar/i }).click()
  expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true)
  await panel.screenshot({ path: "output/playwright/consolidation-blocked-mobile.png" })
})
