import { test, expect, type Page } from "@playwright/test"

const HEALTH = {
  grade: "D", purity_score: 45.7, connectivity: .5, diversity: .5, freshness: .5,
  consolidation_ratio: .1, orphan_rate: .1, activation_efficiency: .5, recall_confidence: .1,
  neuron_count: 17, synapse_count: 29, fiber_count: 7, contradiction_count: 4,
  conflict_rate: .02, warnings: [], recommendations: [], top_penalties: [],
}
const BRAIN = { id: "default", name: "default", neuron_count: 17, synapse_count: 29,
  fiber_count: 7, grade: "—", purity_score: null, diagnostic_status: "pending", is_active: true }
const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) })

async function mocks(page: Page) {
  // No request can reach a real API, including Vite's local backend proxy.
  await page.route(url => url.pathname.startsWith("/api/"), route => route.fulfill(json({})))
  await page.route(url => url.pathname === "/health", route => route.fulfill(json({ status: "ok", version: "fixture" })))
  await page.route("**/api/dashboard/stats", route => route.fulfill(json({
    active_brain: "default", total_brains: 1, total_neurons: 17, total_synapses: 29,
    total_fibers: 7, health_grade: "—", purity_score: null, diagnostic_status: "pending", brains: [BRAIN],
  })))
  await page.route("**/api/dashboard/brains", route => route.fulfill(json([BRAIN])))
  await page.route("**/api/dashboard/consolidation-status", route => route.fulfill(json({}, 503)))
}

test("delayed cold health stays neutral, then short-polls the actual report", async ({ page }, testInfo) => {
  await mocks(page)
  let ready = false
  let calls = 0
  await page.route("**/api/dashboard/health", async route => {
    calls += 1
    await route.fulfill(ready ? json(HEALTH) : json({ status: "pending" }, 202))
  })
  await page.goto("/ui/health")
  await expect(page.getByRole("status")).toContainText("Diagnostics are being calculated")
  await expect(page.getByText("0.0%", { exact: true })).toHaveCount(0)
  await expect(page.getByText("F", { exact: true })).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath("health-pending.png") })
  ready = true
  await expect(page.getByText("D", { exact: true })).toBeVisible({ timeout: 6000 })
  await expect(page.getByText("2.0%", { exact: true })).toBeVisible()
  expect(calls).toBeGreaterThanOrEqual(2)
})

test("health errors expose a working retry action", async ({ page }) => {
  await mocks(page)
  let failed = true
  await page.route("**/api/dashboard/health", route => route.fulfill(
    failed ? json({ detail: "synthetic unavailable" }, 503) : json(HEALTH),
  ))
  await page.goto("/ui/health")
  await expect(page.getByRole("alert")).toContainText("Diagnostics are unavailable")
  await expect(page.getByText("0.0%", { exact: true })).toHaveCount(0)
  failed = false
  await page.getByRole("button", { name: "Retry", exact: true }).click()
  await expect(page.getByText("D", { exact: true })).toBeVisible()
})

test("overview preserves counts and cached brains when a later poll fails", async ({ page }, testInfo) => {
  await mocks(page)
  let failed = false
  await page.route("**/api/dashboard/brains", route => route.fulfill(
    failed ? json({ detail: "synthetic failure" }, 503) : json([BRAIN]),
  ))
  await page.goto("/ui")
  const row = page.getByRole("row").filter({ hasText: "default" })
  await expect(row).toContainText("17")
  await expect(row).toContainText("Pending")
  await expect(row.getByText("F", { exact: true })).toHaveCount(0)
  failed = true
  await expect(page.getByRole("alert")).toContainText("Brain summaries are unavailable", { timeout: 6000 })
  await expect(row).toContainText("17")
  await expect(page.getByText("No brains found", { exact: true })).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath("overview-cached-error.png") })
})


test("unavailable brain diagnostics link to a working Health retry", async ({ page }) => {
  await mocks(page)
  await page.route("**/api/dashboard/brains", route => route.fulfill(json([
    { ...BRAIN, diagnostic_status: "unavailable" },
  ])))
  let failed = true
  await page.route("**/api/dashboard/health", route => route.fulfill(
    failed ? json({ detail: "synthetic unavailable" }, 503) : json(HEALTH),
  ))
  await page.goto("/ui")
  await expect(page.getByText("Select the brain, then open Health to retry diagnostics.")).toBeVisible()
  await page.getByRole("link", { name: "Open Health", exact: true }).click()
  await expect(page.getByRole("alert")).toContainText("Diagnostics are unavailable")
  failed = false
  await page.getByRole("button", { name: "Retry", exact: true }).click()
  await expect(page.getByText("D", { exact: true })).toBeVisible()
})
