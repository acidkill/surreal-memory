import { test, expect, type Page } from "@playwright/test"

/**
 * Browser QA for the U8 3D graph (features/graph/GraphPage.tsx, route /graph).
 *
 * Same harness shape as uncertainty.spec.ts: there is no backend on :8000 in
 * this environment, so every API call is intercepted with page.route(...) and
 * fulfilled with static JSON.
 *
 * NOTE: in this repo's dev environment Playwright's browser download could not
 * be completed, so these specs are the regression net for CI rather than the
 * evidence for U8. The unit-level guarantees (HTML escaping, particle budget,
 * neighbour index) are covered by src/features/graph/graph-data.test.ts, which
 * DOES run here; the mocked-backend flow below was additionally verified
 * against a real running server in a real browser before merge.
 *
 * URL note: main.tsx mounts <BrowserRouter basename="/ui">, so routes live
 * under /ui/<route>.
 */

// Fixture values are arbitrary but internally consistent (large enough to
// exercise the node-count slider's real-total path, not a sampled one).
const BRAIN_NEURON_TOTAL = 40_000

const STATS_BODY = {
  active_brain: "default",
  total_brains: 1,
  total_neurons: BRAIN_NEURON_TOTAL,
  total_synapses: 76_000,
  total_fibers: 6_200,
  health_grade: "B",
  purity_score: 71.4,
  brains: [],
}

function graphBody(nodeCount: number) {
  // neuron-0's content is a plain string XSS payload. No test below asserts on
  // its tooltip rendering — that would need a real hover over a WebGL-projected
  // node position, which isn't stable across headless runs. The actual escaping
  // guarantee is proven elsewhere: graph-data.test.ts exhaustively unit-tests
  // escapeHtml/toGraph3D against this exact payload (vitest, runs in this repo
  // today), and it was additionally verified live in a real browser against a
  // real server before merge. It's kept here only so a future Playwright run in
  // an environment where the browser install succeeds has a payload ready to
  // wire an assertion against.
  const neurons = Array.from({ length: nodeCount }, (_, i) => ({
    id: `neuron-${i}`,
    content: i === 0 ? "<img src=x onerror=alert(1)>" : `memory content ${i}`,
    type: ["concept", "entity", "time", "action", "state"][i % 5],
    metadata: {},
  }))
  const synapses = Array.from({ length: Math.max(0, nodeCount - 1) }, (_, i) => ({
    id: `syn-${i}`,
    source_id: `neuron-${i}`,
    target_id: `neuron-${i + 1}`,
    type: "related_to",
    weight: 0.5,
    direction: "bidirectional",
  }))
  return {
    neurons,
    synapses,
    fibers: [],
    total_neurons: nodeCount,
    total_synapses: synapses.length,
    stats: { neuron_count: nodeCount, synapse_count: synapses.length, fiber_count: 0 },
  }
}

function json(body: unknown) {
  return { status: 200, contentType: "application/json", body: JSON.stringify(body) }
}

/** Records every /api/graph limit the page asked for. */
async function installMocks(page: Page): Promise<number[]> {
  const requestedLimits: number[] = []

  await page.route("**/api/**", (route) => route.fulfill(json({})))
  await page.route("**/health", (route) => route.fulfill(json({ status: "ok", version: "3.3.2" })))
  await page.route("**/api/dashboard/stats", (route) => route.fulfill(json(STATS_BODY)))
  await page.route("**/api/graph**", (route) => {
    const limit = Number(new URL(route.request().url()).searchParams.get("limit") ?? "0")
    requestedLimits.push(limit)
    route.fulfill(json(graphBody(Math.min(limit, 120))))
  })

  return requestedLimits
}

test.describe("U8 Graph page", () => {
  test("renders the 3D canvas, a bounded slider and a truthful cap note", async ({ page }) => {
    const limits = await installMocks(page)
    await page.goto("/ui/graph")

    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 20_000 })

    // The slider is bounded by the API cap, not by the brain's neuron total.
    const slider = page.getByRole("slider")
    await expect(slider).toHaveAttribute("aria-valuemin", "100")
    await expect(slider).toHaveAttribute("aria-valuemax", "2000")
    await expect(slider).toHaveAttribute("aria-valuenow", "100")

    // The cap note names the REAL brain total, not the returned sample.
    await expect(page.getByText(String(BRAIN_NEURON_TOTAL.toLocaleString()))).toBeVisible()

    // WebGL canvas mounted by 3d-force-graph.
    await expect(page.locator("canvas")).toBeVisible()

    // The first request uses the default of 100, without waiting for stats.
    expect(limits[0]).toBe(100)

    await page.screenshot({ path: "e2e/__screenshots__/graph.png", fullPage: true })
  })

  test("dragging the slider to the end issues one debounced request at the cap", async ({
    page,
  }) => {
    const limits = await installMocks(page)
    await page.goto("/ui/graph")
    await expect(page.locator("canvas")).toBeVisible({ timeout: 20_000 })

    const before = limits.length
    await page.getByRole("slider").focus()
    await page.keyboard.press("End")

    await expect
      .poll(() => limits[limits.length - 1], { timeout: 5_000 })
      .toBe(2000)

    // Debounced: End is a single jump, so exactly one extra request.
    expect(limits.length - before).toBe(1)
  })

  test("a failed graph load offers a retry instead of an empty canvas", async ({ page }) => {
    await installMocks(page)
    await page.route("**/api/graph**", (route) => route.fulfill({ status: 500, body: "boom" }))
    await page.goto("/ui/graph")

    await expect(page.getByRole("button", { name: /retry/i })).toBeVisible({ timeout: 20_000 })
    await expect(page.locator("canvas")).toHaveCount(0)
  })
})
