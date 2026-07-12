import { test, expect, type Page } from "@playwright/test"

/**
 * Browser QA for the U6 "Uncertainty" dashboard page (features/uncertainty/UncertaintyPage.tsx,
 * route /uncertainty, nav label i18n key `nav.uncertainty`).
 *
 * There is no backend on :8000 in this environment, so every API call the page (and the app
 * shell) makes is intercepted with page.route(...) and fulfilled with static JSON. This lets the
 * page render fully populated content instead of an error/empty state.
 *
 * NOTE on the URL: main.tsx mounts <BrowserRouter basename="/ui"> (unless the path starts with
 * "/dashboard"), so the app only renders when the URL is under /ui. All navigations therefore
 * target /ui/<route>.
 */

const UNCERTAINTY_BODY = {
  level: "high",
  counts: {
    contradictions: 3,
    low_evidence: 2,
    superseded: 1,
    expiring: 4,
    drift_clusters: 0,
  },
  contradiction_rate: 0.12,
  total_memories: 25,
  scan: { typed_scanned: 25, typed_scan_truncated: false, contradictions_capped: false },
  samples: {
    low_evidence: [
      { fiber_id: "fiber-aaaaaaaa-1111", trust_score: 0.2 },
      { fiber_id: "fiber-bbbbbbbb-2222", trust_score: 0.35 },
    ],
    superseded: [{ fiber_id: "fiber-cccccccc-3333", superseded_by: "fiber-dddddddd-4444" }],
    drift_clusters: [],
  },
}

function json(body: unknown) {
  return { status: 200, contentType: "application/json", body: JSON.stringify(body) }
}

/**
 * Install all route mocks. Playwright checks route handlers in reverse registration order
 * (last-registered wins), so the broad /api/** catch-all is registered FIRST and the specific
 * endpoints override it.
 */
// Minimal-but-valid HealthReport so the /health route (used as the "other route" for the
// navigation test) renders instead of crashing on undefined nested fields.
const HEALTH_REPORT = {
  grade: "A",
  purity_score: 0.9,
  connectivity: 0.8,
  diversity: 0.7,
  freshness: 0.85,
  consolidation_ratio: 0.5,
  orphan_rate: 0.1,
  activation_efficiency: 0.6,
  recall_confidence: 0.75,
  neuron_count: 100,
  synapse_count: 200,
  fiber_count: 50,
  contradiction_count: 3,
  conflict_rate: 0.12,
  warnings: [],
  recommendations: [],
  top_penalties: [],
}

async function installMocks(page: Page) {
  // Catch-all for any dashboard API call the shell may fire (license, watcher, etc.).
  // NOTE: scope this to the real backend prefix `/api/dashboard/` — a broad `**/api/**`
  // would also swallow the app's own Vite source modules (e.g. /src/api/client.ts) and
  // break the module graph, leaving a blank page.
  await page.route("**/api/dashboard/**", (route) => route.fulfill(json({})))

  // App-shell fetches (TopBar + CommandPalette).
  await page.route("**/api/dashboard/stats", (route) =>
    route.fulfill(json({ active_brain: "test-brain" })),
  )
  await page.route("**/api/dashboard/brains", (route) => route.fulfill(json([])))
  await page.route("**/api/dashboard/fibers", (route) => route.fulfill(json({ fibers: [] })))
  await page.route("**/api/dashboard/health", (route) => route.fulfill(json(HEALTH_REPORT)))

  // Version endpoint. Match the exact `/health` pathname via a function matcher — a glob like
  // `**/health` would also intercept the top-level document navigation to `/ui/health` and
  // serve this JSON in place of the SPA.
  await page.route(
    (url) => url.pathname === "/health",
    (route) => route.fulfill(json({ status: "ok", version: "test" })),
  )

  // The page under test.
  await page.route("**/api/dashboard/uncertainty**", (route) =>
    route.fulfill(json(UNCERTAINTY_BODY)),
  )
}

test.describe("U6 Uncertainty page", () => {
  test("renders populated uncertainty content", async ({ page }) => {
    await installMocks(page)
    await page.goto("/ui/uncertainty")

    // Heading (uncertainty.title) — the <h1>, distinct from the sidebar nav link.
    // Generous timeout: the dev server compiles the module graph on the first cold request.
    await expect(page.getByRole("heading", { level: 1, name: "Uncertainty" })).toBeVisible({
      timeout: 20_000,
    })

    // "high" level badge -> t("uncertainty.level.high") == "High".
    await expect(page.getByText("High", { exact: true })).toBeVisible()

    // Window sub-label confirms the 14-day query rendered.
    await expect(page.getByText("Last 14 days")).toBeVisible()

    // Top tiles: contradiction rate (0.12 -> "12.0%") and total memories (25).
    await expect(page.getByText("Contradiction Rate")).toBeVisible()
    await expect(page.getByText("12.0%")).toBeVisible()
    await expect(page.getByText("Total Memories")).toBeVisible()
    await expect(page.getByText("25", { exact: true })).toBeVisible()

    // Low-trust sample table: truncated fiber id + trust score row.
    // "fiber-aaaaaaaa-1111" (19 chars) is truncated to 16 chars + ellipsis.
    await expect(page.getByText(/fiber-aaaaaaaa-1/)).toBeVisible()
    await expect(page.getByText("0.20", { exact: true })).toBeVisible()

    // Drift card shows its SQLite-only note because drift_clusters is empty & count === 0.
    await expect(page.getByText("Drift detection is SQLite-only.")).toBeVisible()

    // No raw untranslated i18n keys should leak into the DOM.
    await expect(page.getByText(/uncertainty\.[a-zA-Z]/)).toHaveCount(0)

    await page.screenshot({ path: "e2e/__screenshots__/uncertainty.png", fullPage: true })
  })

  test("sidebar Uncertainty nav link navigates from another route", async ({ page }) => {
    await installMocks(page)

    // Start on a different route.
    await page.goto("/ui/health")
    // Generous timeout for the first cold compile of the dev server.
    await expect(page.getByRole("navigation")).toBeVisible({ timeout: 20_000 })

    const nav = page.getByRole("navigation")
    const uncertaintyLink = nav.getByRole("link", { name: "Uncertainty" })
    await expect(uncertaintyLink).toBeVisible()

    await uncertaintyLink.click()

    await expect(page).toHaveURL(/\/ui\/uncertainty$/)
    await expect(page.getByRole("heading", { level: 1, name: "Uncertainty" })).toBeVisible()
  })
})
