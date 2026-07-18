import { test, expect, type Page } from "@playwright/test"

/**
 * Browser QA for the U8 "Reasoning Training" dashboard page
 * (features/reasoning/ReasoningPage.tsx, route /reasoning, nav key `nav.reasoningTraining`).
 *
 * No backend runs on :8000 here, so every API call is intercepted and fulfilled with static
 * JSON. main.tsx mounts <BrowserRouter basename="/ui">, so navigations target /ui/<route>.
 */

const CATEGORIES = [
  "debugging",
  "planning",
  "implementation",
  "refactoring",
  "research",
  "verification",
  "architecture",
  "data-analysis",
]

const STATUS_BODY = {
  config: {
    mining_enabled: true,
    injection_enabled: false,
    mining_models: [],
    injection_map: {},
    categories: CATEGORIES,
    min_trace_chars: 200,
    max_trace_chars: 100000,
    scan_lookback_days: 30,
    retention_days: 90,
    max_traces_total: 20000,
    min_cluster_support: 3,
    min_confidence: 0.2,
    min_patterns_per_category: 3,
    injection_max_patterns: 5,
    injection_max_chars: 4000,
    distill_use_llm: false,
    redact_secrets: true,
    pattern_targets: {},
  },
  detected_models: ["claude-fable-5"],
  per_model: [
    {
      model: "claude-fable-5",
      trace_count: 5,
      unprocessed: 2,
      pattern_count: 3,
      has_thinking_text: true,
      last_trace_at: "2026-07-17T10:00:00",
      coverage_percent: 12.5,
    },
  ],
  coverage_by_model: {
    "claude-fable-5": CATEGORIES.map((c) => ({
      category: c,
      pattern_count: c === "debugging" ? 3 : 0,
      covered: c === "debugging",
    })),
  },
  total_traces: 5,
  unprocessed_traces: 2,
  total_patterns: 3,
  mining: {
    running: false,
    started_at: null,
    finished_at: null,
    phase: "idle",
    files_total: 0,
    files_scanned: 0,
    traces_found: 0,
    traces_ingested: 0,
    traces_processed: 0,
    patterns_learned: 0,
    current_model: null,
    models_done: 0,
    models_total: 0,
    dry_run: false,
    error: null,
  },
}

const PATTERNS_BODY = {
  patterns: [
    {
      id: "p1",
      source_model: "claude-fable-5",
      category: "debugging",
      title: "restate then verify",
      confidence: 1.0,
      frequency: 3,
      signature: "sig1",
    },
  ],
  total: 1,
  limit: 20,
  offset: 0,
}

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

function json(body: unknown) {
  return { status: 200, contentType: "application/json", body: JSON.stringify(body) }
}

async function installMocks(page: Page) {
  await page.route("**/api/dashboard/**", (route) => route.fulfill(json({})))
  await page.route("**/api/dashboard/stats", (route) =>
    route.fulfill(json({ active_brain: "test-brain" })),
  )
  await page.route("**/api/dashboard/brains", (route) => route.fulfill(json([])))
  await page.route("**/api/dashboard/fibers", (route) => route.fulfill(json({ fibers: [] })))
  await page.route("**/api/dashboard/health", (route) => route.fulfill(json(HEALTH_REPORT)))
  await page.route(
    (url) => url.pathname === "/health",
    (route) => route.fulfill(json({ status: "ok", version: "test" })),
  )
  // Patterns must be registered before status so the more specific glob wins
  // (Playwright checks handlers last-registered-first).
  await page.route("**/api/dashboard/reasoning/patterns**", (route) =>
    route.fulfill(json(PATTERNS_BODY)),
  )
  await page.route("**/api/dashboard/reasoning/status", (route) =>
    route.fulfill(json(STATUS_BODY)),
  )
}

test.describe("U8 Reasoning Training page", () => {
  test("renders populated reasoning content", async ({ page }) => {
    await installMocks(page)
    await page.goto("/ui/reasoning")

    await expect(
      page.getByRole("heading", { level: 1, name: "Reasoning Training" }),
    ).toBeVisible({ timeout: 20_000 })

    // KPI cards.
    await expect(page.getByText("Total traces")).toBeVisible()
    await expect(page.getByText("Learned patterns")).toBeVisible()

    // Coverage + config cards show the detected model.
    await expect(page.getByText("claude-fable-5").first()).toBeVisible()
    await expect(page.getByText("12.5%")).toBeVisible()

    // Patterns table row.
    await expect(page.getByText("restate then verify")).toBeVisible()

    // No raw untranslated i18n keys should leak into the DOM.
    await expect(page.getByText(/reasoning\.[a-zA-Z]/)).toHaveCount(0)

    await page.screenshot({ path: "e2e/__screenshots__/reasoning.png", fullPage: true })
  })

  test("sidebar Reasoning Training nav link navigates from another route", async ({ page }) => {
    await installMocks(page)
    await page.goto("/ui/health")
    await expect(page.getByRole("navigation")).toBeVisible({ timeout: 20_000 })

    const link = page.getByRole("navigation").getByRole("link", { name: "Reasoning Training" })
    await expect(link).toBeVisible()
    await link.click()

    await expect(page).toHaveURL(/\/ui\/reasoning$/)
    await expect(
      page.getByRole("heading", { level: 1, name: "Reasoning Training" }),
    ).toBeVisible()
  })
})
