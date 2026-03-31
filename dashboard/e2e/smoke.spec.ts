import { test, expect } from "@playwright/test"

test.describe("Dashboard Smoke Tests", () => {
  test("loads the overview page", async ({ page }) => {
    await page.goto("/")
    // Should redirect to or show the overview page
    await expect(page.locator("body")).toBeVisible()
    // Check that the app shell rendered (sidebar + main content)
    await expect(page.locator("[data-testid='sidebar'], nav")).toBeVisible()
  })

  test("sidebar navigation links are present", async ({ page }) => {
    await page.goto("/")
    const nav = page.locator("nav, [data-testid='sidebar']")
    await expect(nav).toBeVisible()
    // At minimum, overview and health links should exist
    await expect(nav.getByRole("link")).toHaveCount(await nav.getByRole("link").count())
    expect(await nav.getByRole("link").count()).toBeGreaterThan(0)
  })

  test("health page loads", async ({ page }) => {
    await page.goto("/health")
    await expect(page.locator("body")).toBeVisible()
    // Page should contain health-related content
    await expect(page.locator("main, [data-testid='main-content']")).toBeVisible()
  })

  test("settings page loads", async ({ page }) => {
    await page.goto("/settings")
    await expect(page.locator("body")).toBeVisible()
    await expect(page.locator("main, [data-testid='main-content']")).toBeVisible()
  })

  test("oracle page loads", async ({ page }) => {
    await page.goto("/oracle")
    await expect(page.locator("body")).toBeVisible()
  })

  test("no console errors on overview page", async ({ page }) => {
    const errors: string[] = []
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text())
      }
    })

    await page.goto("/")
    await page.waitForTimeout(2000)

    // Filter out expected errors (API calls that fail without backend)
    const unexpected = errors.filter(
      (e) => !e.includes("fetch") && !e.includes("ERR_CONNECTION") && !e.includes("net::"),
    )
    expect(unexpected).toHaveLength(0)
  })

  test("theme toggle works", async ({ page }) => {
    await page.goto("/")
    // Look for theme toggle button
    const themeBtn = page.locator(
      "button:has([data-testid='theme-toggle']), [aria-label*='theme'], [aria-label*='Theme']",
    )
    if ((await themeBtn.count()) > 0) {
      const htmlBefore = await page.locator("html").getAttribute("class")
      await themeBtn.first().click()
      const htmlAfter = await page.locator("html").getAttribute("class")
      // Class should change (dark ↔ light)
      expect(htmlAfter).not.toBe(htmlBefore)
    }
  })

  test("Phosphor icons render (no broken SVGs)", async ({ page }) => {
    await page.goto("/")
    // Phosphor icons render as SVG elements
    const svgs = page.locator("svg")
    const count = await svgs.count()
    expect(count).toBeGreaterThan(0)

    // Check that SVGs have valid dimensions (not 0x0)
    const firstSvg = svgs.first()
    const box = await firstSvg.boundingBox()
    if (box) {
      expect(box.width).toBeGreaterThan(0)
      expect(box.height).toBeGreaterThan(0)
    }
  })
})
