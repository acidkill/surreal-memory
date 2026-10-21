import { test, expect } from "@playwright/test"

test.describe("Command palette", () => {
  test("typing narrows the page list", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByRole("navigation")).toBeVisible()
    await page.keyboard.press("Control+k")
    const input = page.getByRole("combobox")
    await expect(input).toBeVisible()

    const options = page.getByRole("option")
    await expect(options.first()).toBeVisible()
    const all = await options.count()

    await input.fill("Graph")
    await expect(options).toHaveCount(1)
    await expect(options.first()).toContainText("Graph")
    expect(all).toBeGreaterThan(1)
  })

  test("a query that matches nothing shows the empty state", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByRole("navigation")).toBeVisible()
    await page.keyboard.press("Control+k")
    await page.getByRole("combobox").fill("zzzz-no-such-page")
    await expect(page.getByRole("option")).toHaveCount(0)
  })
})
