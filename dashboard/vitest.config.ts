import { defineConfig, mergeConfig } from "vitest/config"

import viteConfig from "./vite.config"

/**
 * Vitest covers the pure, browser-free logic (`src/**\/*.test.ts`).
 *
 * `e2e/` is explicitly out of scope: those are Playwright specs, and Playwright
 * refuses to run its own `test.describe` under another runner. Inheriting
 * vite.config keeps the `@/` alias identical to the app's.
 */
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      include: ["src/**/*.test.{ts,tsx}"],
      environment: "node",
    },
  }),
)
