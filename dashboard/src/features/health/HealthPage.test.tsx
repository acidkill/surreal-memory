import { renderToStaticMarkup } from "react-dom/server"
import { beforeEach, expect, it, vi } from "vitest"
import { useHealth, useConsolidationStatus } from "@/api/hooks/useDashboard"
import HealthPage from "./HealthPage"

vi.mock("@/api/hooks/useDashboard", () => ({ useHealth: vi.fn(), useConsolidationStatus: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback?: string | { returnObjects?: boolean }) =>
    typeof fallback === "object" && fallback.returnObjects ? [] : typeof fallback === "string" ? fallback : key,
}) }))

beforeEach(() => {
  vi.mocked(useConsolidationStatus).mockReturnValue({ isLoading: false } as ReturnType<typeof useConsolidationStatus>)
})

it("shows pending diagnostics without fabricated metrics or warnings", () => {
  vi.mocked(useHealth).mockReturnValue({ data: { status: "pending" }, isLoading: false } as ReturnType<typeof useHealth>)
  const html = renderToStaticMarkup(<HealthPage />)
  expect(html).toContain("Diagnostics are being calculated")
  expect(html).not.toContain("0.0%")
  expect(html).not.toContain("health.noWarnings")
  expect(html).not.toContain(">F<")
})

it("shows an explicit retry action when diagnostics fail", () => {
  vi.mocked(useHealth).mockReturnValue({ isError: true, isLoading: false, refetch: vi.fn() } as unknown as ReturnType<typeof useHealth>)
  const html = renderToStaticMarkup(<HealthPage />)
  expect(html).toContain("Diagnostics are unavailable")
  expect(html).toContain("Retry")
  expect(html).not.toContain("0.0%")
})
