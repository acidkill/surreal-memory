import { afterEach, expect, it, vi } from "vitest"
import { useMutation, useQuery } from "@tanstack/react-query"
import * as hooks from "./useDashboard"

vi.mock("@tanstack/react-query", () => ({
  useQuery: vi.fn(),
  keepPreviousData: vi.fn(),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
  useMutation: vi.fn((options: { mutationFn: (id: string) => Promise<unknown> }) => ({
    mutateAsync: options.mutationFn,
  })),
}))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it("sends exact full-delete intent through the dashboard request", async () => {
  const request = vi.fn<typeof fetch>().mockResolvedValue(
    new Response(JSON.stringify({ status: "deleted", brain_id: "target_full_id", full_delete_confirmed: true })),
  )
  vi.stubGlobal("fetch", request)

  await hooks.useDeleteBrain().mutateAsync("target_full_id")

  expect(request).toHaveBeenCalledWith("/brain/target_full_id", expect.objectContaining({
    method: "DELETE",
    body: JSON.stringify({ confirm_brain_id: "target_full_id" }),
  }))
  expect(new Headers(request.mock.calls[0][1]?.headers).get("Content-Type")).toBe("application/json")
})

it.each([
  { status: "deleted", brain_id: "target_full_id" },
  { status: "deleted", brain_id: "target_full_id", full_delete_confirmed: false },
  { status: "deleted", brain_id: "target_full_id", full_delete_confirmed: "true" },
  { status: "deleted", brain_id: "target_full_id", full_delete_confirmed: 1 },
  { status: "deleted", brain_id: "other", full_delete_confirmed: true },
  { status: "cleared", brain_id: "target_full_id", full_delete_confirmed: true },
  { status: "deleted", full_delete_confirmed: true },
  { brain_id: "target_full_id", full_delete_confirmed: true },
  null,
])("rejects an unconfirmed deletion reply without retry %#", async (reply) => {
  const request = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(reply)))
  vi.stubGlobal("fetch", request)

  await expect(hooks.useDeleteBrain().mutateAsync("target_full_id")).rejects.toThrow(
    "Full brain deletion could not be confirmed",
  )
  expect(request).toHaveBeenCalledTimes(1)
  expect(useMutation).toHaveBeenCalledWith(expect.objectContaining({ retry: false }))
})


it("polls pending health and stops when diagnostics finish", async () => {
  hooks.useHealth()
  const options = vi.mocked(useQuery).mock.calls.at(-1)![0]
  const poll = options.refetchInterval as (query: unknown) => number | false
  expect(poll({ state: { data: { status: "pending" } } })).toBe(2000)
  expect(poll({ state: { data: { grade: "D" } } })).toBe(false)
  expect(options.retry).toBe(false)
})

it.each(["useHealth", "useStats", "useBrains"] as const)("bounds and cancels %s requests", async (hook) => {
  const deadline = new AbortController()
  const timeout = vi.spyOn(AbortSignal, "timeout").mockReturnValue(deadline.signal)
  const request = vi.fn<typeof fetch>().mockImplementation((_path, init) =>
    new Promise((_resolve, reject) => {
      init!.signal!.addEventListener("abort", () => reject(init!.signal!.reason))
    }),
  )
  vi.stubGlobal("fetch", request)
  hooks[hook]()
  const options = vi.mocked(useQuery).mock.calls.at(-1)![0]
  const queryFn = options.queryFn as (ctx: { signal: AbortSignal }) => Promise<unknown>
  const reader = new AbortController()
  const result = queryFn({ signal: reader.signal })
  const assertion = expect(result).rejects.toThrow("deadline")
  deadline.abort(new Error("deadline"))
  await assertion
  expect(timeout).toHaveBeenCalledWith(15_000)

  timeout.mockReturnValue(new AbortController().signal)
  const next = queryFn({ signal: reader.signal })
  const cancelled = expect(next).rejects.toThrow("cancelled")
  reader.abort(new Error("cancelled"))
  await cancelled
  timeout.mockRestore()
})
