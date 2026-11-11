import { afterEach, expect, it, vi } from "vitest"
import * as hooks from "./useDashboard"

vi.mock("@tanstack/react-query", () => ({
  useQuery: vi.fn(),
  keepPreviousData: vi.fn(),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
  useMutation: (options: { mutationFn: (id: string) => Promise<unknown> }) => ({
    mutateAsync: options.mutationFn,
  }),
}))

afterEach(() => vi.unstubAllGlobals())

it("sends exact full-delete intent through the dashboard request", async () => {
  const request = vi.fn<typeof fetch>().mockResolvedValue(
    new Response(JSON.stringify({ status: "deleted", brain_id: "target_full_id" })),
  )
  vi.stubGlobal("fetch", request)

  await hooks.useDeleteBrain().mutateAsync("target_full_id")

  expect(request).toHaveBeenCalledWith("/brain/target_full_id", expect.objectContaining({
    method: "DELETE",
    body: JSON.stringify({ confirm_brain_id: "target_full_id" }),
  }))
  expect(new Headers(request.mock.calls[0][1]?.headers).get("Content-Type")).toBe("application/json")
})
