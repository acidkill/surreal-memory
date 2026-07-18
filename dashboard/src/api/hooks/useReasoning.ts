import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"
import type {
  MineRequest,
  MineResponse,
  PatternsListResponse,
  ReasoningConfig,
  ReasoningConfigUpdate,
  ReasoningConfigUpdateResponse,
  ReasoningDeleteResponse,
  ReasoningStatusResponse,
} from "@/api/types"

const keys = {
  status: ["reasoning", "status"] as const,
  patterns: ["reasoning", "patterns"] as const,
  patternList: (model: string, category: string, limit: number, offset: number) =>
    ["reasoning", "patterns", model, category, limit, offset] as const,
}

export function useReasoningStatus() {
  return useQuery({
    queryKey: keys.status,
    queryFn: () => api.get<ReasoningStatusResponse>("/api/dashboard/reasoning/status"),
    // Poll while a mining job is running so live progress + counts update.
    refetchInterval: (query) => (query.state.data?.mining.running ? 1500 : false),
  })
}

export function useUpdateReasoningConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: ReasoningConfigUpdate) =>
      api.put<ReasoningConfigUpdateResponse>("/api/dashboard/reasoning/config", body),
    // Optimistically patch the cached status so toggles/checkboxes don't briefly
    // revert to the old value in the window before the refetch lands.
    onMutate: async (body) => {
      await queryClient.cancelQueries({ queryKey: keys.status })
      const prev = queryClient.getQueryData<ReasoningStatusResponse>(keys.status)
      if (prev) {
        queryClient.setQueryData<ReasoningStatusResponse>(keys.status, {
          ...prev,
          config: { ...prev.config, ...body } as ReasoningConfig,
        })
      }
      return { prev }
    },
    onError: (_err, _body, context) => {
      if (context?.prev) queryClient.setQueryData(keys.status, context.prev)
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: keys.status }),
  })
}

export function useTriggerMining() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: MineRequest) =>
      api.post<MineResponse>("/api/dashboard/reasoning/mine", body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.status }),
  })
}

export function useReasoningPatterns(model = "", category = "", limit = 50, offset = 0) {
  const params = new URLSearchParams()
  if (model) params.set("model", model)
  if (category) params.set("category", category)
  params.set("limit", String(limit))
  params.set("offset", String(offset))
  return useQuery({
    queryKey: keys.patternList(model, category, limit, offset),
    queryFn: () =>
      api.get<PatternsListResponse>(`/api/dashboard/reasoning/patterns?${params.toString()}`),
  })
}

export function useDeleteReasoningPattern() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      api.delete<ReasoningDeleteResponse>(
        `/api/dashboard/reasoning/patterns/${encodeURIComponent(id)}`,
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.status })
      queryClient.invalidateQueries({ queryKey: keys.patterns })
    },
  })
}

export function useDeletePatternsByModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (model: string) =>
      api.delete<ReasoningDeleteResponse>(
        `/api/dashboard/reasoning/patterns?model=${encodeURIComponent(model)}`,
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.status })
      queryClient.invalidateQueries({ queryKey: keys.patterns })
    },
  })
}

export function useWipeTraces() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (model: string) =>
      api.delete<ReasoningDeleteResponse>(
        `/api/dashboard/reasoning/traces?model=${encodeURIComponent(model)}`,
      ),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.status }),
  })
}
