import type { ConsolidationState, ConsolidationStatus } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

const STATUS_LABELS: Record<ConsolidationState, string> = {
  blocked: "Blocked",
  failed: "Failed",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  deferred_resource: "Waiting for resources",
  deferred_lease: "Another worker is active",
  unknown: "Unknown",
  idle: "No saved run",
}

function formatTime(value: string | null | undefined) {
  if (!value) return "Unknown"
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? "Unknown"
    : date.toISOString().replace("T", " ").replace(".000Z", " UTC")
}

export default function ConsolidationWorkPanel({
  status,
  loading = false,
  unavailable = false,
}: {
  status?: ConsolidationStatus
  loading?: boolean
  unavailable?: boolean
}) {
  const saved = unavailable ? undefined : status
  const state = saved?.state ?? "unknown"
  const variant = state === "blocked" || state === "failed"
    ? "destructive"
    : state === "completed"
      ? "success"
      : state === "paused" || state.startsWith("deferred_")
        ? "warning"
        : "secondary"
  const completed = saved?.completed_strategies
  const requested = saved?.requested_strategies
  const fields = [
    ["Run", saved?.run_id ?? "Unknown"],
    ["Strategies completed", completed != null && requested != null ? `${completed} / ${requested}` : "Unknown"],
    ["Current strategy", saved?.current_strategy ?? (state === "completed" ? "Completed" : "Unknown")],
    ["Work sequence", saved?.progress_known && saved.work_progress_seq != null ? String(saved.work_progress_seq) : "Unknown"],
    ["Last useful work", formatTime(saved?.last_work_progress_at)],
    ["Scheduler observation", formatTime(saved?.last_observed_at)],
    ["Last confirmed success", formatTime(saved?.last_confirmed_success_at)],
  ] as const

  return (
    <Card aria-labelledby="consolidation-work-title" aria-busy={loading} className="min-w-0" style={{ overflowWrap: "anywhere" }}>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle id="consolidation-work-title" className="min-w-0 max-w-full">Consolidation work · default</CardTitle>
          <Badge variant={variant}>{STATUS_LABELS[state] ?? "Unknown"}</Badge>
        </div>
        <p className="text-sm text-muted-foreground">Committed work from the saved run.</p>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div role="status" aria-label="Loading consolidation status"><Skeleton className="h-20 w-full" /></div>
        ) : (
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {fields.map(([label, value]) => (
              <div key={label} className="min-w-0">
                <dt className="text-xs text-muted-foreground">{label}</dt>
                <dd className="mt-1 break-words text-sm font-medium">{value}</dd>
              </div>
            ))}
          </dl>
        )}
        {!loading && (unavailable || !saved) && (
          <p className="mt-3 text-sm text-muted-foreground">Saved consolidation status is unavailable.</p>
        )}
        {saved && !saved.progress_known && (
          <p className="mt-3 text-xs text-muted-foreground">Useful-work history is unknown for this checkpoint.</p>
        )}
        {saved?.observation_stale && (
          <p className="mt-3 text-sm text-amber-600">Scheduler observation is stale.</p>
        )}
      </CardContent>
    </Card>
  )
}
