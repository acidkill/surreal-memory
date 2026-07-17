import { useEffect, useRef } from "react"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useReasoningStatus } from "@/api/hooks/useReasoning"
import type { ReasoningStatusResponse } from "@/api/types"
import { MiningConfigCard } from "./MiningConfigCard"
import { InjectionMappingCard } from "./InjectionMappingCard"
import { PatternsTable } from "./PatternsTable"

function Kpi({ label, value }: { label: string; value: number }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="font-mono text-2xl font-bold">{value.toLocaleString()}</p>
      </CardContent>
    </Card>
  )
}

function CoverageCard({ status }: { status: ReasoningStatusResponse }) {
  const { t } = useTranslation()
  const models = status.per_model.filter((m) => m.pattern_count > 0)
  if (models.length === 0) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("reasoning.coverageTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {models.map((m) => {
          const cats = status.coverage_by_model[m.model] ?? []
          return (
            <div key={m.model} className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs">{m.model}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  {m.coverage_percent}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-500"
                  style={{ width: `${m.coverage_percent}%` }}
                />
              </div>
              {cats.length > 0 && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {cats.map((c) => (
                    <Badge key={c.category} variant={c.covered ? "success" : "outline"}>
                      {c.category} ({c.pattern_count})
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

export default function ReasoningPage() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const { data: status, isLoading, isError, refetch } = useReasoningStatus()

  // A mining run creates new pattern fibers; when it finishes (running true→false),
  // refresh the patterns table (the status poll already refreshes KPIs/coverage).
  const running = status?.mining.running ?? false
  const prevRunning = useRef(running)
  useEffect(() => {
    if (prevRunning.current && !running) {
      queryClient.invalidateQueries({ queryKey: ["reasoning", "patterns"] })
    }
    prevRunning.current = running
  }, [running, queryClient])

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="font-display text-2xl font-bold">{t("reasoning.pageTitle")}</h1>
        <p className="text-sm text-muted-foreground">{t("reasoning.pageSubtitle")}</p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      ) : isError || !status ? (
        <Card>
          <CardContent className="space-y-2 p-6">
            <p className="text-sm text-destructive">{t("reasoning.statusError")}</p>
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              {t("common.retry")}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
            <Kpi label={t("reasoning.kpiTraces")} value={status.total_traces} />
            <Kpi label={t("reasoning.kpiUnprocessed")} value={status.unprocessed_traces} />
            <Kpi label={t("reasoning.kpiPatterns")} value={status.total_patterns} />
            <Kpi label={t("reasoning.kpiModels")} value={status.detected_models.length} />
          </div>

          {status.total_traces === 0 && !status.mining.running && (
            <Card>
              <CardContent className="p-6 text-sm text-muted-foreground">
                {t("reasoning.emptyState")}
              </CardContent>
            </Card>
          )}

          <CoverageCard status={status} />

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <MiningConfigCard status={status} />
            <InjectionMappingCard status={status} />
          </div>

          <PatternsTable
            detectedModels={status.detected_models}
            categories={status.config.categories}
          />
        </>
      )}
    </div>
  )
}
