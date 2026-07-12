import type { ReactNode } from "react"
import { useUncertainty } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Warning,
  ShieldWarning,
  Swap,
  ClockCountdown,
  GitBranch,
  Info,
} from "@phosphor-icons/react"
import { useTranslation } from "react-i18next"
import type {
  UncertaintyLevel,
  DriftClusterSample,
  LowEvidenceSample,
  SupersededSample,
} from "@/api/types"

const WITHIN_DAYS = 14
const ID_MAX = 16

const LEVEL_VARIANT: Record<UncertaintyLevel, "success" | "warning" | "destructive"> = {
  low: "success",
  medium: "warning",
  high: "destructive",
}

function truncateId(id: string): string {
  return id.length > ID_MAX ? `${id.slice(0, ID_MAX)}…` : id
}

function formatPct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

export default function UncertaintyPage() {
  const { data, isLoading } = useUncertainty(WITHIN_DAYS)
  const { t } = useTranslation()

  const counts = data?.counts
  const samples = data?.samples

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-4">
        <h1 className="font-display text-2xl font-bold">{t("uncertainty.title")}</h1>
        {data && (
          <Badge variant={LEVEL_VARIANT[data.level]} className="px-3 py-1 text-sm capitalize">
            {t(`uncertainty.level.${data.level}`)}
          </Badge>
        )}
        <span className="text-sm text-muted-foreground">
          {t("uncertainty.window", { days: WITHIN_DAYS })}
        </span>
      </div>

      {/* Top tiles */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label={t("uncertainty.contradictionRate")}
          value={data ? formatPct(data.contradiction_rate) : undefined}
          isLoading={isLoading}
        />
        <StatTile
          label={t("uncertainty.totalMemories")}
          value={data ? data.total_memories.toLocaleString() : undefined}
          isLoading={isLoading}
        />
        <CountTile
          icon={<Warning className="size-4 text-red-500" />}
          label={t("uncertainty.contradictions")}
          count={counts?.contradictions}
          isLoading={isLoading}
        />
        <CountTile
          icon={<ClockCountdown className="size-4 text-amber-500" />}
          label={t("uncertainty.expiring")}
          count={counts?.expiring}
          isLoading={isLoading}
        />
      </div>

      {/* Truncation note */}
      {data?.scan.typed_scan_truncated && (
        <p className="flex items-start gap-2 text-xs text-muted-foreground">
          <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          <span>
            {t("uncertainty.truncatedNote", { count: data.scan.typed_scanned })}
          </span>
        </p>
      )}

      {/* Detail cards */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Low-trust */}
        <SampleCard
          icon={<ShieldWarning className="size-5 text-orange-500" />}
          title={t("uncertainty.lowTrust")}
          count={counts?.low_evidence}
          isLoading={isLoading}
          isEmpty={!samples?.low_evidence.length}
        >
          <SampleTable
            columns={[t("uncertainty.fiberId"), t("uncertainty.trustScore")]}
            rows={samples?.low_evidence ?? []}
            renderRow={(s: LowEvidenceSample) => (
              <>
                <td className="py-2 pr-2 font-mono text-xs" title={s.fiber_id}>
                  {truncateId(s.fiber_id)}
                </td>
                <td className="py-2 text-right font-mono text-xs">
                  {s.trust_score.toFixed(2)}
                </td>
              </>
            )}
          />
        </SampleCard>

        {/* Superseded */}
        <SampleCard
          icon={<Swap className="size-5 text-indigo-500" />}
          title={t("uncertainty.superseded")}
          count={counts?.superseded}
          isLoading={isLoading}
          isEmpty={!samples?.superseded.length}
        >
          <SampleTable
            columns={[t("uncertainty.fiberId"), t("uncertainty.supersededBy")]}
            rows={samples?.superseded ?? []}
            renderRow={(s: SupersededSample) => (
              <>
                <td className="py-2 pr-2 font-mono text-xs" title={s.fiber_id}>
                  {truncateId(s.fiber_id)}
                </td>
                <td className="py-2 text-right font-mono text-xs" title={s.superseded_by}>
                  {truncateId(s.superseded_by)}
                </td>
              </>
            )}
          />
        </SampleCard>

        {/* Drift */}
        <SampleCard
          icon={<GitBranch className="size-5 text-cyan-500" />}
          title={t("uncertainty.drift")}
          count={counts?.drift_clusters}
          isLoading={isLoading}
          isEmpty={!samples?.drift_clusters.length}
          emptyLabel={counts?.drift_clusters === 0 ? t("uncertainty.driftSqliteOnly") : undefined}
        >
          <SampleTable
            columns={[t("uncertainty.canonical"), t("uncertainty.confidence")]}
            rows={samples?.drift_clusters ?? []}
            renderRow={(s: DriftClusterSample) => (
              <>
                <td className="py-2 pr-2 text-xs" title={s.canonical ?? ""}>
                  {s.canonical ? truncateId(s.canonical) : "—"}
                </td>
                <td className="py-2 text-right font-mono text-xs">
                  {s.confidence != null ? formatPct(s.confidence) : "—"}
                </td>
              </>
            )}
          />
        </SampleCard>
      </div>
    </div>
  )
}

function StatTile({
  label,
  value,
  isLoading,
}: {
  label: string
  value: string | undefined
  isLoading: boolean
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-8 w-24" />
        ) : (
          <p className="font-mono text-2xl font-bold">{value ?? "—"}</p>
        )}
      </CardContent>
    </Card>
  )
}

function CountTile({
  icon,
  label,
  count,
  isLoading,
}: {
  icon: ReactNode
  label: string
  count: number | undefined
  isLoading: boolean
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
          {icon}
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-8 w-16" />
        ) : (
          <p className="font-mono text-2xl font-bold">{count ?? 0}</p>
        )}
      </CardContent>
    </Card>
  )
}

function SampleCard({
  icon,
  title,
  count,
  isLoading,
  isEmpty,
  emptyLabel,
  children,
}: {
  icon: ReactNode
  title: string
  count: number | undefined
  isLoading: boolean
  isEmpty: boolean
  emptyLabel?: string
  children: ReactNode
}) {
  const { t } = useTranslation()
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="flex items-center gap-2">
            {icon}
            {title}
          </span>
          {!isLoading && (
            <Badge variant="secondary" className="font-mono">
              {count ?? 0}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-6 w-full" />
            ))}
          </div>
        ) : isEmpty ? (
          <p className="py-6 text-center text-sm text-muted-foreground">
            {emptyLabel ?? t("uncertainty.noSamples")}
          </p>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  )
}

function SampleTable<T>({
  columns,
  rows,
  renderRow,
}: {
  columns: [string, string]
  rows: T[]
  renderRow: (row: T) => ReactNode
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="pb-2 font-medium">{columns[0]}</th>
            <th className="pb-2 text-right font-medium">{columns[1]}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border/50">
              {renderRow(row)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
