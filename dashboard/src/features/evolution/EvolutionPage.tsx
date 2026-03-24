import { useEvolution } from "@/api/hooks/useDashboard"
import { ProGate } from "@/components/common/ProGate"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import { useTranslation } from "react-i18next"

const STAGE_COLORS: Record<string, string> = {
  short_term: "var(--color-chart-4)",
  working: "var(--color-chart-3)",
  episodic: "var(--color-chart-1)",
  semantic: "var(--color-chart-2)",
}

function ProgressBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-medium">{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function EvolutionPage() {
  const { data: evo, isLoading } = useEvolution()
  const { t } = useTranslation()

  const stageData = evo?.stage_distribution
    ? [
        { stage: t("evolution.shortTerm"), count: evo.stage_distribution.short_term, key: "short_term" },
        { stage: t("evolution.working"), count: evo.stage_distribution.working, key: "working" },
        { stage: t("evolution.episodic"), count: evo.stage_distribution.episodic, key: "episodic" },
        { stage: t("evolution.semantic"), count: evo.stage_distribution.semantic, key: "semantic" },
      ]
    : []

  return (
    <ProGate label={t("license.pro_feature", "Pro Feature")}>
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">{t("evolution.title")}</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>{t("evolution.brainMetrics")}</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : evo ? (
              <div className="space-y-4">
                <ProgressBar label={t("evolution.maturity")} value={evo.maturity_level} />
                <ProgressBar label={t("evolution.plasticity")} value={evo.plasticity} />
                <ProgressBar label={t("evolution.semanticRatio")} value={evo.semantic_ratio} />
                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">{t("evolution.totalFibers")}</p>
                    <p className="font-mono text-lg font-bold">
                      {evo.total_fibers.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">{t("evolution.totalNeurons")}</p>
                    <p className="font-mono text-lg font-bold">
                      {evo.total_neurons.toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

        {/* Stage Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>{t("evolution.stageDistribution")}</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={stageData}>
                  <XAxis
                    dataKey="stage"
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
                  />
                  <YAxis
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--color-card)",
                      border: "1px solid var(--color-border)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {stageData.map((entry) => (
                      <Cell
                        key={entry.stage}
                        fill={STAGE_COLORS[entry.key] ?? "var(--color-chart-1)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
    </ProGate>
  )
}
