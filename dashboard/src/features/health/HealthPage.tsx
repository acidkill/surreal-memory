import { useState } from "react"
import { useHealth } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { ChevronDown, ChevronUp, Brain, Lightbulb, Zap, BookOpen } from "lucide-react"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts"
import { useTranslation } from "react-i18next"

const ENRICHMENT_ICONS = [Brain, Lightbulb, Zap, BookOpen] as const
const ENRICHMENT_COLORS = ["#6366f1", "#f59e0b", "#059669", "#06b6d4"] as const
const ENRICHMENT_KEYS = ["remember", "causal", "diverse", "train"] as const

export default function HealthPage() {
  const { data: health, isLoading } = useHealth()
  const { t } = useTranslation()

  const radarData = health
    ? [
        { metric: t("health.purity"), value: health.purity_score * 100 },
        { metric: t("health.freshness"), value: health.freshness * 100 },
        { metric: t("health.connectivity"), value: health.connectivity * 100 },
        { metric: t("health.diversity"), value: health.diversity * 100 },
        { metric: t("health.consolidation"), value: health.consolidation_ratio * 100 },
        { metric: t("health.activation"), value: health.activation_efficiency * 100 },
        { metric: t("health.recall"), value: health.recall_confidence * 100 },
        { metric: t("health.orphanRate"), value: (1 - health.orphan_rate) * 100 },
      ]
    : []

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center gap-4">
        <h1 className="font-display text-2xl font-bold">{t("health.title")}</h1>
        {health && (
          <Badge
            variant={
              health.grade.startsWith("A")
                ? "success"
                : health.grade.startsWith("B")
                  ? "secondary"
                  : "warning"
            }
            className="text-lg px-3 py-1"
          >
            {health.grade}
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>{t("health.brainMetrics")}</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-80 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="var(--color-border)" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 100]}
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
                  />
                  <Radar
                    name={t("health.radarName")}
                    dataKey="value"
                    stroke="var(--color-primary)"
                    fill="var(--color-primary)"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Warnings */}
        <Card>
          <CardHeader>
            <CardTitle>{t("health.warnings")}</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {health?.warnings && health.warnings.length > 0 ? (
                  <div className="space-y-2">
                    {health.warnings.map((w, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded-lg border border-border p-3"
                      >
                        <Badge
                          variant={
                            w.severity === "critical"
                              ? "destructive"
                              : w.severity === "warning"
                                ? "warning"
                                : "secondary"
                          }
                          className="mt-0.5 shrink-0"
                        >
                          {w.severity}
                        </Badge>
                        <span className="text-sm">{w.message}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {t("health.noWarnings")}
                  </p>
                )}

                {health?.recommendations && health.recommendations.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">
                      {t("health.recommendations")}
                    </h3>
                    <ul className="space-y-1 text-sm">
                      {health.recommendations.map((r, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="text-primary">-</span>
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      {/* Memory Enrichment Guide */}
      <MemoryEnrichmentGuide />
    </div>
  )
}

function MemoryEnrichmentGuide() {
  const [expanded, setExpanded] = useState(false)
  const { t } = useTranslation()

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Brain className="size-5 text-primary" />
            {t("health.enrichTitle")}
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded((v) => !v)}
            aria-label={expanded ? t("health.collapseTips") : t("health.expandTips")}
          >
            {expanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
            <span className="ml-1 text-xs">{expanded ? t("health.less") : t("health.more")}</span>
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          {t("health.enrichDesc")}
        </p>
      </CardHeader>
      <CardContent>
        <div className={`grid grid-cols-1 gap-4 ${expanded ? "md:grid-cols-2" : "md:grid-cols-4"}`}>
          {ENRICHMENT_KEYS.map((key, idx) => {
            const Icon = ENRICHMENT_ICONS[idx]
            const color = ENRICHMENT_COLORS[idx]
            const title = t(`enrichment.${key}Title`)
            const tips = t(`enrichment.${key}Tips`, { returnObjects: true }) as string[]

            return (
              <div
                key={key}
                className="rounded-lg border border-border p-4 transition-shadow hover:shadow-sm"
              >
                <div className="mb-3 flex items-center gap-2">
                  <div
                    className="flex size-8 items-center justify-center rounded-lg"
                    style={{ backgroundColor: `${color}15` }}
                  >
                    <Icon className="size-4" style={{ color }} />
                  </div>
                  <h3 className="text-sm font-semibold">{title}</h3>
                </div>
                <ul className="space-y-2">
                  {(expanded ? tips : tips.slice(0, 2)).map((tip, i) => (
                    <li key={i} className="flex gap-2 text-xs leading-relaxed">
                      <span className="mt-0.5 shrink-0 text-muted-foreground">-</span>
                      <span className={tip.startsWith("BAD:") || tip.startsWith("TỆ:") ? "text-destructive" : tip.startsWith("GOOD:") || tip.startsWith("TỐT:") ? "text-primary" : ""}>
                        {tip}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
