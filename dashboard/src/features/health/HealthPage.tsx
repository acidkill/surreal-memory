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

const ENRICHMENT_TIPS = [
  {
    icon: Brain,
    title: "Ask your agent to remember frequently",
    color: "#6366f1",
    tips: [
      "After every decision: \"Remember that we chose PostgreSQL over MongoDB for the user service\"",
      "After debugging: \"Remember the root cause was a race condition in the WebSocket handler\"",
      "After meetings: \"Remember Alice suggested rate limiting at the API gateway level\"",
      "After learning: \"Remember that Vite HMR requires export default for React components\"",
    ],
  },
  {
    icon: Lightbulb,
    title: "Use rich, causal language",
    color: "#f59e0b",
    tips: [
      "BAD: \"PostgreSQL\" → creates a single flat neuron",
      "GOOD: \"We chose PostgreSQL over MongoDB because we need ACID transactions for payment processing\" → creates concept + entity + decision neurons with CAUSED_BY synapses",
      "Include WHY, not just WHAT — causal chains create richer neural connections",
      "Mention people, dates, and context — they become separate neurons linked by synapses",
    ],
  },
  {
    icon: Zap,
    title: "Diverse memory types = stronger recall",
    color: "#059669",
    tips: [
      "Facts: \"The API rate limit is 1000 req/min per user\"",
      "Decisions: \"We decided to use JWT over sessions because of microservice architecture\"",
      "Errors: \"Import failed because the column 'email' was renamed to 'user_email' in v3\"",
      "Insights: \"Pattern: always validate webhook signatures before processing payloads\"",
      "Workflows: \"Deploy process: lint → test → build → push → verify health check\"",
    ],
  },
  {
    icon: BookOpen,
    title: "Train from documents for permanent knowledge",
    color: "#06b6d4",
    tips: [
      "Use nmem_train to import docs (PDF, DOCX, MD) — trained memories never decay",
      "Use nmem_index to index your codebase — enables code-aware recall",
      "Pin critical memories with nmem_pin — they skip decay and consolidation",
      "Use nmem_eternal for project-level context that should persist across all sessions",
    ],
  },
] as const

export default function HealthPage() {
  const { data: health, isLoading } = useHealth()

  const radarData = health
    ? [
        { metric: "Purity", value: health.purity_score * 100 },
        { metric: "Freshness", value: health.freshness * 100 },
        { metric: "Connectivity", value: health.connectivity * 100 },
        { metric: "Diversity", value: health.diversity * 100 },
        { metric: "Consolidation", value: health.consolidation_ratio * 100 },
        { metric: "Activation", value: health.activation_efficiency * 100 },
        { metric: "Recall", value: health.recall_confidence * 100 },
        { metric: "Orphan Rate", value: (1 - health.orphan_rate) * 100 },
      ]
    : []

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center gap-4">
        <h1 className="font-display text-2xl font-bold">Health</h1>
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
            <CardTitle>Brain Metrics</CardTitle>
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
                    name="Health"
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
            <CardTitle>Warnings & Recommendations</CardTitle>
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
                    No warnings. Brain is healthy!
                  </p>
                )}

                {health?.recommendations && health.recommendations.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">
                      Recommendations
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

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Brain className="size-5 text-primary" />
            How to Enrich Your Brain
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded((v) => !v)}
            aria-label={expanded ? "Collapse tips" : "Expand tips"}
          >
            {expanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
            <span className="ml-1 text-xs">{expanded ? "Less" : "More"}</span>
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          A richer brain means better recall. Here&apos;s how to build detailed neurons and strong synapses.
        </p>
      </CardHeader>
      <CardContent>
        <div className={`grid grid-cols-1 gap-4 ${expanded ? "md:grid-cols-2" : "md:grid-cols-4"}`}>
          {ENRICHMENT_TIPS.map(({ icon: Icon, title, color, tips }) => (
            <div
              key={title}
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
                    <span className={tip.startsWith("BAD:") ? "text-destructive" : tip.startsWith("GOOD:") ? "text-primary" : ""}>
                      {tip}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
