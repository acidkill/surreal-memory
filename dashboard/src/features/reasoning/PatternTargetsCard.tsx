import { useEffect, useState } from "react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { apiErrorMessage } from "@/api/client"
import { useUpdateReasoningConfig } from "@/api/hooks/useReasoning"
import type { ReasoningStatusResponse } from "@/api/types"

interface Props {
  status: ReasoningStatusResponse
}

const STEP = 10
const MAX = 100

/**
 * Per-model distillation targets. Each mineable model gets a 0–100 (step 10)
 * slider for how many strategy patterns to distill; 0 means "detect only".
 * Changing a slider PUTs the full pattern_targets map (optimistic via the
 * shared config mutation).
 */
export function PatternTargetsCard({ status }: Props) {
  const { t } = useTranslation()
  const updateConfig = useUpdateReasoningConfig()

  // Only models that actually produce thinking text can be distilled.
  const models = status.per_model.filter((m) => m.has_thinking_text)
  const configTargets = status.config.pattern_targets

  const [targets, setTargets] = useState<Record<string, number>>(configTargets)
  // Re-sync local slider state when the server config changes.
  useEffect(() => {
    setTargets(configTargets)
  }, [configTargets])

  const setTarget = (model: string, value: number) => {
    const next = { ...targets, [model]: value }
    setTargets(next)
    updateConfig.mutate(
      { pattern_targets: next },
      {
        onSuccess: () => toast.success(t("reasoning.configSaved")),
        onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.configSaveFailed"))),
      },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("reasoning.targetsTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {models.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("reasoning.noModels")}</p>
        ) : (
          <div className="space-y-4">
            {models.map((m) => {
              const value = targets[m.model] ?? 0
              return (
                <div key={m.model} className="space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs">{m.model}</span>
                    <span className="font-mono text-xs tabular-nums text-muted-foreground">
                      {value}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={MAX}
                    step={STEP}
                    value={value}
                    aria-label={m.model}
                    onChange={(e) => setTarget(m.model, Number(e.target.value))}
                    className="w-full cursor-pointer accent-primary"
                  />
                  <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                    <span>
                      {t("reasoning.targetCounts", {
                        patterns: m.pattern_count,
                        traces: m.trace_count,
                      })}
                    </span>
                    {value === 0 && <span>{t("reasoning.targetsZeroHint")}</span>}
                  </div>
                </div>
              )
            })}
          </div>
        )}
        <p className="pt-1 text-xs text-muted-foreground">{t("reasoning.targetsHint")}</p>
      </CardContent>
    </Card>
  )
}
