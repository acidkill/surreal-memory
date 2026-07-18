import { useState } from "react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import { Trash } from "@phosphor-icons/react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { ApiError, apiErrorMessage } from "@/api/client"
import {
  useTriggerMining,
  useUpdateReasoningConfig,
  useWipeTraces,
} from "@/api/hooks/useReasoning"
import type { MiningJobState, ReasoningStatusResponse } from "@/api/types"

interface Props {
  status: ReasoningStatusResponse
}

function MiningProgressBlock({ mining }: { mining: MiningJobState }) {
  const { t } = useTranslation()
  const { phase, files_total, files_scanned, traces_found, traces_ingested } = mining
  const pct = files_total > 0 ? Math.round((files_scanned / files_total) * 100) : 0
  const scanning = phase === "scanning" || phase === "ingesting"
  const phaseLabel =
    phase === "scanning"
      ? t("reasoning.progressScanning")
      : phase === "ingesting"
        ? t("reasoning.progressIngesting")
        : phase === "distilling"
          ? t("reasoning.progressDistilling")
          : t("reasoning.progressDone")

  return (
    <div
      data-testid="mining-progress"
      className="space-y-2 rounded-md border border-border bg-muted/40 p-3"
    >
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">{phaseLabel}</span>
        {scanning && files_total > 0 && (
          <span className="font-mono tabular-nums text-muted-foreground">
            {t("reasoning.progressFiles", { scanned: files_scanned, total: files_total })}
          </span>
        )}
      </div>
      {scanning && files_total > 0 && (
        <div className="h-2 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      {scanning && (
        <p className="text-xs text-muted-foreground">
          {t("reasoning.progressTraces", { found: traces_found, ingested: traces_ingested })}
        </p>
      )}
      {phase === "distilling" && (
        <p className="font-mono text-xs text-muted-foreground">
          {t("reasoning.progressModels", {
            model: mining.current_model ?? "",
            done: mining.models_done,
            total: mining.models_total,
            patterns: mining.patterns_learned,
          })}
        </p>
      )}
    </div>
  )
}

export function MiningConfigCard({ status }: Props) {
  const { t } = useTranslation()
  const updateConfig = useUpdateReasoningConfig()
  const triggerMining = useTriggerMining()
  const wipeTraces = useWipeTraces()

  const miningEnabled = status.config.mining_enabled
  const selected = new Set(status.config.mining_models)
  const running = status.mining.running

  const [backfill, setBackfill] = useState(false)
  const [wipeModel, setWipeModel] = useState<string | null>(null)

  const confirmWipe = () => {
    if (!wipeModel) return
    const m = wipeModel
    setWipeModel(null)
    wipeTraces.mutate(m, {
      onSuccess: (res) => toast.success(t("reasoning.tracesWiped", { count: res.deleted })),
      onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.configSaveFailed"))),
    })
  }

  const toggleMining = (enabled: boolean) => {
    updateConfig.mutate(
      { mining_enabled: enabled },
      {
        onSuccess: () => toast.success(t("reasoning.configSaved")),
        onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.configSaveFailed"))),
      },
    )
  }

  const toggleModel = (model: string, on: boolean) => {
    const next = new Set(selected)
    if (on) next.add(model)
    else next.delete(model)
    updateConfig.mutate(
      { mining_models: [...next] },
      {
        onSuccess: () => toast.success(t("reasoning.configSaved")),
        onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.configSaveFailed"))),
      },
    )
  }

  const runMining = () => {
    triggerMining.mutate(
      { backfill },
      {
        onSuccess: () => toast.success(t("reasoning.miningStarted")),
        onError: (err) => {
          if (err instanceof ApiError && err.status === 409) {
            toast.error(t("reasoning.miningInProgress"))
          } else if (err instanceof ApiError && err.status === 400) {
            toast.error(t("reasoning.miningDisabled"))
          } else {
            toast.error(t("reasoning.miningFailed"))
          }
        },
      },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("reasoning.miningTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <label className="flex cursor-pointer items-center gap-3">
          <input
            type="checkbox"
            checked={miningEnabled}
            onChange={(e) => toggleMining(e.target.checked)}
            disabled={updateConfig.isPending}
            className="size-4 cursor-pointer"
          />
          <span>{t("reasoning.miningEnabled")}</span>
        </label>

        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">
            {t("reasoning.miningModels")}
          </p>
          {status.detected_models.length === 0 ? (
            <p className="text-xs text-muted-foreground">{t("reasoning.noModels")}</p>
          ) : (
            <div className="space-y-1.5">
              {status.per_model.map((m) => (
                <div key={m.model} className="flex items-center gap-2">
                  <label
                    className="flex flex-1 items-center gap-2"
                    title={m.has_thinking_text ? m.model : t("reasoning.noThinkingText")}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(m.model)}
                      disabled={!m.has_thinking_text || updateConfig.isPending}
                      onChange={(e) => toggleModel(m.model, e.target.checked)}
                      className="size-4 cursor-pointer disabled:cursor-not-allowed"
                    />
                    <span
                      className={
                        m.has_thinking_text ? "font-mono" : "font-mono text-muted-foreground"
                      }
                    >
                      {m.model}
                    </span>
                    {!m.has_thinking_text && (
                      <span className="text-xs text-muted-foreground">
                        ({t("reasoning.noThinkingShort")})
                      </span>
                    )}
                  </label>
                  {m.trace_count > 0 && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setWipeModel(m.model)}
                      disabled={wipeTraces.isPending}
                      title={t("reasoning.wipeTraces")}
                      aria-label={t("reasoning.wipeTraces")}
                    >
                      <Trash className="size-4" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}
          <p className="pt-1 text-xs text-muted-foreground">{t("reasoning.miningModelsHint")}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <label className="flex cursor-pointer items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={backfill}
              onChange={(e) => setBackfill(e.target.checked)}
              className="size-4 cursor-pointer"
            />
            <span>{t("reasoning.backfill")}</span>
          </label>
          <Button
            size="sm"
            onClick={runMining}
            disabled={!miningEnabled || running || triggerMining.isPending}
          >
            {running ? t("reasoning.miningRunning") : t("reasoning.runMining")}
          </Button>
        </div>

        {running && <MiningProgressBlock mining={status.mining} />}
      </CardContent>

      <ConfirmDialog
        open={wipeModel !== null}
        title={t("reasoning.wipeTracesTitle")}
        description={t("reasoning.wipeTracesDesc", { model: wipeModel ?? "" })}
        variant="destructive"
        confirmLabel={t("reasoning.wipeTraces")}
        onConfirm={confirmWipe}
        onCancel={() => setWipeModel(null)}
      />
    </Card>
  )
}
