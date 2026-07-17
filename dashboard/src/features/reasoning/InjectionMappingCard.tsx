import { useState } from "react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import { Plus, Trash } from "@phosphor-icons/react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { apiErrorMessage } from "@/api/client"
import { useUpdateReasoningConfig } from "@/api/hooks/useReasoning"
import type { ReasoningStatusResponse } from "@/api/types"

interface Props {
  status: ReasoningStatusResponse
}

interface Pair {
  id: string
  target: string
  source: string
}

// Monotonic row-id source. Stable per-row ids keep React from reusing one row's
// DOM node for another after a delete (which would strand focus mid-edit). A
// module counter avoids a ref (which eslint bans from render) and only advances
// when rows are (re)created, not on every render.
let pairSeq = 0
const nextPairId = () => `pair-${pairSeq++}`

const makePairs = (map: Record<string, string>): Pair[] =>
  Object.entries(map).map(([target, source]) => ({ id: nextPairId(), target, source }))

export function InjectionMappingCard({ status }: Props) {
  const { t } = useTranslation()
  const updateConfig = useUpdateReasoningConfig()

  const injectionEnabled = status.config.injection_enabled
  // Source models must have thinking text (the backend rejects others with 422).
  const sourceModels = status.per_model.filter((m) => m.has_thinking_text).map((m) => m.model)

  const [pairs, setPairs] = useState<Pair[]>(() => makePairs(status.config.injection_map))
  const [dirty, setDirty] = useState(false)

  // Reset the local editable list when the server map changes (after a save →
  // refetch gives a new object reference). Adjusting state during render is the
  // React-blessed alternative to a syncing effect.
  const [prevMap, setPrevMap] = useState(status.config.injection_map)
  if (status.config.injection_map !== prevMap) {
    setPrevMap(status.config.injection_map)
    setPairs(makePairs(status.config.injection_map))
    setDirty(false)
  }

  const toggleInjection = (enabled: boolean) => {
    updateConfig.mutate(
      { injection_enabled: enabled },
      {
        onSuccess: () => toast.success(t("reasoning.configSaved")),
        onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.configSaveFailed"))),
      },
    )
  }

  const setPair = (id: string, patch: Partial<Pair>) => {
    setPairs((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p)))
    setDirty(true)
  }
  const addPair = () => {
    setPairs((prev) => [...prev, { id: nextPairId(), target: "", source: sourceModels[0] ?? "" }])
    setDirty(true)
  }
  const removePair = (id: string) => {
    setPairs((prev) => prev.filter((p) => p.id !== id))
    setDirty(true)
  }

  const save = () => {
    const seen = new Set<string>()
    const map: Record<string, string> = {}
    for (const p of pairs) {
      const target = p.target.trim()
      const source = p.source.trim()
      if (!target || !source) continue
      if (seen.has(target)) {
        // Two rows with the same target would silently collapse — surface it.
        toast.error(t("reasoning.duplicateTarget", { target }))
        return
      }
      seen.add(target)
      map[target] = source
    }
    updateConfig.mutate(
      { injection_map: map },
      {
        onSuccess: () => {
          toast.success(t("reasoning.configSaved"))
          setDirty(false)
        },
        onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.configSaveFailed"))),
      },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("reasoning.injectionTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <label className="flex cursor-pointer items-center gap-3">
          <input
            type="checkbox"
            checked={injectionEnabled}
            onChange={(e) => toggleInjection(e.target.checked)}
            disabled={updateConfig.isPending}
            className="size-4 cursor-pointer"
          />
          <span>{t("reasoning.injectionEnabled")}</span>
        </label>

        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">
            {t("reasoning.injectionMap")}
          </p>
          {pairs.length === 0 && (
            <p className="text-xs text-muted-foreground">{t("reasoning.noMappings")}</p>
          )}
          {pairs.map((p) => (
            <div key={p.id} className="flex items-center gap-2">
              <select
                value={p.source}
                onChange={(e) => setPair(p.id, { source: e.target.value })}
                className="w-40 rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs"
                aria-label={t("reasoning.source")}
              >
                {!sourceModels.includes(p.source) && p.source && (
                  <option value={p.source}>{p.source}</option>
                )}
                {sourceModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <span className="text-muted-foreground">→</span>
              <input
                type="text"
                value={p.target}
                onChange={(e) => setPair(p.id, { target: e.target.value })}
                placeholder={t("reasoning.targetPlaceholder")}
                className="w-40 rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs"
                aria-label={t("reasoning.target")}
              />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => removePair(p.id)}
                aria-label={t("common.delete")}
              >
                <Trash className="size-4" />
              </Button>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-1">
            <Button
              variant="outline"
              size="sm"
              onClick={addPair}
              disabled={sourceModels.length === 0}
            >
              <Plus className="size-4" /> {t("reasoning.addMapping")}
            </Button>
            <Button size="sm" onClick={save} disabled={!dirty || updateConfig.isPending}>
              {t("reasoning.save")}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">{t("reasoning.injectionMapHint")}</p>
        </div>
      </CardContent>
    </Card>
  )
}
