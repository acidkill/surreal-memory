import { useCallback, useMemo, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X } from "@phosphor-icons/react"
import { useTranslation } from "react-i18next"

import { useGraph, useStats } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { useDebounce } from "@/hooks/use-debounce"

import { NetworkGraph3D } from "./NetworkGraph3D"
import { LEGEND_KEYS, colorForType } from "./graph-colors"
import type { Graph3DNode } from "./graph-data"

/** The API caps /api/graph at le=2000; the slider must not promise more. */
const API_NODE_CAP = 2000
const MIN_NODES = 100
const NODE_STEP = 50
const SLIDER_DEBOUNCE_MS = 400

export default function GraphPage() {
  const { t } = useTranslation()

  const [nodeCount, setNodeCount] = useState(MIN_NODES)
  const [focused, setFocused] = useState<Graph3DNode | null>(null)
  const [clearSignal, setClearSignal] = useState(0)

  // Only bounds the slider — the graph itself does not wait for it.
  const { data: stats } = useStats()
  const brainNeurons = stats?.total_neurons
  const sliderMax = useMemo(
    () => (brainNeurons ? Math.max(MIN_NODES, Math.min(API_NODE_CAP, brainNeurons)) : API_NODE_CAP),
    [brainNeurons],
  )

  // Clamped during render rather than corrected by a setState in an effect:
  // switching to a smaller brain must not leave the handle past the end, and
  // deriving avoids the extra render pass (and the cascading-render lint rule).
  const effectiveCount = Math.min(nodeCount, sliderMax)
  const debouncedCount = useDebounce(effectiveCount, SLIDER_DEBOUNCE_MS)

  // Fires immediately at the default of 100 rather than waiting for stats — the
  // graph is the point of the page.
  const { data: graph, isLoading, isFetching, isError, refetch } = useGraph(debouncedCount)

  const handleClearFocus = useCallback(() => {
    setFocused(null)
    setClearSignal((n) => n + 1)
  }, [])

  const isCapped = brainNeurons != null && brainNeurons > API_NODE_CAP

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4 p-4">
      {/* Header: title + node-count slider */}
      <div className="flex flex-wrap items-center justify-between gap-4 shrink-0">
        <h1 className="font-display text-2xl font-bold">{t("graph.title")}</h1>
        <div className="flex min-w-[18rem] flex-1 items-center justify-end gap-3">
          <span className="shrink-0 text-sm text-muted-foreground">{t("graph.nodes")}</span>
          <Slider
            aria-label={t("graph.nodes")}
            className="max-w-xs"
            min={MIN_NODES}
            max={sliderMax}
            step={NODE_STEP}
            value={[effectiveCount]}
            disabled={brainNeurons == null}
            onValueChange={([value]) => setNodeCount(value)}
          />
          <span className="w-12 shrink-0 text-right font-mono text-sm tabular-nums">
            {effectiveCount.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Truthful cap annotation: the graph payload's own totals describe the
          returned sample and the connected-node universe, not the brain. */}
      {isCapped && (
        <p className="shrink-0 text-xs text-muted-foreground">
          {t("graph.cappedNote", {
            shown: effectiveCount.toLocaleString(),
            total: brainNeurons.toLocaleString(),
          })}
        </p>
      )}

      <Card className="relative flex min-h-0 flex-1 flex-col">
        <CardHeader className="flex shrink-0 flex-row items-center justify-between px-4 py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            {t("graph.networkVisualization")}
            {graph && (
              <span className="font-normal text-muted-foreground">
                {t("graph.nodesCount", {
                  nodes: graph.neurons.length.toLocaleString(),
                  edges: graph.synapses.length.toLocaleString(),
                })}
              </span>
            )}
            {isFetching && !isLoading && (
              <span className="text-xs font-normal text-muted-foreground">
                {t("graph.updating")}
              </span>
            )}
          </CardTitle>
          <div className="flex items-center gap-3">
            {LEGEND_KEYS.map((key) => (
              <div key={key} className="flex items-center gap-1">
                <div
                  className="size-2 rounded-full"
                  style={{ backgroundColor: colorForType(key) }}
                />
                <span className="text-[10px] capitalize text-muted-foreground">
                  {t(`graph.${key}`)}
                </span>
              </div>
            ))}
          </div>
        </CardHeader>

        <CardContent className="relative min-h-0 flex-1 p-2">
          {isError ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground">{t("graph.loadError")}</p>
              <Button size="sm" variant="outline" onClick={() => refetch()}>
                {t("graph.retry")}
              </Button>
            </div>
          ) : isLoading ? (
            <Skeleton className="size-full" />
          ) : graph && graph.neurons.length > 0 ? (
            <>
              <div className="size-full overflow-hidden rounded-lg border border-border bg-muted/20">
                <NetworkGraph3D
                  data={graph}
                  onFocusChange={setFocused}
                  clearFocusSignal={clearSignal}
                />
              </div>
              {!focused && (
                <p className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 text-[11px] text-muted-foreground">
                  {t("graph.focusHint")}
                </p>
              )}
              <AnimatePresence>
                {focused && (
                  <motion.div
                    initial={{ opacity: 0, x: 16 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 16 }}
                    transition={{ duration: 0.18 }}
                    className="absolute right-4 top-4 w-72 rounded-lg border border-border bg-background/95 p-3 shadow-lg backdrop-blur"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <Badge variant="secondary" className="shrink-0">
                        {focused.type}
                      </Badge>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="size-6 shrink-0"
                        aria-label={t("graph.close")}
                        onClick={handleClearFocus}
                      >
                        <X className="size-3.5" />
                      </Button>
                    </div>
                    {/* React escapes this; the 3D tooltip does not, which is why
                        the label fed to the canvas is escaped in graph-data.ts. */}
                    <p className="mt-2 max-h-40 overflow-y-auto text-sm leading-relaxed">
                      {focused.content}
                    </p>
                    <dl className="mt-3 space-y-1 text-[11px] text-muted-foreground">
                      <div className="flex justify-between gap-2">
                        <dt>{t("graph.connections")}</dt>
                        <dd className="font-mono tabular-nums">{focused.degree}</dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt>ID</dt>
                        <dd className="truncate font-mono" title={focused.id}>
                          {focused.id.slice(0, 12)}…
                        </dd>
                      </div>
                    </dl>
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          ) : (
            <div className="flex h-full items-center justify-center rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground">{t("graph.noNeurons")}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
