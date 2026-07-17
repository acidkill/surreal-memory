import { useState } from "react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import { Trash } from "@phosphor-icons/react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { apiErrorMessage } from "@/api/client"
import {
  useDeletePatternsByModel,
  useDeleteReasoningPattern,
  useReasoningPatterns,
} from "@/api/hooks/useReasoning"
import type { PatternSummary } from "@/api/types"

interface Props {
  detectedModels: string[]
  categories: string[]
}

const PAGE_SIZE = 20

export function PatternsTable({ detectedModels, categories }: Props) {
  const { t } = useTranslation()
  const [model, setModel] = useState("")
  const [category, setCategory] = useState("")
  const [offset, setOffset] = useState(0)
  const [toDelete, setToDelete] = useState<PatternSummary | null>(null)
  const [bulkModel, setBulkModel] = useState<string | null>(null)

  const { data, isLoading, isError, refetch } = useReasoningPatterns(
    model,
    category,
    PAGE_SIZE,
    offset,
  )
  const deletePattern = useDeleteReasoningPattern()
  const deleteByModel = useDeletePatternsByModel()

  const total = data?.total ?? 0
  const patterns = data?.patterns ?? []

  // If the data shrank below the current page (e.g. after a delete), step back so
  // the user isn't stranded on an empty page with hidden pagination controls.
  if (data && offset > 0 && offset >= total) {
    setOffset(Math.max(0, total - PAGE_SIZE))
  }

  const resetPaging = () => setOffset(0)

  const confirmDelete = () => {
    if (!toDelete) return
    const id = toDelete.id
    setToDelete(null)
    deletePattern.mutate(id, {
      onSuccess: () => toast.success(t("reasoning.patternDeleted")),
      onError: () => toast.error(t("reasoning.patternDeleteFailed")),
    })
  }

  const confirmBulkDelete = () => {
    if (!bulkModel) return
    const m = bulkModel
    setBulkModel(null)
    deleteByModel.mutate(m, {
      onSuccess: (res) =>
        toast.success(t("reasoning.patternsDeletedCount", { count: res.deleted })),
      onError: (err) => toast.error(apiErrorMessage(err, t("reasoning.patternDeleteFailed"))),
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("reasoning.patternsTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <div className="flex flex-wrap gap-2">
          <select
            value={model}
            onChange={(e) => {
              setModel(e.target.value)
              resetPaging()
            }}
            className="rounded-md border border-border bg-background px-2 py-1.5 text-xs"
            aria-label={t("reasoning.filterModel")}
          >
            <option value="">{t("reasoning.allModels")}</option>
            {detectedModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <select
            value={category}
            onChange={(e) => {
              setCategory(e.target.value)
              resetPaging()
            }}
            className="rounded-md border border-border bg-background px-2 py-1.5 text-xs"
            aria-label={t("reasoning.filterCategory")}
          >
            <option value="">{t("reasoning.allCategories")}</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          {model && (
            <Button
              variant="outline"
              size="sm"
              className="text-destructive"
              onClick={() => setBulkModel(model)}
              disabled={deleteByModel.isPending}
            >
              <Trash className="size-4" /> {t("reasoning.deleteAllForModel")}
            </Button>
          )}
        </div>

        {isLoading ? (
          <Skeleton className="h-32 w-full" />
        ) : isError ? (
          <div className="space-y-2">
            <p className="text-sm text-destructive">{t("reasoning.patternsError")}</p>
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              {t("common.retry")}
            </Button>
          </div>
        ) : patterns.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("reasoning.noPatterns")}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">{t("reasoning.colTitle")}</th>
                  <th className="py-2 pr-3 font-medium">{t("reasoning.colModel")}</th>
                  <th className="py-2 pr-3 font-medium">{t("reasoning.colCategory")}</th>
                  <th className="py-2 pr-3 text-right font-medium">
                    {t("reasoning.colConfidence")}
                  </th>
                  <th className="py-2 pr-3 text-right font-medium">
                    {t("reasoning.colFrequency")}
                  </th>
                  <th className="py-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {patterns.map((p) => (
                  <tr key={p.id} className="border-b border-border/50">
                    <td className="py-2 pr-3">{p.title}</td>
                    <td className="py-2 pr-3 font-mono text-xs">{p.source_model}</td>
                    <td className="py-2 pr-3">
                      <Badge variant="secondary">{p.category}</Badge>
                    </td>
                    <td className="py-2 pr-3 text-right font-mono">
                      {Math.round(p.confidence * 100)}%
                    </td>
                    <td className="py-2 pr-3 text-right font-mono">{p.frequency}</td>
                    <td className="py-2 text-right">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setToDelete(p)}
                        aria-label={t("common.delete")}
                      >
                        <Trash className="size-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {total > PAGE_SIZE && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} / {total}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                {t("common.prev")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                {t("common.next")}
              </Button>
            </div>
          </div>
        )}
      </CardContent>

      <ConfirmDialog
        open={toDelete !== null}
        title={t("reasoning.deletePatternTitle")}
        description={t("reasoning.deletePatternDesc", { title: toDelete?.title ?? "" })}
        variant="destructive"
        confirmLabel={t("common.delete")}
        onConfirm={confirmDelete}
        onCancel={() => setToDelete(null)}
      />

      <ConfirmDialog
        open={bulkModel !== null}
        title={t("reasoning.deleteAllTitle")}
        description={t("reasoning.deleteAllDesc", { model: bulkModel ?? "" })}
        variant="destructive"
        confirmLabel={t("common.delete")}
        onConfirm={confirmBulkDelete}
        onCancel={() => setBulkModel(null)}
      />
    </Card>
  )
}
