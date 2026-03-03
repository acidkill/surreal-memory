import { useState } from "react"
import { useStats, useBrains, useSwitchBrain, useDeleteBrain } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { Brain, Zap, Link2, Layers, Trash2 } from "lucide-react"
import { toast } from "sonner"

function KpiCard({
  label,
  value,
  icon: Icon,
  loading,
}: {
  label: string
  value: string | number
  icon: React.ElementType
  loading: boolean
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4 p-6">
        <div className="flex size-12 items-center justify-center rounded-lg bg-primary/10">
          <Icon className="size-6 text-primary" aria-hidden="true" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          {loading ? (
            <Skeleton className="mt-1 h-7 w-20" />
          ) : (
            <p className="font-mono text-2xl font-bold tracking-tight">
              {typeof value === "number" ? value.toLocaleString() : value}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function OverviewPage() {
  const { data: stats, isLoading: statsLoading } = useStats()
  const { data: brains, isLoading: brainsLoading } = useBrains()
  const switchBrain = useSwitchBrain()
  const deleteBrain = useDeleteBrain()
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null)

  const handleSwitchBrain = (brainName: string) => {
    switchBrain.mutate(brainName, {
      onSuccess: () => toast.success(`Switched to brain: ${brainName}`),
      onError: () => toast.error(`Failed to switch brain`),
    })
  }

  const handleDeleteBrain = () => {
    if (!deleteTarget) return
    deleteBrain.mutate(deleteTarget.id, {
      onSuccess: () => {
        toast.success(`Deleted brain: ${deleteTarget.name}`)
        setDeleteTarget(null)
      },
      onError: () => {
        toast.error(`Failed to delete brain`)
        setDeleteTarget(null)
      },
    })
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">Overview</h1>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          label="Neurons"
          value={stats?.total_neurons ?? 0}
          icon={Brain}
          loading={statsLoading}
        />
        <KpiCard
          label="Synapses"
          value={stats?.total_synapses ?? 0}
          icon={Link2}
          loading={statsLoading}
        />
        <KpiCard
          label="Fibers"
          value={stats?.total_fibers ?? 0}
          icon={Layers}
          loading={statsLoading}
        />
        <KpiCard
          label="Brains"
          value={stats?.total_brains ?? 0}
          icon={Zap}
          loading={statsLoading}
        />
      </div>

      {/* Brain List */}
      <Card>
        <CardHeader>
          <CardTitle>Brains</CardTitle>
        </CardHeader>
        <CardContent>
          {brainsLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : brains && brains.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 font-medium">Name</th>
                    <th className="pb-2 font-medium">Neurons</th>
                    <th className="pb-2 font-medium">Synapses</th>
                    <th className="pb-2 font-medium">Fibers</th>
                    <th className="pb-2 font-medium">Grade</th>
                    <th className="pb-2 font-medium">Status</th>
                    <th className="pb-2 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {brains.map((brain) => (
                    <tr
                      key={brain.id}
                      className={`border-b border-border/50 last:border-0 transition-colors ${
                        !brain.is_active
                          ? "cursor-pointer hover:bg-accent/50"
                          : ""
                      }`}
                      onClick={() => {
                        if (!brain.is_active) handleSwitchBrain(brain.name)
                      }}
                      title={
                        brain.is_active
                          ? "Current active brain"
                          : `Click to switch to ${brain.name}`
                      }
                    >
                      <td className="py-3 font-mono font-medium">
                        {brain.name}
                      </td>
                      <td className="py-3 font-mono">
                        {brain.neuron_count.toLocaleString()}
                      </td>
                      <td className="py-3 font-mono">
                        {brain.synapse_count.toLocaleString()}
                      </td>
                      <td className="py-3 font-mono">
                        {brain.fiber_count.toLocaleString()}
                      </td>
                      <td className="py-3">
                        <Badge
                          variant={
                            brain.grade === "A" || brain.grade === "A+"
                              ? "success"
                              : brain.grade === "B" || brain.grade === "B+"
                                ? "secondary"
                                : "warning"
                          }
                        >
                          {brain.grade}
                        </Badge>
                      </td>
                      <td className="py-3">
                        {brain.is_active ? (
                          <Badge variant="default">Active</Badge>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td className="py-3">
                        {!brain.is_active && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-8 text-muted-foreground hover:text-destructive"
                            onClick={(e) => {
                              e.stopPropagation()
                              setDeleteTarget({ id: brain.id, name: brain.name })
                            }}
                            aria-label={`Delete brain ${brain.name}`}
                          >
                            <Trash2 className="size-4" />
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No brains found.</p>
          )}
        </CardContent>
      </Card>

      {/* Delete confirmation dialog */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Brain"
        description={`Delete brain "${deleteTarget?.name}"? This will remove all neurons, synapses, and fibers permanently. This cannot be undone.`}
        confirmLabel="Delete"
        variant="destructive"
        onConfirm={handleDeleteBrain}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
