import { useState, useCallback } from "react"
import { useGraph } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { NetworkGraph } from "./NetworkGraph"

const LIMIT_OPTIONS = [100, 250, 500, 1000] as const

interface SelectedNode {
  id: string
  content: string
  type: string
}

export default function GraphPage() {
  const [limit, setLimit] = useState<number>(250)
  const { data: graph, isLoading } = useGraph(limit)
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null)

  const handleNodeClick = useCallback((id: string, content: string, type: string) => {
    setSelectedNode({ id, content, type })
  }, [])

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold">Neural Graph</h1>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Nodes:</span>
          {LIMIT_OPTIONS.map((opt) => (
            <Button
              key={opt}
              variant={limit === opt ? "default" : "outline"}
              size="sm"
              onClick={() => setLimit(opt)}
            >
              {opt}
            </Button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        {/* Graph area */}
        <Card className="lg:col-span-3">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              Network Visualization
              {graph && (
                <span className="text-sm font-normal text-muted-foreground">
                  {graph.neurons.length.toLocaleString()} nodes,{" "}
                  {graph.synapses.length.toLocaleString()} edges
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-[500px] w-full" />
            ) : graph && graph.neurons.length > 0 ? (
              <NetworkGraph
                data={graph}
                height="500px"
                onNodeClick={handleNodeClick}
              />
            ) : (
              <div className="flex h-[500px] items-center justify-center rounded-lg border border-border bg-muted/30">
                <p className="text-sm text-muted-foreground">
                  No neurons found in this brain.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Details panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Node Details</CardTitle>
          </CardHeader>
          <CardContent>
            {selectedNode ? (
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-muted-foreground">Type</p>
                  <Badge variant="secondary" className="mt-1">
                    {selectedNode.type}
                  </Badge>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Content</p>
                  <p className="mt-1 text-sm leading-relaxed">
                    {selectedNode.content}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">ID</p>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    {selectedNode.id.slice(0, 12)}...
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Click a node in the graph to see its details.
              </p>
            )}
          </CardContent>

          {/* Legend */}
          <CardContent className="border-t border-border pt-4">
            <p className="mb-2 text-xs font-medium text-muted-foreground">Legend</p>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { type: "concept", color: "#6366f1" },
                { type: "entity", color: "#06b6d4" },
                { type: "time", color: "#f59e0b" },
                { type: "action", color: "#059669" },
                { type: "state", color: "#8b5cf6" },
                { type: "other", color: "#a8a29e" },
              ].map(({ type, color }) => (
                <div key={type} className="flex items-center gap-1.5">
                  <div
                    className="size-2.5 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs capitalize">{type}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
