import { useState, useCallback } from "react"
import { useGraph } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { NetworkGraph } from "./NetworkGraph"

const LIMIT_OPTIONS = [100, 250, 500, 1000] as const

const LEGEND = [
  { type: "concept", color: "#6366f1" },
  { type: "entity", color: "#06b6d4" },
  { type: "time", color: "#f59e0b" },
  { type: "action", color: "#059669" },
  { type: "state", color: "#8b5cf6" },
  { type: "other", color: "#a8a29e" },
] as const

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
    <div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4 p-4">
      {/* Header row */}
      <div className="flex items-center justify-between shrink-0">
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

      {/* Graph — full remaining space */}
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="py-3 px-4 shrink-0 flex flex-row items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            Network Visualization
            {graph && (
              <span className="font-normal text-muted-foreground">
                {graph.neurons.length.toLocaleString()} nodes,{" "}
                {graph.synapses.length.toLocaleString()} edges
              </span>
            )}
          </CardTitle>
          {/* Inline legend */}
          <div className="flex items-center gap-3">
            {LEGEND.map(({ type, color }) => (
              <div key={type} className="flex items-center gap-1">
                <div
                  className="size-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-[10px] capitalize text-muted-foreground">{type}</span>
              </div>
            ))}
          </div>
        </CardHeader>
        <CardContent className="flex-1 p-2 min-h-0">
          {isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : graph && graph.neurons.length > 0 ? (
            <NetworkGraph
              data={graph}
              height="100%"
              onNodeClick={handleNodeClick}
            />
          ) : (
            <div className="flex h-full items-center justify-center rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground">
                No neurons found in this brain.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Node detail bar — bottom, only when selected */}
      {selectedNode && (
        <Card className="shrink-0">
          <CardContent className="flex items-start gap-4 p-3">
            <Badge variant="secondary" className="shrink-0 mt-0.5">
              {selectedNode.type}
            </Badge>
            <p className="flex-1 text-sm leading-relaxed">
              {selectedNode.content}
            </p>
            <p className="shrink-0 font-mono text-[10px] text-muted-foreground mt-0.5">
              {selectedNode.id.slice(0, 12)}...
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
