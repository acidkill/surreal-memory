import { useMemo, useCallback } from "react"
import {
  ReactFlow,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import Dagre from "@dagrejs/dagre"
import type { FiberDiagramResponse } from "@/api/types"

/* ------------------------------------------------------------------ */
/*  Color palette (same as GraphPage)                                  */
/* ------------------------------------------------------------------ */

const TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1",
  entity: "#06b6d4",
  time: "#f59e0b",
  action: "#059669",
  state: "#8b5cf6",
  other: "#a8a29e",
  relation: "#ec4899",
  attribute: "#14b8a6",
  root: "#f97316",
  group: "#64748b",
}

const TYPE_BG: Record<string, string> = {
  concept: "#6366f115",
  entity: "#06b6d415",
  time: "#f59e0b15",
  action: "#05966915",
  state: "#8b5cf615",
  other: "#a8a29e15",
  relation: "#ec489915",
  attribute: "#14b8a615",
  root: "#f9731620",
  group: "#64748b15",
}

/* ------------------------------------------------------------------ */
/*  Custom node components                                             */
/* ------------------------------------------------------------------ */

interface MindmapNodeData {
  label: string
  fullContent: string
  neuronType: string
  isGroup: boolean
  count?: number
  [key: string]: unknown
}

type MindmapNode = Node<MindmapNodeData>

function RootNode({ data }: NodeProps<MindmapNode>) {
  const color = TYPE_COLORS.root
  return (
    <div
      className="rounded-xl px-5 py-3 text-center shadow-md"
      style={{
        background: `linear-gradient(135deg, ${color}20, ${color}40)`,
        border: `2px solid ${color}`,
        minWidth: 100,
      }}
    >
      <p className="font-display text-sm font-bold" style={{ color }}>
        {data.label}
      </p>
      <Handle type="source" position={Position.Right} className="!bg-transparent !border-0" />
    </div>
  )
}

function GroupNode({ data }: NodeProps<MindmapNode>) {
  const color = TYPE_COLORS[data.neuronType] ?? TYPE_COLORS.group
  return (
    <div
      className="rounded-lg px-4 py-2 shadow-sm"
      style={{
        background: TYPE_BG[data.neuronType] ?? TYPE_BG.group,
        border: `2px solid ${color}80`,
        minWidth: 100,
      }}
    >
      <Handle type="target" position={Position.Left} className="!bg-transparent !border-0" />
      <div className="flex items-center gap-2">
        <div className="size-3 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm font-semibold">{data.neuronType}</span>
        {data.count !== undefined && (
          <span
            className="rounded-full px-1.5 py-0.5 font-mono text-[10px] font-bold"
            style={{ backgroundColor: `${color}30`, color }}
          >
            {data.count}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-transparent !border-0" />
    </div>
  )
}

function LeafNode({ data }: NodeProps<MindmapNode>) {
  const color = TYPE_COLORS[data.neuronType] ?? TYPE_COLORS.other
  return (
    <div
      className="cursor-pointer rounded-lg px-3 py-1.5 shadow-sm transition-shadow hover:shadow-md"
      style={{
        background: TYPE_BG[data.neuronType] ?? TYPE_BG.other,
        border: `1.5px solid ${color}50`,
        maxWidth: 260,
        minWidth: 80,
      }}
    >
      <Handle type="target" position={Position.Left} className="!bg-transparent !border-0" />
      <p className="text-xs leading-snug" style={{ color: "var(--color-foreground)" }}>
        {data.label}
      </p>
    </div>
  )
}

const nodeTypes = {
  root: RootNode,
  group: GroupNode,
  leaf: LeafNode,
}

/* ------------------------------------------------------------------ */
/*  Dagre layout                                                       */
/* ------------------------------------------------------------------ */

function layoutTree(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  g.setGraph({
    rankdir: "LR",
    nodesep: 16,
    ranksep: 80,
    edgesep: 10,
  })

  for (const node of nodes) {
    const width = node.type === "root" ? 160 : node.type === "group" ? 140 : 200
    const height = node.type === "root" ? 50 : node.type === "group" ? 40 : 36
    g.setNode(node.id, { width, height })
  }

  for (const edge of edges) {
    g.setEdge(edge.source, edge.target)
  }

  Dagre.layout(g)

  const layoutedNodes = nodes.map((node) => {
    const pos = g.node(node.id)
    const width = node.type === "root" ? 160 : node.type === "group" ? 140 : 200
    const height = node.type === "root" ? 50 : node.type === "group" ? 40 : 36
    return {
      ...node,
      position: {
        x: pos.x - width / 2,
        y: pos.y - height / 2,
      },
    }
  })

  return { nodes: layoutedNodes, edges }
}

/* ------------------------------------------------------------------ */
/*  Build ReactFlow nodes/edges from FiberDiagramResponse              */
/* ------------------------------------------------------------------ */

function buildFlowData(diagram: FiberDiagramResponse): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  // Root node
  const rootId = `root-${diagram.fiber_id}`
  nodes.push({
    id: rootId,
    type: "root",
    position: { x: 0, y: 0 },
    data: {
      label: `Fiber`,
      fullContent: diagram.fiber_id,
      neuronType: "root",
      isGroup: false,
    },
  })

  // Group neurons by type
  const groups = new Map<string, typeof diagram.neurons>()
  for (const neuron of diagram.neurons) {
    const group = groups.get(neuron.type) ?? []
    group.push(neuron)
    groups.set(neuron.type, group)
  }

  // Create group + leaf nodes
  for (const [type, neurons] of groups) {
    const groupId = `group-${type}`
    nodes.push({
      id: groupId,
      type: "group",
      position: { x: 0, y: 0 },
      data: {
        label: type,
        fullContent: "",
        neuronType: type,
        isGroup: true,
        count: neurons.length,
      },
    })
    edges.push({
      id: `e-root-${type}`,
      source: rootId,
      target: groupId,
      style: { stroke: TYPE_COLORS[type] ?? TYPE_COLORS.other, strokeWidth: 2, opacity: 0.5 },
      type: "smoothstep",
    })

    for (const neuron of neurons) {
      const label = neuron.content.length > 80
        ? neuron.content.slice(0, 80) + "..."
        : neuron.content
      nodes.push({
        id: neuron.id,
        type: "leaf",
        position: { x: 0, y: 0 },
        data: {
          label,
          fullContent: neuron.content,
          neuronType: neuron.type,
          isGroup: false,
        },
      })
      edges.push({
        id: `e-${groupId}-${neuron.id}`,
        source: groupId,
        target: neuron.id,
        style: {
          stroke: TYPE_COLORS[type] ?? TYPE_COLORS.other,
          strokeWidth: 1.5,
          opacity: 0.35,
        },
        type: "smoothstep",
      })
    }
  }

  // Also add synapse connections (as dashed lines between leaf nodes)
  const neuronIds = new Set(diagram.neurons.map((n) => n.id))
  for (const syn of diagram.synapses) {
    if (neuronIds.has(syn.source_id) && neuronIds.has(syn.target_id)) {
      edges.push({
        id: `syn-${syn.id}`,
        source: syn.source_id,
        target: syn.target_id,
        style: { stroke: "#94a3b8", strokeWidth: 1, strokeDasharray: "4 2", opacity: 0.3 },
        animated: true,
      })
    }
  }

  return layoutTree(nodes, edges)
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

interface FiberMindmapProps {
  diagram: FiberDiagramResponse
  onSelectNeuron?: (id: string, content: string, type: string) => void
}

function FiberMindmapInner({ diagram, onSelectNeuron }: FiberMindmapProps) {
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => buildFlowData(diagram),
    [diagram],
  )

  const [nodes, , onNodesChange] = useNodesState(layoutedNodes)
  const [edges, , onEdgesChange] = useEdgesState(layoutedEdges)

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const data = node.data as MindmapNodeData
      if (!data.isGroup && node.type === "leaf") {
        onSelectNeuron?.(node.id, data.fullContent, data.neuronType)
      }
    },
    [onSelectNeuron],
  )

  return (
    <div className="h-[calc(100vh-14rem)] min-h-[500px] w-full rounded-lg border border-border bg-background">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="var(--color-border)" gap={20} size={1} />
        <Controls
          showInteractive={false}
          className="!bg-card !border-border !shadow-sm [&>button]:!bg-card [&>button]:!border-border [&>button]:!fill-foreground"
        />
        <MiniMap
          nodeColor={(n) => TYPE_COLORS[(n.data as MindmapNodeData)?.neuronType] ?? "#a8a29e"}
          maskColor="rgba(0,0,0,0.15)"
          className="!bg-card !border-border"
        />
      </ReactFlow>
    </div>
  )
}

export function FiberMindmap(props: FiberMindmapProps) {
  return (
    <ReactFlowProvider>
      <FiberMindmapInner {...props} />
    </ReactFlowProvider>
  )
}
