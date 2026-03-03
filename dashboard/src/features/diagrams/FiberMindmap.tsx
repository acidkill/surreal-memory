import { useMemo, useState } from "react"
import type { FiberDiagramResponse } from "@/api/types"

const TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1",
  entity: "#06b6d4",
  time: "#f59e0b",
  action: "#059669",
  state: "#8b5cf6",
  other: "#a8a29e",
  relation: "#ec4899",
  attribute: "#14b8a6",
}

interface TreeNode {
  id: string
  label: string
  fullContent: string
  type: string
  children: TreeNode[]
  isGroup?: boolean
  count?: number
}

function buildTree(diagram: FiberDiagramResponse): TreeNode {
  // Group neurons by type
  const groups = new Map<string, typeof diagram.neurons>()
  for (const neuron of diagram.neurons) {
    const group = groups.get(neuron.type) ?? []
    group.push(neuron)
    groups.set(neuron.type, group)
  }

  const children: TreeNode[] = []
  for (const [type, neurons] of groups) {
    children.push({
      id: `group-${type}`,
      label: `${type} (${neurons.length})`,
      fullContent: "",
      type,
      isGroup: true,
      count: neurons.length,
      children: neurons.map((n) => ({
        id: n.id,
        label: n.content.length > 60 ? n.content.slice(0, 60) + "..." : n.content,
        fullContent: n.content,
        type: n.type,
        children: [],
      })),
    })
  }

  return {
    id: diagram.fiber_id,
    label: `Fiber`,
    fullContent: diagram.fiber_id,
    type: "root",
    children,
  }
}

interface MindmapNodeProps {
  node: TreeNode
  depth: number
  onSelect: (node: TreeNode) => void
  selectedId: string | null
}

function MindmapNode({ node, depth, onSelect, selectedId }: MindmapNodeProps) {
  const [expanded, setExpanded] = useState(depth < 2)
  const hasChildren = node.children.length > 0
  const isSelected = selectedId === node.id
  const color = TYPE_COLORS[node.type] ?? TYPE_COLORS.other

  return (
    <div className="relative">
      {/* Connector line */}
      {depth > 0 && (
        <div
          className="absolute -left-4 top-3 h-px w-4"
          style={{ backgroundColor: `${color}40` }}
        />
      )}

      {/* Node */}
      <div
        className={`group flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 transition-colors ${
          isSelected
            ? "bg-primary/10 ring-1 ring-primary/30"
            : "hover:bg-accent"
        }`}
        onClick={(e) => {
          e.stopPropagation()
          if (hasChildren) setExpanded((prev) => !prev)
          if (!node.isGroup) onSelect(node)
        }}
      >
        {/* Expand/collapse indicator */}
        {hasChildren ? (
          <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center rounded text-xs font-bold text-muted-foreground">
            {expanded ? "−" : "+"}
          </span>
        ) : (
          <span className="mt-0.5 size-4 shrink-0" />
        )}

        {/* Color dot */}
        <div
          className="mt-1.5 size-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: color }}
        />

        {/* Label */}
        <span
          className={`text-sm leading-snug ${
            node.isGroup ? "font-semibold" : "font-normal"
          } ${depth === 0 ? "font-display text-base font-bold" : ""}`}
        >
          {node.label}
          {node.count !== undefined && (
            <span className="ml-1 font-mono text-xs text-muted-foreground">
              {node.count}
            </span>
          )}
        </span>
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div className="ml-6 border-l border-border/50 pl-2">
          {node.children.map((child) => (
            <MindmapNode
              key={child.id}
              node={child}
              depth={depth + 1}
              onSelect={onSelect}
              selectedId={selectedId}
            />
          ))}
        </div>
      )}
    </div>
  )
}

interface FiberMindmapProps {
  diagram: FiberDiagramResponse
  onSelectNeuron?: (id: string, content: string, type: string) => void
}

export function FiberMindmap({ diagram, onSelectNeuron }: FiberMindmapProps) {
  const tree = useMemo(() => buildTree(diagram), [diagram])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const handleSelect = (node: TreeNode) => {
    setSelectedId(node.id)
    onSelectNeuron?.(node.id, node.fullContent, node.type)
  }

  return (
    <div className="max-h-[500px] overflow-y-auto pr-2">
      <MindmapNode
        node={tree}
        depth={0}
        onSelect={handleSelect}
        selectedId={selectedId}
      />
    </div>
  )
}
