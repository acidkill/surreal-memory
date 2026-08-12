/**
 * Neuron type -> colour. Single source of truth for the graph.
 *
 * These used to exist twice: `TYPE_COLORS` inside NetworkGraph.tsx (used by the
 * renderer) and `LEGEND_COLORS` inside GraphPage.tsx (used by the legend). Two
 * copies of a colour map means the legend can quietly stop describing the
 * picture it sits next to.
 */

export const TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1",
  entity: "#06b6d4",
  time: "#f59e0b",
  action: "#059669",
  state: "#8b5cf6",
  other: "#a8a29e",
  relation: "#ec4899",
  attribute: "#14b8a6",
}

/** The subset shown in the legend, in display order. */
export const LEGEND_KEYS = ["concept", "entity", "time", "action", "state", "other"] as const

export function colorForType(type: string): string {
  return TYPE_COLORS[type] ?? TYPE_COLORS.other
}

/**
 * The same colour, dimmed, for nodes and links outside the focused
 * neighbourhood. three-forcegraph honours the alpha channel, and it caches
 * materials per colour string, so returning a stable `rgba(...)` per type keeps
 * the number of materials bounded rather than growing with the node count.
 */
export function dimmedColorForType(type: string, alpha = 0.12): string {
  const hex = colorForType(type)
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
