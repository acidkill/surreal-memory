/* ------------------------------------------------------------------ */
/*  A* pathfinding for corridor generation between rooms              */
/* ------------------------------------------------------------------ */

import type { Position } from "../engine/types"
import { TileType } from "../engine/types"

interface AStarNode {
  x: number
  y: number
  g: number
  h: number
  f: number
  parent: AStarNode | null
}

function heuristic(a: Position, b: Position): number {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y)
}

function key(x: number, y: number): string {
  return `${x},${y}`
}

const NEIGHBORS: readonly Position[] = [
  { x: 0, y: -1 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
  { x: 1, y: 0 },
]

/**
 * A* pathfinding on a tile grid.
 * Returns path from start to end, avoiding VOID tiles.
 * Prefers carving through WALL (cost 1) over existing FLOOR (cost 0.5).
 */
export function findPath(
  tiles: TileType[][],
  start: Position,
  end: Position,
  width: number,
  height: number,
): Position[] {
  const open = new Map<string, AStarNode>()
  const closed = new Set<string>()

  const startNode: AStarNode = {
    x: start.x,
    y: start.y,
    g: 0,
    h: heuristic(start, end),
    f: heuristic(start, end),
    parent: null,
  }
  open.set(key(start.x, start.y), startNode)

  while (open.size > 0) {
    // Find node with lowest f
    let current: AStarNode | null = null
    for (const node of open.values()) {
      if (current === null || node.f < current.f) {
        current = node
      }
    }
    if (current === null) break

    // Reached destination
    if (current.x === end.x && current.y === end.y) {
      const path: Position[] = []
      let node: AStarNode | null = current
      while (node !== null) {
        path.push({ x: node.x, y: node.y })
        node = node.parent
      }
      return path.reverse()
    }

    const currentKey = key(current.x, current.y)
    open.delete(currentKey)
    closed.add(currentKey)

    for (const dir of NEIGHBORS) {
      const nx = current.x + dir.x
      const ny = current.y + dir.y
      const nk = key(nx, ny)

      if (nx < 1 || nx >= width - 1 || ny < 1 || ny >= height - 1) continue
      if (closed.has(nk)) continue

      // Cost: prefer existing floor, allow carving walls
      const tile = tiles[ny]![nx]!
      if (tile === TileType.VOID) continue // can't path through void
      const moveCost = tile === TileType.FLOOR || tile === TileType.CORRIDOR ? 0.5 : 1

      const g = current.g + moveCost
      const existing = open.get(nk)
      if (existing !== undefined && g >= existing.g) continue

      const node: AStarNode = {
        x: nx,
        y: ny,
        g,
        h: heuristic({ x: nx, y: ny }, end),
        f: g + heuristic({ x: nx, y: ny }, end),
        parent: current,
      }
      open.set(nk, node)
    }
  }

  // No path found — fallback to straight line
  return straightLine(start, end)
}

/** Simple L-shaped fallback path */
function straightLine(start: Position, end: Position): Position[] {
  const path: Position[] = []
  let { x, y } = start

  // Horizontal first
  while (x !== end.x) {
    path.push({ x, y })
    x += x < end.x ? 1 : -1
  }
  // Then vertical
  while (y !== end.y) {
    path.push({ x, y })
    y += y < end.y ? 1 : -1
  }
  path.push({ x: end.x, y: end.y })
  return path
}
