/**
 * Pure transforms feeding the 3D graph. No React, no three.js, no DOM — so the
 * parts that are easy to get dangerously wrong (HTML escaping, the particle
 * budget) are unit-testable without a browser or a WebGL context.
 */

import type { GraphResponse } from "@/api/types"

/** Above this node count, per-link particles cost more frame time than they are worth. */
export const PARTICLE_NODE_THRESHOLD = 1000

/** Hard ceiling on particles across the whole graph, regardless of link count. */
export const PARTICLE_GLOBAL_CAP = 600

/** Only the strongest links get particles, so the budget goes where it reads. */
export const PARTICLE_WEIGHT_QUANTILE = 0.8

export interface Graph3DNode {
  id: string
  /** Escaped, display-ready label. NEVER the raw content — see escapeHtml. */
  label: string
  /** Raw content, for the React detail panel (React escapes its own text). */
  content: string
  type: string
  /** Node volume, driven by degree. */
  val: number
  degree: number
}

export interface Graph3DLink {
  source: string
  target: string
  weight: number
  /** Particle count for this link; 0 means none. */
  particles: number
}

export interface Graph3DData {
  nodes: Graph3DNode[]
  links: Graph3DLink[]
  /** node id -> the set of directly connected node ids (focus mode). */
  neighbors: Map<string, Set<string>>
}

/**
 * Escape HTML-special characters.
 *
 * This is a security control, not formatting. 3d-force-graph renders
 * `nodeLabel` as raw HTML inside its tooltip element, and a node's label is
 * derived from neuron content — data the user (or anything with write access to
 * the brain, including an MCP client) supplies. Without this, storing a memory
 * containing `<img src=x onerror=...>` lands stored XSS in the dashboard of
 * whoever later hovers that node.
 */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
}

const LABEL_MAX = 60

function truncate(value: string, max = LABEL_MAX): string {
  return value.length > max ? `${value.slice(0, max)}…` : value
}

/**
 * Particle count per link, subject to a global budget.
 *
 * Particles are per-link GPU objects animated every frame. Handing one to every
 * link scales the cost with the edge count, which is precisely backwards: the
 * big graphs that most need a sense of flow are the ones that can least afford
 * it. So: nothing above the node threshold (unless the caller is in focus mode
 * and has already narrowed the set), only links at or above the weight
 * quantile, and never more than PARTICLE_GLOBAL_CAP in total.
 */
export function assignParticleBudget(
  links: Graph3DLink[],
  nodeCount: number,
  { force = false }: { force?: boolean } = {},
): Graph3DLink[] {
  if (!force && nodeCount > PARTICLE_NODE_THRESHOLD) {
    return links.map((l) => ({ ...l, particles: 0 }))
  }
  if (links.length === 0) return links

  const sorted = [...links].map((l) => l.weight).sort((a, b) => a - b)
  const cutoff = sorted[Math.floor(sorted.length * PARTICLE_WEIGHT_QUANTILE)] ?? 0

  // Strongest first, so a tight budget is spent on the most meaningful links
  // rather than on whichever happened to be earliest in the payload.
  const order = links
    .map((link, index) => ({ index, weight: link.weight }))
    .sort((a, b) => b.weight - a.weight)

  const granted = new Set<number>()
  let spent = 0
  for (const { index, weight } of order) {
    if (spent >= PARTICLE_GLOBAL_CAP) break
    if (weight < cutoff) continue
    granted.add(index)
    spent += 1
  }

  return links.map((link, index) => ({ ...link, particles: granted.has(index) ? 1 : 0 }))
}

/**
 * Turn the API's payload into what 3d-force-graph wants.
 *
 * Drops synapses whose endpoints are not both present. The endpoint already
 * guarantees this (it picks the densest node set first, then keeps only edges
 * with both ends inside it), so this is belt-and-braces against a server-side
 * change rather than a fix for anything observed — an edge referencing a
 * missing node makes the force layout throw rather than degrade.
 */
export function toGraph3D(data: GraphResponse): Graph3DData {
  const nodeIds = new Set(data.neurons.map((n) => n.id))

  const degree = new Map<string, number>()
  const neighbors = new Map<string, Set<string>>()
  const seenEdge = new Set<string>()
  const links: Graph3DLink[] = []

  for (const synapse of data.synapses) {
    const { source_id: source, target_id: target } = synapse
    if (!nodeIds.has(source) || !nodeIds.has(target)) continue
    if (source === target) continue

    // Undirected dedup: a<->b and b<->a are one line on screen, and feeding
    // both doubles the layout forces between that pair. JSON.stringify of the
    // sorted pair (not a "|"-joined string) so a node id that itself contains
    // the delimiter can never collide two distinct pairs onto the same key —
    // ids are uuid4() today, but nothing guarantees that stays true.
    const key = JSON.stringify(source < target ? [source, target] : [target, source])
    if (seenEdge.has(key)) continue
    seenEdge.add(key)

    links.push({ source, target, weight: synapse.weight, particles: 0 })

    degree.set(source, (degree.get(source) ?? 0) + 1)
    degree.set(target, (degree.get(target) ?? 0) + 1)

    if (!neighbors.has(source)) neighbors.set(source, new Set())
    if (!neighbors.has(target)) neighbors.set(target, new Set())
    neighbors.get(source)!.add(target)
    neighbors.get(target)!.add(source)
  }

  const nodes: Graph3DNode[] = data.neurons.map((neuron) => {
    const d = degree.get(neuron.id) ?? 0
    // The API contract says `content: string`, but nothing at this boundary
    // validates the response — a coerced fallback keeps a malformed payload
    // from crashing escapeHtml's .replace() calls instead of rendering.
    const rawContent = typeof neuron.content === "string" ? neuron.content : String(neuron.content ?? "")
    return {
      id: neuron.id,
      label: escapeHtml(truncate(rawContent || neuron.id)),
      content: rawContent,
      type: neuron.type,
      // sqrt so a 400-connection hub is visibly bigger than a 4-connection node
      // without being 100x its volume and swallowing the view.
      val: 1 + Math.sqrt(d),
      degree: d,
    }
  })

  return {
    nodes,
    links: assignParticleBudget(links, nodes.length),
    neighbors,
  }
}

/** Direct neighbourhood of a node, including the node itself. */
export function focusSet(neighbors: Map<string, Set<string>>, nodeId: string): Set<string> {
  const set = new Set<string>([nodeId])
  for (const id of neighbors.get(nodeId) ?? []) set.add(id)
  return set
}
