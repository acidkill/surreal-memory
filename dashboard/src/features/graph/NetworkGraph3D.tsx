import { useCallback, useEffect, useRef } from "react"
import ForceGraph3D, {
  type ForceGraph3DInstance,
  type LinkObject,
  type NodeObject,
} from "3d-force-graph"

import type { GraphResponse } from "@/api/types"
import { colorForType, dimmedColorForType } from "./graph-colors"
import { focusSet, toGraph3D, type Graph3DData, type Graph3DNode } from "./graph-data"

/** Above this node count, draw cheaper spheres. */
const LOW_DETAIL_NODE_THRESHOLD = 1000
const NODE_RESOLUTION_HIGH = 12
const NODE_RESOLUTION_LOW = 6

const CAMERA_TRANSITION_MS = 800
const FOCUS_DISTANCE = 90

/**
 * The library's simulation adds its own mutable fields (x/y/z, index, and it
 * replaces link source/target with node references once the force layout runs),
 * so its own types are intersected with ours rather than replacing them.
 */
type GNode = NodeObject & Graph3DNode
type GLink = LinkObject<GNode> & { weight: number; particles: number }
type GraphInstance = ForceGraph3DInstance<GNode, GLink>

/**
 * The package's default export is the NON-generic constructor
 * (`declare const ForceGraph3D: IForceGraph3D`), and `IForceGraph3D` is not
 * exported, so there is no way to parameterise it without a cast. One cast
 * here buys fully typed accessors everywhere below.
 */
const ForceGraph3DCtor = ForceGraph3D as unknown as new (element: HTMLElement) => GraphInstance

/** The layout replaces string endpoints with node objects once it has run. */
function endpointId(endpoint: GLink["source"]): string {
  if (typeof endpoint === "string") return endpoint
  if (typeof endpoint === "number") return String(endpoint)
  return String(endpoint?.id ?? "")
}

export interface NetworkGraph3DProps {
  data: GraphResponse
  /** Fires with the clicked node, or null when focus is cleared. */
  onFocusChange?: (node: Graph3DNode | null) => void
  /** Bump to clear focus from outside (the detail panel's close button). */
  clearFocusSignal?: number
}

/**
 * 3D force graph over `3d-force-graph` (three.js + d3-force-3d).
 *
 * Lifecycle is the whole difficulty here. The instance owns a WebGL context and
 * browsers hard-cap how many may be live (~16), so creating one per data change
 * would kill the graph after a handful of slider moves. The instance is created
 * exactly ONCE per mount; data arrives through a separate effect. Teardown calls
 * the library's own `_destructor()` and disconnects the ResizeObserver, which is
 * also what makes React StrictMode's deliberate double mount/unmount safe.
 */
export function NetworkGraph3D({ data, onFocusChange, clearFocusSignal }: NetworkGraph3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<GraphInstance | null>(null)

  // Read by the colour accessors on every frame. Refs, not state: changing the
  // focus must repaint the existing scene, never re-run the effect that owns
  // the WebGL context.
  const focusedRef = useRef<Set<string> | null>(null)
  const dataRef = useRef<Graph3DData | null>(null)
  const onFocusChangeRef = useRef(onFocusChange)

  // Assigned in an effect, not during render: a ref write during render is a
  // side effect on a value React may discard.
  useEffect(() => {
    onFocusChangeRef.current = onFocusChange
  }, [onFocusChange])

  const applyFocus = useCallback((next: Set<string> | null) => {
    focusedRef.current = next
    graphRef.current?.refresh()
  }, [])

  // ---- init once per mount -------------------------------------------------
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const graph = new ForceGraph3DCtor(container)
      .backgroundColor("rgba(0,0,0,0)")
      .showNavInfo(false)
      .nodeVal((node) => node.val)
      // Rendered as raw HTML by the library's tooltip — the label is escaped in
      // graph-data.ts, which is a security control, not formatting.
      .nodeLabel((node) => node.label)
      .nodeColor((node) => {
        const focused = focusedRef.current
        if (!focused || focused.has(node.id)) return colorForType(node.type)
        return dimmedColorForType(node.type)
      })
      // Cheap THREE.Line segments; width > 0 would switch to tube geometry.
      .linkWidth(0)
      .linkColor((link) => {
        const focused = focusedRef.current
        if (!focused) return "rgba(128, 128, 128, 0.35)"
        const source = endpointId(link.source)
        const target = endpointId(link.target)
        return focused.has(source) && focused.has(target)
          ? "rgba(148, 163, 184, 0.85)"
          : "rgba(128, 128, 128, 0.06)"
      })
      .linkDirectionalParticles((link) => link.particles)
      .linkDirectionalParticleSpeed((link) => 0.002 + link.weight * 0.006)
      .linkDirectionalParticleWidth(1.6)
      .warmupTicks(80)
      .cooldownTicks(160)
      .onNodeClick((node) => {
        const graphData = dataRef.current
        if (!graphData) return
        applyFocus(focusSet(graphData.neighbors, node.id))
        onFocusChangeRef.current?.(node)

        if (node.x != null && node.y != null && node.z != null) {
          const ratio = 1 + FOCUS_DISTANCE / (Math.hypot(node.x, node.y, node.z) || 1)
          graph.cameraPosition(
            { x: node.x * ratio, y: node.y * ratio, z: node.z * ratio },
            { x: node.x, y: node.y, z: node.z },
            CAMERA_TRANSITION_MS,
          )
        }
      })
      .onBackgroundClick(() => {
        applyFocus(null)
        onFocusChangeRef.current?.(null)
        graph.zoomToFit(CAMERA_TRANSITION_MS)
      })

    graphRef.current = graph

    const resize = new ResizeObserver(() => {
      graph.width(container.clientWidth).height(container.clientHeight)
    })
    resize.observe(container)
    graph.width(container.clientWidth).height(container.clientHeight)

    return () => {
      resize.disconnect()
      graph._destructor()
      graphRef.current = null
      dataRef.current = null
      focusedRef.current = null
    }
  }, [applyFocus])

  // ---- feed data separately ------------------------------------------------
  useEffect(() => {
    const graph = graphRef.current
    if (!graph) return

    const graph3d = toGraph3D(data)
    dataRef.current = graph3d

    // A new payload invalidates any focused neighbourhood: those ids may not
    // even be present any more.
    focusedRef.current = null
    onFocusChangeRef.current?.(null)

    graph.nodeResolution(
      graph3d.nodes.length > LOW_DETAIL_NODE_THRESHOLD
        ? NODE_RESOLUTION_LOW
        : NODE_RESOLUTION_HIGH,
    )
    graph.graphData({
      nodes: graph3d.nodes as GNode[],
      links: graph3d.links as unknown as GLink[],
    })
  }, [data])

  // ---- external focus reset ------------------------------------------------
  useEffect(() => {
    if (clearFocusSignal === undefined) return
    applyFocus(null)
    graphRef.current?.zoomToFit(CAMERA_TRANSITION_MS)
  }, [clearFocusSignal, applyFocus])

  return <div ref={containerRef} className="size-full" />
}
