/* ------------------------------------------------------------------ */
/*  Transform API graph data into dungeon generation input            */
/* ------------------------------------------------------------------ */

import { useMemo } from "react"
import { useGraph, useFibers } from "@/api/hooks/useDashboard"
import type { DungeonGenInput } from "../engine/types"
import { hashSeed } from "../utils/noise"

/**
 * Fetches graph + fiber data and prepares DungeonGenInput.
 * Returns { inputs, isLoading, error } where inputs is an array
 * of DungeonGenInput (one per fiber/floor).
 */
export function useDungeonData() {
  const { data: graph, isLoading: graphLoading, error: graphError } = useGraph(500)
  const { data: fibers, isLoading: fibersLoading, error: fibersError } = useFibers()

  const inputs = useMemo(() => {
    if (!graph || !fibers) return []

    const fiberList = fibers.fibers
    if (fiberList.length === 0) return []

    // Sort fibers by neuron count (ascending = easier floors first)
    const sorted = [...fiberList].sort((a, b) => a.neuron_count - b.neuron_count)

    // Limit to 5 floors max
    const selected = sorted.slice(0, 5)

    return selected.map((fiber, i): DungeonGenInput => ({
      neurons: graph.neurons,
      synapses: graph.synapses,
      fibers: graph.fibers,
      fiberId: fiber.id,
      floorDepth: i,
      seed: hashSeed(fiber.id + fiber.summary),
    }))
  }, [graph, fibers])

  return {
    inputs,
    isLoading: graphLoading || fibersLoading,
    error: graphError ?? fibersError ?? null,
    totalNeurons: graph?.total_neurons ?? 0,
    totalFibers: fibers?.fibers.length ?? 0,
  }
}
