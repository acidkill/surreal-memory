import { describe, expect, it } from "vitest"

import type { GraphResponse } from "@/api/types"
import {
  PARTICLE_GLOBAL_CAP,
  PARTICLE_NODE_THRESHOLD,
  assignParticleBudget,
  escapeHtml,
  focusSet,
  toGraph3D,
  type Graph3DLink,
} from "./graph-data"

function response(
  neurons: Array<{ id: string; content?: string; type?: string }>,
  synapses: Array<{ source_id: string; target_id: string; weight?: number }>,
): GraphResponse {
  return {
    neurons: neurons.map((n) => ({
      id: n.id,
      content: n.content ?? n.id,
      type: n.type ?? "concept",
      metadata: {},
    })),
    synapses: synapses.map((s, i) => ({
      id: `s${i}`,
      source_id: s.source_id,
      target_id: s.target_id,
      type: "related_to",
      weight: s.weight ?? 0.5,
      direction: "bidirectional",
    })),
    fibers: [],
    total_neurons: neurons.length,
    total_synapses: synapses.length,
    stats: { neuron_count: neurons.length, synapse_count: synapses.length, fiber_count: 0 },
  }
}

// ── escapeHtml — a security control, not formatting ────────────────────────

describe("escapeHtml", () => {
  it("neutralises a script-bearing image tag", () => {
    const escaped = escapeHtml('<img src=x onerror="alert(1)">')
    expect(escaped).not.toContain("<img")
    expect(escaped).not.toContain('"')
    expect(escaped).toBe("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;")
  })

  it("escapes every character that can break out of an HTML text node or attribute", () => {
    expect(escapeHtml(`& < > " '`)).toBe("&amp; &lt; &gt; &quot; &#39;")
  })

  it("escapes the ampersand first, so entities are not double-decoded", () => {
    // Naive ordering yields "&amp;lt;" for "<" because the & of &lt; is escaped
    // after the fact. This asserts the ampersand pass runs first.
    expect(escapeHtml("<")).toBe("&lt;")
    expect(escapeHtml("&lt;")).toBe("&amp;lt;")
  })

  it("leaves ordinary prose untouched", () => {
    expect(escapeHtml("Redis caching decision")).toBe("Redis caching decision")
  })
})

describe("toGraph3D label escaping", () => {
  it("never puts raw neuron content into the label fed to the 3D tooltip", () => {
    const graph = toGraph3D(response([{ id: "n1", content: "<b>bold</b>" }], []))
    expect(graph.nodes[0].label).not.toContain("<b>")
    expect(graph.nodes[0].label).toContain("&lt;b&gt;")
    // The raw value is still carried for React, which escapes its own text.
    expect(graph.nodes[0].content).toBe("<b>bold</b>")
  })
})

// ── graph construction ─────────────────────────────────────────────────────

describe("toGraph3D", () => {
  it("drops synapses whose endpoints are not both present", () => {
    const graph = toGraph3D(
      response(
        [{ id: "a" }, { id: "b" }],
        [
          { source_id: "a", target_id: "b" },
          { source_id: "a", target_id: "ghost" },
        ],
      ),
    )
    expect(graph.links).toHaveLength(1)
  })

  it("collapses a reciprocal pair into one link", () => {
    const graph = toGraph3D(
      response(
        [{ id: "a" }, { id: "b" }],
        [
          { source_id: "a", target_id: "b" },
          { source_id: "b", target_id: "a" },
        ],
      ),
    )
    expect(graph.links).toHaveLength(1)
  })

  it("drops self-links", () => {
    const graph = toGraph3D(response([{ id: "a" }], [{ source_id: "a", target_id: "a" }]))
    expect(graph.links).toHaveLength(0)
    expect(graph.nodes[0].degree).toBe(0)
  })

  it("sizes nodes by degree, sublinearly", () => {
    const graph = toGraph3D(
      response(
        [{ id: "hub" }, { id: "a" }, { id: "b" }, { id: "c" }, { id: "lonely" }],
        [
          { source_id: "hub", target_id: "a" },
          { source_id: "hub", target_id: "b" },
          { source_id: "hub", target_id: "c" },
        ],
      ),
    )
    const hub = graph.nodes.find((n) => n.id === "hub")!
    const lonely = graph.nodes.find((n) => n.id === "lonely")!
    expect(hub.degree).toBe(3)
    expect(lonely.degree).toBe(0)
    expect(hub.val).toBeGreaterThan(lonely.val)
    // sqrt-scaled: a 3-degree hub must not be 3x the volume of an isolated node
    expect(hub.val).toBeLessThan(lonely.val * 3)
  })

  it("builds a symmetric neighbour index", () => {
    const graph = toGraph3D(
      response([{ id: "a" }, { id: "b" }], [{ source_id: "a", target_id: "b" }]),
    )
    expect(graph.neighbors.get("a")).toContain("b")
    expect(graph.neighbors.get("b")).toContain("a")
  })

  it("falls back to the id when a neuron has empty content", () => {
    const graph = toGraph3D(response([{ id: "n1", content: "" }], []))
    expect(graph.nodes[0].label).toBe("n1")
  })

  it("does not collide two distinct edges when a node id contains the old '|' delimiter", () => {
    // Regression: a naive `${a}|${b}` dedup key lets ("a", "b|c") and
    // ("a|b", "c") both synthesize the key "a|b|c", silently dropping one edge.
    const graph = toGraph3D(
      response(
        [{ id: "a" }, { id: "b|c" }, { id: "a|b" }, { id: "c" }],
        [
          { source_id: "a", target_id: "b|c" },
          { source_id: "a|b", target_id: "c" },
        ],
      ),
    )
    expect(graph.links).toHaveLength(2)
  })

  it("does not crash when a neuron's content violates the string contract", () => {
    // The API contract says `content: string`, but nothing validates the
    // response at this boundary — a malformed payload must degrade, not throw.
    const malformed = response([{ id: "n1" }], [])
    ;(malformed.neurons[0] as { content: unknown }).content = 42

    expect(() => toGraph3D(malformed)).not.toThrow()
    const graph = toGraph3D(malformed)
    expect(graph.nodes[0].label).toBe("42")
    expect(graph.nodes[0].content).toBe("42")
  })
})

describe("focusSet", () => {
  it("includes the node itself and its direct neighbours only", () => {
    const graph = toGraph3D(
      response(
        [{ id: "a" }, { id: "b" }, { id: "c" }],
        [
          { source_id: "a", target_id: "b" },
          { source_id: "b", target_id: "c" },
        ],
      ),
    )
    const set = focusSet(graph.neighbors, "a")
    expect(set.has("a")).toBe(true)
    expect(set.has("b")).toBe(true)
    // two hops away — must stay dimmed
    expect(set.has("c")).toBe(false)
  })

  it("returns just the node when it has no neighbours", () => {
    expect([...focusSet(new Map(), "alone")]).toEqual(["alone"])
  })
})

// ── particle budget — a performance guard, so assert the bounds ─────────────

describe("assignParticleBudget", () => {
  function links(count: number, weight = 1): Graph3DLink[] {
    return Array.from({ length: count }, (_, i) => ({
      source: `s${i}`,
      target: `t${i}`,
      weight,
      particles: 0,
    }))
  }

  it("never exceeds the global cap however many links there are", () => {
    const out = assignParticleBudget(links(5000), 10)
    const total = out.reduce((sum, l) => sum + l.particles, 0)
    expect(total).toBeLessThanOrEqual(PARTICLE_GLOBAL_CAP)
    expect(total).toBeGreaterThan(0)
  })

  it("switches particles off entirely above the node threshold", () => {
    const out = assignParticleBudget(links(10), PARTICLE_NODE_THRESHOLD + 1)
    expect(out.every((l) => l.particles === 0)).toBe(true)
  })

  it("still grants particles above the threshold when forced (focus mode)", () => {
    const out = assignParticleBudget(links(10), PARTICLE_NODE_THRESHOLD + 1, { force: true })
    expect(out.some((l) => l.particles > 0)).toBe(true)
  })

  it("spends the budget on the strongest links", () => {
    const mixed: Graph3DLink[] = [
      { source: "a", target: "b", weight: 0.01, particles: 0 },
      { source: "c", target: "d", weight: 0.99, particles: 0 },
    ]
    const out = assignParticleBudget(mixed, 2)
    const weak = out.find((l) => l.weight === 0.01)!
    const strong = out.find((l) => l.weight === 0.99)!
    expect(strong.particles).toBeGreaterThan(0)
    expect(weak.particles).toBe(0)
  })

  it("handles an empty link list", () => {
    expect(assignParticleBudget([], 0)).toEqual([])
  })

  it("is applied by toGraph3D, not left to the caller", () => {
    const many = Array.from({ length: 40 }, (_, i) => ({ id: `n${i}` }))
    const edges = Array.from({ length: 39 }, (_, i) => ({
      source_id: `n${i}`,
      target_id: `n${i + 1}`,
      weight: i / 39,
    }))
    const graph = toGraph3D(response(many, edges))
    const total = graph.links.reduce((sum, l) => sum + l.particles, 0)
    expect(total).toBeLessThanOrEqual(PARTICLE_GLOBAL_CAP)
  })
})
