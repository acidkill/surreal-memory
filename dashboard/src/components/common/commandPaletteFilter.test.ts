import { describe, expect, it } from "vitest"

import { matchesQuery } from "./commandPaletteFilter"

describe("matchesQuery", () => {
  it("keeps everything for an empty or blank query", () => {
    expect(matchesQuery("Graph", "")).toBe(true)
    expect(matchesQuery("Graph", "   ")).toBe(true)
  })

  it("matches case-insensitively on a substring of the label", () => {
    expect(matchesQuery("Graph", "gra")).toBe(true)
    expect(matchesQuery("Tool Stats", "STATS")).toBe(true)
  })

  it("drops labels that do not contain the query", () => {
    expect(matchesQuery("Overview", "graph")).toBe(false)
    expect(matchesQuery("Settings", "zzz")).toBe(false)
  })
})
