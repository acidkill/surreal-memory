import type { OracleCard, DailyReading, WhatIfScenario, MatchupState } from "./types"
import { WHAT_IF_TEMPLATES, DAILY_INTERPRETATIONS, MATCHUP_PROMPTS } from "./templates"

// Deterministic seed from date + brain name
function dailySeed(date: string, brainName: string): number {
  let hash = 0
  const str = `${date}:${brainName}`
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i)
    hash |= 0
  }
  return Math.abs(hash)
}

// Seeded pseudo-random (mulberry32)
function seededRandom(seed: number): () => number {
  let s = seed | 0
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Pick N unique items from array using seeded random
function pickN<T>(items: readonly T[], n: number, rand: () => number): T[] {
  const pool = [...items]
  const picked: T[] = []
  for (let i = 0; i < n && pool.length > 0; i++) {
    const idx = Math.floor(rand() * pool.length)
    picked.push(pool[idx])
    pool.splice(idx, 1)
  }
  return picked
}

// Pick a template string using random
function pickTemplate(templates: readonly string[], rand: () => number): string {
  return templates[Math.floor(rand() * templates.length)]
}

// Interpolate {past}, {present}, {future}, {cardA}, {cardB}, {suit} etc.
function interpolate(
  template: string,
  vars: Record<string, string>,
): string {
  return Object.entries(vars).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, value),
    template,
  )
}

export function generateDailyReading(
  cards: readonly OracleCard[],
  brainName: string,
  date?: string,
): DailyReading | null {
  if (cards.length < 3) return null

  const today = date ?? new Date().toISOString().slice(0, 10)
  const seed = dailySeed(today, brainName)
  const rand = seededRandom(seed)

  // Sort by activation descending, then pick from top half for better cards
  const sorted = [...cards].sort((a, b) => b.activation - a.activation)
  const topPool = sorted.slice(0, Math.max(Math.ceil(sorted.length * 0.6), 3))
  const [past, present, future] = pickN(topPool, 3, rand)

  const interpretation = interpolate(
    pickTemplate(DAILY_INTERPRETATIONS, rand),
    {
      past: past.suit.name,
      present: present.suit.name,
      future: future.suit.name,
      pastContent: past.content,
      presentContent: present.content,
      futureContent: future.content,
    },
  )

  return { past, present, future, interpretation, date: today, brainName }
}

export function generateWhatIf(
  cards: readonly OracleCard[],
  seed?: number,
): WhatIfScenario | null {
  if (cards.length < 3) return null

  const rand = seededRandom(seed ?? Date.now())

  // Pick 2 decisions + 1 wildcard
  const decisions = pickN(cards, 2, rand)
  const remaining = cards.filter((c) => !decisions.includes(c))
  const [wildcard] = pickN(remaining.length > 0 ? remaining : cards, 1, rand)

  const scenario = interpolate(pickTemplate(WHAT_IF_TEMPLATES, rand), {
    cardA: decisions[0].content,
    cardB: decisions[1].content,
    suitA: decisions[0].suit.name,
    suitB: decisions[1].suit.name,
    wildcard: wildcard.content,
    wildcardSuit: wildcard.suit.name,
  })

  return { decisions, error: wildcard, scenario }
}

export function generateMatchup(
  cards: readonly OracleCard[],
  round = 1,
  totalRounds = 5,
  previousPicks: readonly string[] = [],
  seed?: number,
): MatchupState | null {
  if (cards.length < 2) return null

  const rand = seededRandom(seed ?? Date.now() + round)
  const available = cards.filter((c) => !previousPicks.includes(c.id))
  const pool = available.length >= 2 ? available : cards

  const [cardA, cardB] = pickN(pool, 2, rand)
  return { cardA, cardB, round, score: 0, totalRounds }
}

export function getMatchupPrompt(
  cardA: OracleCard,
  cardB: OracleCard,
  seed?: number,
): string {
  const rand = seededRandom(seed ?? Date.now())
  return interpolate(pickTemplate(MATCHUP_PROMPTS, rand), {
    suitA: cardA.suit.name,
    suitB: cardB.suit.name,
    contentA: cardA.content,
    contentB: cardB.content,
  })
}
