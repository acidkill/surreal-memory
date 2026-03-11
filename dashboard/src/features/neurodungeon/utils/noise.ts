/* ------------------------------------------------------------------ */
/*  Simple seeded PRNG + noise for deterministic dungeon generation   */
/* ------------------------------------------------------------------ */

/** Mulberry32 — fast seeded 32-bit PRNG */
export function createRng(seed: number): () => number {
  let s = seed | 0
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Seeded random integer in [min, max] inclusive */
export function randInt(rng: () => number, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min
}

/** Shuffle array in-place using Fisher-Yates */
export function shuffle<T>(rng: () => number, arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const tmp = arr[i]!
    arr[i] = arr[j]!
    arr[j] = tmp
  }
  return arr
}

/** Pick random element */
export function pick<T>(rng: () => number, arr: readonly T[]): T {
  return arr[Math.floor(rng() * arr.length)]!
}

/** Generate a seed from a string (brain name + fiber id) */
export function hashSeed(str: string): number {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i)
    hash = ((hash << 5) - hash + ch) | 0
  }
  return hash
}
