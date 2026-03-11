/* ------------------------------------------------------------------ */
/*  Engagement systems — chain combos, danger, floor ratings           */
/*                                                                      */
/*  These systems create tension and meaningful decisions:              */
/*  - Chain: follow synapse paths for multiplier (uses real brain data) */
/*  - Danger: stay too long = enemies get stronger, forces pacing      */
/*  - Rating: S/A/B/C/D per floor, drives completionism + replay      */
/*  - Kill streak: consecutive kills without damage = bonus            */
/* ------------------------------------------------------------------ */

import type {
  MemoryChain,
  DangerLevel,
  FloorResult,
  FloorRating,
  DungeonFloor,
  Room,
  DungeonState,
} from "./types"
import { FloorRating as FR } from "./types"

// --------------- Constants ---------------

const DANGER_ESCALATION_RATE = 15  // turns per danger level
const MAX_DANGER = 10
const CHAIN_MULTIPLIER_STEP = 0.5  // each chain link adds 0.5x
const KILL_STREAK_BONUS = 25       // score per streak kill

// --------------- Memory Chain ---------------

export function createChain(): MemoryChain {
  return {
    currentNeuronId: null,
    chainLength: 0,
    maxChain: 0,
    multiplier: 1,
  }
}

/**
 * Update chain when visiting a room.
 * Chain continues if the new room's neuron is connected to the previous via synapse.
 */
export function updateChain(
  chain: MemoryChain,
  room: Room,
  floor: DungeonFloor,
): { chain: MemoryChain; chainBroke: boolean; chainMessage: string | null } {
  if (!room.neuron) {
    return { chain, chainBroke: false, chainMessage: null }
  }

  const newNeuronId = room.neuron.id

  // First room — start chain
  if (!chain.currentNeuronId) {
    return {
      chain: {
        currentNeuronId: newNeuronId,
        chainLength: 1,
        maxChain: Math.max(chain.maxChain, 1),
        multiplier: 1,
      },
      chainBroke: false,
      chainMessage: null,
    }
  }

  // Check if connected via corridor (synapse)
  const isConnected = floor.corridors.some(
    (c) =>
      (c.from === chain.currentNeuronId && c.to === newNeuronId) ||
      (c.to === chain.currentNeuronId && c.from === newNeuronId),
  )

  if (isConnected) {
    const newLength = chain.chainLength + 1
    const newMultiplier = 1 + newLength * CHAIN_MULTIPLIER_STEP
    return {
      chain: {
        currentNeuronId: newNeuronId,
        chainLength: newLength,
        maxChain: Math.max(chain.maxChain, newLength),
        multiplier: newMultiplier,
      },
      chainBroke: false,
      chainMessage: `Memory Chain x${newLength}! (${newMultiplier.toFixed(1)}x score)`,
    }
  }

  // Chain broken — visited unconnected room
  const hadChain = chain.chainLength >= 2
  return {
    chain: {
      currentNeuronId: newNeuronId,
      chainLength: 1,
      maxChain: chain.maxChain,
      multiplier: 1,
    },
    chainBroke: hadChain,
    chainMessage: hadChain
      ? `Chain broke! Was x${chain.chainLength}.`
      : null,
  }
}

// --------------- Danger Escalation ---------------

export function createDanger(): DangerLevel {
  return {
    level: 0,
    turnsOnFloor: 0,
    escalationRate: DANGER_ESCALATION_RATE,
  }
}

/** Tick danger each turn. Returns new danger + whether level increased. */
export function tickDanger(danger: DangerLevel): {
  danger: DangerLevel
  levelUp: boolean
} {
  const newTurns = danger.turnsOnFloor + 1
  const newLevel = Math.min(MAX_DANGER, Math.floor(newTurns / danger.escalationRate))
  const levelUp = newLevel > danger.level

  return {
    danger: {
      ...danger,
      turnsOnFloor: newTurns,
      level: newLevel,
    },
    levelUp,
  }
}

/** Reset danger when entering new floor */
export function resetDangerForFloor(): DangerLevel {
  return createDanger()
}

/** Get danger-based enemy stat modifier */
export function dangerStatModifier(danger: DangerLevel): number {
  return 1 + danger.level * 0.1 // +10% per danger level
}

/** Get danger color for UI */
export function dangerColor(level: number): string {
  if (level <= 2) return "#00d084"
  if (level <= 4) return "#ffa726"
  if (level <= 6) return "#ff6b6b"
  return "#f44336"
}

/** Get danger label */
export function dangerLabel(level: number): string {
  if (level <= 1) return "Safe"
  if (level <= 3) return "Uneasy"
  if (level <= 5) return "Dangerous"
  if (level <= 7) return "Perilous"
  return "Nightmare"
}

// --------------- Floor Rating ---------------

export function calculateFloorRating(
  floor: DungeonFloor,
  state: DungeonState,
): FloorResult {
  const totalRooms = floor.rooms.length
  const visitedRooms = floor.rooms.filter((r) => r.visited).length
  const explorationPct = totalRooms > 0 ? visitedRooms / totalRooms : 0

  const totalEnemies = floor.rooms.reduce(
    (sum, r) => sum + r.entities.filter((e) => e.type === "enemy" || e.type === "boss").length,
    0,
  )
  const killedEnemies = floor.rooms.reduce(
    (sum, r) => sum + r.entities.filter((e) => (e.type === "enemy" || e.type === "boss") && e.defeated).length,
    0,
  )
  const killPct = totalEnemies > 0 ? killedEnemies / totalEnemies : 1

  const chainBonus = state.chain.maxChain
  const turnsUsed = state.danger.turnsOnFloor

  // Score components
  const explorationScore = explorationPct * 40
  const killScore = killPct * 30
  const chainScore = Math.min(20, chainBonus * 4)
  const speedScore = Math.max(0, 10 - turnsUsed / 20)
  const total = explorationScore + killScore + chainScore + speedScore

  let rating: FloorRating
  if (total >= 85) rating = FR.S
  else if (total >= 70) rating = FR.A
  else if (total >= 50) rating = FR.B
  else if (total >= 30) rating = FR.C
  else rating = FR.D

  const ratingMultiplier = { S: 3, A: 2, B: 1.5, C: 1, D: 0.5 }[rating]
  const scoreEarned = Math.floor(200 * ratingMultiplier)

  return {
    floorName: floor.name,
    rating,
    explorationPct,
    killPct,
    chainBonus,
    turnsUsed,
    scoreEarned,
  }
}

export function ratingColor(rating: FloorRating): string {
  switch (rating) {
    case FR.S: return "#ffd700"
    case FR.A: return "#00d084"
    case FR.B: return "#64b5f6"
    case FR.C: return "#ffa726"
    case FR.D: return "#ff6b6b"
  }
}

// --------------- Kill Streak ---------------

export function killStreakBonus(streak: number): number {
  if (streak < 2) return 0
  return streak * KILL_STREAK_BONUS
}

export function killStreakMessage(streak: number): string | null {
  if (streak === 2) return "Double Kill!"
  if (streak === 3) return "Triple Kill!"
  if (streak === 4) return "Quad Kill!"
  if (streak >= 5) return `UNSTOPPABLE! x${streak} Kill Streak!`
  return null
}
