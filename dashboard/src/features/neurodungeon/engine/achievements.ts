/* ------------------------------------------------------------------ */
/*  Achievement system — definitions, checks, persistence             */
/* ------------------------------------------------------------------ */

import type { DungeonState } from "./types"
import { GamePhase } from "./types"

// --------------- Achievement definitions ---------------

export interface Achievement {
  id: string
  name: string
  description: string
  icon: string
  check: (state: DungeonState) => boolean
}

export const ACHIEVEMENTS: Achievement[] = [
  {
    id: "first_blood",
    name: "First Blood",
    description: "Defeat your first enemy",
    icon: "X",
    check: (s) => s.player.enemiesDefeated >= 1,
  },
  {
    id: "monster_slayer",
    name: "Monster Slayer",
    description: "Defeat 10 enemies in one run",
    icon: "X",
    check: (s) => s.player.enemiesDefeated >= 10,
  },
  {
    id: "explorer",
    name: "Explorer",
    description: "Explore 10 rooms in one run",
    icon: "M",
    check: (s) => s.player.roomsExplored >= 10,
  },
  {
    id: "cartographer",
    name: "Cartographer",
    description: "Explore 25 rooms in one run",
    icon: "M",
    check: (s) => s.player.roomsExplored >= 25,
  },
  {
    id: "dungeon_master",
    name: "Dungeon Master",
    description: "Complete the dungeon",
    icon: "C",
    check: (s) => s.phase === GamePhase.VICTORY,
  },
  {
    id: "hoarder",
    name: "Hoarder",
    description: "Have a full inventory (5 items)",
    icon: "*",
    check: (s) => s.player.inventory.length >= 5,
  },
  {
    id: "memory_keeper",
    name: "Memory Keeper",
    description: "Visit 15 unique neurons",
    icon: "N",
    check: (s) => s.visitedNeuronIds.size >= 15,
  },
  {
    id: "speed_runner",
    name: "Speed Runner",
    description: "Complete a floor in under 50 turns",
    icon: ">",
    check: (s) => s.phase === GamePhase.VICTORY && s.player.turnsElapsed < 50,
  },
  {
    id: "survivor",
    name: "Survivor",
    description: "Reach floor 3",
    icon: "F",
    check: (s) => s.currentFloorIndex >= 2,
  },
  {
    id: "deep_diver",
    name: "Deep Diver",
    description: "Reach floor 5",
    icon: "F",
    check: (s) => s.currentFloorIndex >= 4,
  },
  {
    id: "untouchable",
    name: "Untouchable",
    description: "Defeat an enemy without taking damage",
    icon: "S",
    check: (s) => s.player.enemiesDefeated > 0 && s.player.stats.hp === s.player.stats.maxHp,
  },
  {
    id: "high_roller",
    name: "High Roller",
    description: "Score 1000+ points",
    icon: "$",
    check: (s) => s.player.score >= 1000,
  },
  {
    id: "legend",
    name: "Legend",
    description: "Score 5000+ points",
    icon: "$",
    check: (s) => s.player.score >= 5000,
  },
]

// --------------- Persistence ---------------

const STORAGE_KEY = "neurodungeon_achievements"

export function getUnlockedAchievements(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return new Set()
    return new Set(JSON.parse(raw) as string[])
  } catch {
    return new Set()
  }
}

export function saveAchievements(ids: Set<string>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...ids]))
  } catch {
    // localStorage unavailable
  }
}

/** Check all achievements, return newly unlocked ones */
export function checkAchievements(
  state: DungeonState,
  alreadyUnlocked: Set<string>,
): Achievement[] {
  const newlyUnlocked: Achievement[] = []
  for (const ach of ACHIEVEMENTS) {
    if (alreadyUnlocked.has(ach.id)) continue
    if (ach.check(state)) {
      newlyUnlocked.push(ach)
    }
  }
  return newlyUnlocked
}

/** Get full achievement list with unlock status */
export function getAchievementList(unlocked: Set<string>): Array<Achievement & { isUnlocked: boolean }> {
  return ACHIEVEMENTS.map((a) => ({ ...a, isUnlocked: unlocked.has(a.id) }))
}
