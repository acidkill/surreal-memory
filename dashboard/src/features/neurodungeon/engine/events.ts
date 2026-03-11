/* ------------------------------------------------------------------ */
/*  World events — generated from brain health data                   */
/*  Events modify dungeon state: spawn enemies, block corridors, etc  */
/* ------------------------------------------------------------------ */

import type { WorldEvent, DungeonFloor, Room, Entity, Position } from "./types"
import { WorldEventType, EntityType, RoomType, TileType } from "./types"
import type { HealthReport } from "@/api/types"

/** Generate world events based on brain health report */
export function generateEvents(health: HealthReport | null): WorldEvent[] {
  if (!health) return []

  const events: WorldEvent[] = []

  // Orphan rate > 20% -> zombie horde
  if (health.orphan_rate > 0.2) {
    events.push({
      type: WorldEventType.ZOMBIE_HORDE,
      message: `Orphan memories rise as undead! (${Math.round(health.orphan_rate * 100)}% orphan rate)`,
      severity: health.orphan_rate > 0.4 ? 3 : 2,
    })
  }

  // Low connectivity -> corridors collapse
  if (health.connectivity < 0.4) {
    events.push({
      type: WorldEventType.CORRIDOR_COLLAPSE,
      message: `Weak connections crumble. Some paths are blocked. (${Math.round(health.connectivity * 100)}% connectivity)`,
      severity: health.connectivity < 0.2 ? 3 : 2,
    })
  }

  // Low activation efficiency -> thicker fog
  if (health.activation_efficiency < 0.3) {
    events.push({
      type: WorldEventType.FOG_THICKEN,
      message: `Dormant memories thicken the fog. Visibility reduced. (${Math.round(health.activation_efficiency * 100)}% activation)`,
      severity: 1,
    })
  }

  // Low freshness -> rooms decay
  if (health.freshness < 0.3) {
    events.push({
      type: WorldEventType.ROOM_DECAY,
      message: `Stale memories erode the dungeon. Rooms are unstable. (${Math.round(health.freshness * 100)}% freshness)`,
      severity: 2,
    })
  }

  // Low consolidation ratio -> boss spawns
  if (health.consolidation_ratio < 0.3) {
    events.push({
      type: WorldEventType.BOSS_SPAWN,
      message: `Unconsolidated knowledge manifests as a powerful guardian!`,
      severity: 3,
    })
  }

  // Low purity -> poison traps
  if (health.purity_score < 0.5) {
    events.push({
      type: WorldEventType.POISON_TRAPS,
      message: `Impure memories leak toxins. Watch your step! (${Math.round(health.purity_score * 100)}% purity)`,
      severity: health.purity_score < 0.3 ? 3 : 1,
    })
  }

  return events
}

/** Apply world events to a dungeon floor, return modified floor */
export function applyEvents(
  floor: DungeonFloor,
  events: WorldEvent[],
): DungeonFloor {
  let modifiedFloor = floor

  for (const event of events) {
    switch (event.type) {
      case WorldEventType.ZOMBIE_HORDE:
        modifiedFloor = spawnZombies(modifiedFloor, event.severity)
        break
      case WorldEventType.CORRIDOR_COLLAPSE:
        modifiedFloor = collapseCorridors(modifiedFloor, event.severity)
        break
      case WorldEventType.POISON_TRAPS:
        modifiedFloor = addPoisonTraps(modifiedFloor, event.severity)
        break
      // FOG_THICKEN, ROOM_DECAY, BOSS_SPAWN handled at render/game-loop level
      default:
        break
    }
  }

  return modifiedFloor
}

/** Spawn extra zombie enemies in empty rooms */
function spawnZombies(floor: DungeonFloor, severity: number): DungeonFloor {
  const count = severity * 2
  let spawned = 0

  const updatedRooms = floor.rooms.map((room): Room => {
    if (spawned >= count) return room
    if (room.type !== RoomType.EMPTY) return room
    if (room.entities.length > 0) return room

    spawned++
    const zombie: Entity = {
      id: `zombie-${room.id}`,
      type: EntityType.ENEMY,
      position: { ...room.center },
      name: "Orphan Memory",
      content: "A disconnected memory, wandering without purpose...",
      stats: {
        hp: 15 + severity * 5,
        maxHp: 15 + severity * 5,
        atk: 4 + severity * 2,
        def: 1,
        spd: 3,
        neuronType: "error",
      },
      item: null,
      defeated: false,
      neuronId: null,
    }

    return { ...room, type: RoomType.ENEMY_ROOM, entities: [zombie] }
  })

  return { ...floor, rooms: updatedRooms }
}

/** Block some corridor tiles (convert back to walls) */
function collapseCorridors(floor: DungeonFloor, severity: number): DungeonFloor {
  const tiles = floor.tiles.map((row) => [...row])
  let blocked = 0
  const maxBlocks = severity * 3

  for (const corridor of floor.corridors) {
    if (blocked >= maxBlocks) break
    if (corridor.weight > 0.5) continue // strong connections don't collapse

    // Block middle section of weak corridors
    const mid = Math.floor(corridor.tiles.length / 2)
    for (let i = mid - 1; i <= mid + 1 && i < corridor.tiles.length; i++) {
      const pos = corridor.tiles[i]
      if (!pos) continue
      if (tiles[pos.y]?.[pos.x] === TileType.CORRIDOR) {
        tiles[pos.y]![pos.x] = TileType.WALL
        blocked++
      }
    }
  }

  return { ...floor, tiles }
}

/** Place trap tiles in some rooms */
function addPoisonTraps(floor: DungeonFloor, severity: number): DungeonFloor {
  const tiles = floor.tiles.map((row) => [...row])
  let placed = 0
  const maxTraps = severity * 2

  for (const room of floor.rooms) {
    if (placed >= maxTraps) break
    if (room.type === RoomType.TREASURE || room.type === RoomType.STAIRS) continue

    // Place trap at room entrance (one tile inside)
    const trapPos: Position = {
      x: room.rect.x + 1,
      y: room.rect.y + 1,
    }
    if (tiles[trapPos.y]?.[trapPos.x] === TileType.FLOOR) {
      tiles[trapPos.y]![trapPos.x] = TileType.TRAP
      placed++
    }
  }

  return { ...floor, tiles }
}

/** Get fog radius modifier from events */
export function getFogModifier(events: WorldEvent[]): number {
  const fogEvent = events.find((e) => e.type === WorldEventType.FOG_THICKEN)
  if (!fogEvent) return 0
  return -fogEvent.severity // reduce fog radius
}
