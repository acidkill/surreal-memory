/* ------------------------------------------------------------------ */
/*  Turn-based game loop — processes actions, returns new state       */
/*  Pure function: (state, action) -> state (no side effects)         */
/* ------------------------------------------------------------------ */

import type {
  DungeonState,
  GameAction,
  DungeonFloor,
  Position,
  Room,
  Entity,
  TurnLogEntry,
} from "./types"
import {
  Direction,
  GamePhase,
  TileType,
  EntityType,
  RoomType,
  FOG_RADIUS,
} from "./types"
import { tickBuffs, DROP_SCORE_BONUS } from "./items"
import {
  updateChain,
  tickDanger,
  dangerLabel,
  calculateFloorRating,
  resetDangerForFloor,
  createChain,
} from "./engagement"

/** Process one game action, return new state (immutable). */
export function processTurn(state: DungeonState, action: GameAction): DungeonState {
  switch (action.type) {
    case "move":
      return handleMove(state, action.direction)
    case "interact":
      return handleInteract(state)
    case "wait":
      return advanceTurn(state, "You wait...")
    case "next_floor":
      return handleNextFloor(state)
    case "use_item":
      return handleUseItem(state, action.itemIndex)
    case "drop_item":
      return handleDropItem(state, action.itemIndex)
    case "attack":
    case "defend":
    case "flee":
      // Combat handled externally via CombatOverlay
      return state
    case "answer_quiz":
      return state
  }
}

// --------------- Movement ---------------

function handleMove(state: DungeonState, direction: Direction): DungeonState {
  if (state.phase !== GamePhase.EXPLORING) return state

  const floor = currentFloor(state)
  const pos = state.player.position
  const next = movePosition(pos, direction)

  // Bounds check
  if (next.x < 0 || next.x >= floor.width || next.y < 0 || next.y >= floor.height) {
    return state
  }

  // Collision — can't walk through walls
  const tile = floor.tiles[next.y]![next.x]!
  if (tile === TileType.WALL || tile === TileType.VOID) {
    return state
  }

  // Move player + tick buffs + tick danger
  const tickedPlayer = tickBuffs(state.player)
  const { danger: newDanger, levelUp: dangerLevelUp } = tickDanger(state.danger)

  let newState: DungeonState = {
    ...state,
    player: {
      ...tickedPlayer,
      position: { ...next },
      turnsElapsed: state.player.turnsElapsed + 1,
    },
    danger: newDanger,
  }

  // Danger level up warning
  if (dangerLevelUp) {
    newState = {
      ...newState,
      turnLog: [
        ...newState.turnLog,
        logEntry(newState.player.turnsElapsed, `Danger rising! ${dangerLabel(newDanger.level)}...`, "event"),
      ],
    }
  }

  // Reveal fog of war
  newState = revealFog(newState)

  // Check room entry
  const room = findRoomAt(floor, next)
  if (room && !room.visited) {
    newState = enterRoom(newState, room)

    // Update memory chain
    const { chain: newChain, chainMessage } = updateChain(newState.chain, room, floor)
    newState = { ...newState, chain: newChain }
    if (chainMessage) {
      newState = {
        ...newState,
        turnLog: [
          ...newState.turnLog,
          logEntry(newState.player.turnsElapsed, chainMessage, "discovery"),
        ],
      }
    }
  }

  // Check entity collision
  const entity = findEntityAt(floor, next)
  if (entity && !entity.defeated) {
    newState = entityEncounter(newState, entity, room)
  }

  // Check stairs
  if (tile === TileType.STAIRS_DOWN) {
    newState = advanceTurn(newState, "You see stairs leading deeper... Press > to descend.")
  }

  return newState
}

function movePosition(pos: Position, dir: Direction): Position {
  switch (dir) {
    case Direction.UP:
      return { x: pos.x, y: pos.y - 1 }
    case Direction.DOWN:
      return { x: pos.x, y: pos.y + 1 }
    case Direction.LEFT:
      return { x: pos.x - 1, y: pos.y }
    case Direction.RIGHT:
      return { x: pos.x + 1, y: pos.y }
  }
}

// --------------- Room entry ---------------

function enterRoom(state: DungeonState, room: Room): DungeonState {
  const floor = currentFloor(state)
  const updatedRooms = floor.rooms.map((r) =>
    r.id === room.id ? { ...r, visited: true, revealed: true } : r,
  )
  const updatedFloor = { ...floor, rooms: updatedRooms }
  const newFloors = [...state.floors]
  newFloors[state.currentFloorIndex] = updatedFloor

  const visitedNeuronIds = new Set(state.visitedNeuronIds)
  if (room.neuron) visitedNeuronIds.add(room.neuron.id)

  const message = roomEntryMessage(room)

  return {
    ...state,
    floors: newFloors,
    player: {
      ...state.player,
      roomsExplored: state.player.roomsExplored + 1,
    },
    visitedNeuronIds,
    turnLog: [...state.turnLog, logEntry(state.player.turnsElapsed, message, "discovery")],
  }
}

function roomEntryMessage(room: Room): string {
  switch (room.type) {
    case RoomType.LIBRARY:
      return `You enter a library. Ancient knowledge lingers here.`
    case RoomType.NPC_ROOM:
      return `Someone is here. Press E to talk.`
    case RoomType.TRAP_ROOM:
      return `The floor looks unstable...`
    case RoomType.ENEMY_ROOM:
      return `A hostile presence blocks your path!`
    case RoomType.BOSS_ROOM:
      return `A powerful guardian awaits!`
    case RoomType.TREASURE:
      return `You spot something glimmering. Press E to pick up.`
    case RoomType.FORK:
      return `The path splits. A choice must be made.`
    case RoomType.PUZZLE:
      return `Strange patterns cover the walls...`
    case RoomType.SECRET:
      return `You discovered a hidden chamber!`
    default:
      return `You enter a quiet room.`
  }
}

// --------------- Entity encounters ---------------

function entityEncounter(
  state: DungeonState,
  entity: Entity,
  _room: Room | null,
): DungeonState {
  if (entity.type === EntityType.ENEMY || entity.type === EntityType.BOSS) {
    return {
      ...state,
      phase: GamePhase.COMBAT,
      turnLog: [
        ...state.turnLog,
        logEntry(state.player.turnsElapsed, `${entity.name} attacks!`, "combat"),
      ],
    }
  }

  if (entity.type === EntityType.NPC) {
    return {
      ...state,
      phase: GamePhase.DIALOGUE,
    }
  }

  return state
}

// --------------- Interaction ---------------

function handleInteract(state: DungeonState): DungeonState {
  if (state.phase !== GamePhase.EXPLORING) return state

  const floor = currentFloor(state)
  const pos = state.player.position
  const room = findRoomAt(floor, pos)

  if (!room) return state

  // Pick up items
  const itemEntity = room.entities.find(
    (e) => e.type === EntityType.ITEM && !e.defeated,
  )
  if (itemEntity?.item) {
    if (state.player.inventory.length >= 5) {
      return {
        ...state,
        turnLog: [
          ...state.turnLog,
          logEntry(state.player.turnsElapsed, "Inventory full! Use (1-5) or drop (Shift+1-5) an item.", "item"),
        ],
      }
    }

    const updatedRooms = floor.rooms.map((r) => {
      if (r.id !== room.id) return r
      return {
        ...r,
        entities: r.entities.map((e) =>
          e.id === itemEntity.id ? { ...e, defeated: true } : e,
        ),
      }
    })
    const updatedFloor = { ...floor, rooms: updatedRooms }
    const newFloors = [...state.floors]
    newFloors[state.currentFloorIndex] = updatedFloor

    return {
      ...state,
      floors: newFloors,
      player: {
        ...state.player,
        inventory: [...state.player.inventory, itemEntity.item],
        score: state.player.score + 10,
      },
      turnLog: [
        ...state.turnLog,
        logEntry(state.player.turnsElapsed, `Picked up: ${itemEntity.item.name}`, "item"),
      ],
    }
  }

  // Talk to NPC
  const npc = room.entities.find((e) => e.type === EntityType.NPC)
  if (npc) {
    return {
      ...state,
      phase: GamePhase.DIALOGUE,
    }
  }

  return state
}

// --------------- Items ---------------

function handleUseItem(state: DungeonState, index: number): DungeonState {
  const item = state.player.inventory[index]
  if (!item) return state

  let newStats = { ...state.player.stats }
  let message = ""

  switch (item.type) {
    case "heal_potion":
      newStats = {
        ...newStats,
        hp: Math.min(newStats.maxHp, newStats.hp + item.value),
      }
      message = `Used ${item.name}. Restored ${item.value} HP.`
      break
    case "atk_scroll":
      newStats = { ...newStats, atk: newStats.atk + item.value }
      message = `Used ${item.name}. ATK +${item.value}!`
      break
    case "def_scroll":
      newStats = { ...newStats, def: newStats.def + item.value }
      message = `Used ${item.name}. DEF +${item.value}!`
      break
    case "map_reveal": {
      // Reveal all rooms on current floor
      const floor = currentFloor(state)
      const updatedRooms = floor.rooms.map((r) => ({ ...r, revealed: true }))
      const updatedFloor = { ...floor, rooms: updatedRooms }
      const newFloors = [...state.floors]
      newFloors[state.currentFloorIndex] = updatedFloor
      message = `Used ${item.name}. The floor layout is revealed!`

      return {
        ...state,
        floors: newFloors,
        player: {
          ...state.player,
          stats: newStats,
          inventory: state.player.inventory.filter((_, i) => i !== index),
        },
        turnLog: [...state.turnLog, logEntry(state.player.turnsElapsed, message, "item")],
      }
    }
    default:
      return state
  }

  return {
    ...state,
    player: {
      ...state.player,
      stats: newStats,
      inventory: state.player.inventory.filter((_, i) => i !== index),
    },
    turnLog: [...state.turnLog, logEntry(state.player.turnsElapsed, message, "item")],
  }
}

// --------------- Drop item ---------------

function handleDropItem(state: DungeonState, index: number): DungeonState {
  if (state.phase !== GamePhase.EXPLORING) return state
  const item = state.player.inventory[index]
  if (!item) return state

  return {
    ...state,
    player: {
      ...state.player,
      inventory: state.player.inventory.filter((_, i) => i !== index),
      score: state.player.score + DROP_SCORE_BONUS,
    },
    turnLog: [
      ...state.turnLog,
      logEntry(state.player.turnsElapsed, `Recycled: ${item.name} (+${DROP_SCORE_BONUS} score)`, "item"),
    ],
  }
}

// --------------- Floor transition ---------------

function handleNextFloor(state: DungeonState): DungeonState {
  const floor = currentFloor(state)
  const pos = state.player.position
  const tile = floor.tiles[pos.y]![pos.x]!

  if (tile !== TileType.STAIRS_DOWN) return state

  // Calculate floor rating
  const floorResult = calculateFloorRating(floor, state)

  if (state.currentFloorIndex >= state.floors.length - 1) {
    // Last floor — victory!
    return {
      ...state,
      phase: GamePhase.VICTORY,
      player: {
        ...state.player,
        score: state.player.score + floorResult.scoreEarned,
      },
      floorResults: [...state.floorResults, floorResult],
      turnLog: [
        ...state.turnLog,
        logEntry(state.player.turnsElapsed, `Floor ${floorResult.rating}! +${floorResult.scoreEarned} score. You conquered the dungeon!`, "event"),
      ],
    }
  }

  const nextFloor = state.floors[state.currentFloorIndex + 1]!
  return {
    ...state,
    currentFloorIndex: state.currentFloorIndex + 1,
    player: {
      ...state.player,
      position: { ...nextFloor.playerStart },
      score: state.player.score + floorResult.scoreEarned,
    },
    chain: createChain(), // reset chain for new floor
    danger: resetDangerForFloor(),
    floorResults: [...state.floorResults, floorResult],
    phase: GamePhase.FLOOR_COMPLETE,
    turnLog: [
      ...state.turnLog,
      logEntry(state.player.turnsElapsed, `Floor rated ${floorResult.rating}! +${floorResult.scoreEarned} score. Descending...`, "event"),
    ],
  }
}

// --------------- Fog of war ---------------

function revealFog(state: DungeonState): DungeonState {
  const pos = state.player.position
  const floorIdx = state.currentFloorIndex
  const newExplored = new Set(state.exploredTiles)

  for (let dy = -FOG_RADIUS; dy <= FOG_RADIUS; dy++) {
    for (let dx = -FOG_RADIUS; dx <= FOG_RADIUS; dx++) {
      if (dx * dx + dy * dy <= FOG_RADIUS * FOG_RADIUS) {
        newExplored.add(`${pos.x + dx},${pos.y + dy},${floorIdx}`)
      }
    }
  }

  return { ...state, exploredTiles: newExplored }
}

// --------------- Helpers ---------------

export function currentFloor(state: DungeonState): DungeonFloor {
  return state.floors[state.currentFloorIndex]!
}

function findRoomAt(floor: DungeonFloor, pos: Position): Room | null {
  return (
    floor.rooms.find(
      (r) =>
        pos.x >= r.rect.x &&
        pos.x < r.rect.x + r.rect.w &&
        pos.y >= r.rect.y &&
        pos.y < r.rect.y + r.rect.h,
    ) ?? null
  )
}

function findEntityAt(floor: DungeonFloor, pos: Position): Entity | null {
  for (const room of floor.rooms) {
    for (const entity of room.entities) {
      if (entity.position.x === pos.x && entity.position.y === pos.y && !entity.defeated) {
        return entity
      }
    }
  }
  return null
}

function logEntry(
  turn: number,
  message: string,
  type: TurnLogEntry["type"],
): TurnLogEntry {
  return { turn, message, type }
}

function advanceTurn(state: DungeonState, message: string): DungeonState {
  return {
    ...state,
    player: {
      ...state.player,
      turnsElapsed: state.player.turnsElapsed + 1,
    },
    turnLog: [...state.turnLog, logEntry(state.player.turnsElapsed, message, "move")],
  }
}

/** Find entity in combat range (current room) */
export function getActiveEnemy(state: DungeonState): Entity | null {
  const floor = currentFloor(state)
  const room = findRoomAt(floor, state.player.position)
  if (!room) return null
  return (
    room.entities.find(
      (e) =>
        (e.type === EntityType.ENEMY || e.type === EntityType.BOSS) &&
        !e.defeated,
    ) ?? null
  )
}

/** Mark enemy as defeated in floor state */
export function defeatEnemy(state: DungeonState, entityId: string): DungeonState {
  const floor = currentFloor(state)
  const updatedRooms = floor.rooms.map((r) => ({
    ...r,
    entities: r.entities.map((e) =>
      e.id === entityId ? { ...e, defeated: true } : e,
    ),
  }))
  const updatedFloor = { ...floor, rooms: updatedRooms }
  const newFloors = [...state.floors]
  newFloors[state.currentFloorIndex] = updatedFloor

  const defeatedNeuronIds = new Set(state.defeatedNeuronIds)
  const entity = floor.rooms.flatMap((r) => r.entities).find((e) => e.id === entityId)
  if (entity?.neuronId) defeatedNeuronIds.add(entity.neuronId)

  return {
    ...state,
    floors: newFloors,
    player: {
      ...state.player,
      enemiesDefeated: state.player.enemiesDefeated + 1,
      score: state.player.score + 50,
    },
    phase: GamePhase.EXPLORING,
    defeatedNeuronIds,
    turnLog: [
      ...state.turnLog,
      logEntry(state.player.turnsElapsed, `Defeated ${entity?.name ?? "enemy"}!`, "combat"),
    ],
  }
}
