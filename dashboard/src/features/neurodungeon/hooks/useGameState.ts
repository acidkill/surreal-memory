/* ------------------------------------------------------------------ */
/*  Zustand store for Neurodungeon game state                         */
/* ------------------------------------------------------------------ */

import { create } from "zustand"
import type { DungeonState, GameAction, DungeonFloor } from "../engine/types"
import { GamePhase } from "../engine/types"
import { processTurn } from "../engine/game-loop"
import { createPlayer } from "../engine/dungeon-gen"
import { createChain, createDanger } from "../engine/engagement"

interface GameStore {
  /** Current dungeon state (null = not started) */
  dungeon: DungeonState | null

  /** Initialize a new game with generated floors */
  startGame: (floors: DungeonFloor[]) => void

  /** Process a game action (move, interact, etc.) */
  dispatch: (action: GameAction) => void

  /** Return to exploring phase (after dialogue/combat) */
  resumeExploring: () => void

  /** Update dungeon state directly (for combat resolution) */
  updateState: (updater: (state: DungeonState) => DungeonState) => void

  /** Reset game */
  reset: () => void
}

export const useGameStore = create<GameStore>((set) => ({
  dungeon: null,

  startGame: (floors) => {
    if (floors.length === 0) return
    const firstFloor = floors[0]!
    const player = createPlayer(firstFloor.playerStart)

    const initialState: DungeonState = {
      floors,
      currentFloorIndex: 0,
      player,
      phase: GamePhase.EXPLORING,
      exploredTiles: new Set<string>(),
      visitedNeuronIds: new Set<string>(),
      defeatedNeuronIds: new Set<string>(),
      turnLog: [{ turn: 0, message: `Entering ${firstFloor.name}...`, type: "event" }],
      activeEvent: null,
      chain: createChain(),
      danger: createDanger(),
      floorResults: [],
      killStreak: 0,
    }

    // Reveal initial fog
    const revealed = new Set<string>()
    const { x: px, y: py } = player.position
    const radius = 7
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy <= radius * radius) {
          revealed.add(`${px + dx},${py + dy},0`)
        }
      }
    }

    // Mark starting room as visited
    const startRoom = firstFloor.rooms.find(
      (r) =>
        px >= r.rect.x &&
        px < r.rect.x + r.rect.w &&
        py >= r.rect.y &&
        py < r.rect.y + r.rect.h,
    )
    if (startRoom) {
      const updatedRooms = firstFloor.rooms.map((r) =>
        r.id === startRoom.id ? { ...r, visited: true, revealed: true } : r,
      )
      const updatedFloors = [{ ...firstFloor, rooms: updatedRooms }, ...floors.slice(1)]
      const visitedNeurons = new Set<string>()
      if (startRoom.neuron) visitedNeurons.add(startRoom.neuron.id)

      set({
        dungeon: {
          ...initialState,
          floors: updatedFloors,
          exploredTiles: revealed,
          visitedNeuronIds: visitedNeurons,
          player: { ...player, roomsExplored: 1 },
        },
      })
      return
    }

    set({ dungeon: { ...initialState, exploredTiles: revealed } })
  },

  dispatch: (action) => {
    set((store) => {
      if (!store.dungeon) return store
      if (
        store.dungeon.phase === GamePhase.GAME_OVER ||
        store.dungeon.phase === GamePhase.VICTORY
      ) {
        return store
      }
      return { dungeon: processTurn(store.dungeon, action) }
    })
  },

  resumeExploring: () => {
    set((store) => {
      if (!store.dungeon) return store
      return { dungeon: { ...store.dungeon, phase: GamePhase.EXPLORING } }
    })
  },

  updateState: (updater) => {
    set((store) => {
      if (!store.dungeon) return store
      return { dungeon: updater(store.dungeon) }
    })
  },

  reset: () => set({ dungeon: null }),
}))
