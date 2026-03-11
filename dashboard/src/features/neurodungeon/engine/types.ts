/* ------------------------------------------------------------------ */
/*  Neurodungeon — Game types & constants                             */
/*  Roguelike dungeon generated from Neural Memory brain data         */
/* ------------------------------------------------------------------ */

import type { GraphNeuron, GraphSynapse, GraphFiber } from "@/api/types"

// --------------- Constants ---------------

export const TILE_SIZE = 16
export const MAP_WIDTH = 80
export const MAP_HEIGHT = 50
export const FOG_RADIUS = 7
export const ROOM_MIN_SIZE = 3
export const ROOM_MAX_SIZE = 7
export const CORRIDOR_WIDTH = 1
export const MAX_ITEMS = 5
export const PLAYER_BASE_HP = 100
export const PLAYER_BASE_ATK = 10
export const PLAYER_BASE_DEF = 5

// --------------- Value objects (TS 5.9 erasableSyntaxOnly compat) --

export const TileType = {
  VOID: 0,
  WALL: 1,
  FLOOR: 2,
  CORRIDOR: 3,
  DOOR: 4,
  STAIRS_DOWN: 5,
  STAIRS_UP: 6,
  TRAP: 7,
  CHEST: 8,
  WATER: 9,
} as const
export type TileType = (typeof TileType)[keyof typeof TileType]

export const EntityType = {
  PLAYER: "player",
  ENEMY: "enemy",
  NPC: "npc",
  ITEM: "item",
  BOSS: "boss",
} as const
export type EntityType = (typeof EntityType)[keyof typeof EntityType]

export const RoomType = {
  LIBRARY: "library",
  NPC_ROOM: "npc_room",
  TRAP_ROOM: "trap_room",
  ENEMY_ROOM: "enemy_room",
  TREASURE: "treasure",
  FORK: "fork",
  PUZZLE: "puzzle",
  EMPTY: "empty",
  SECRET: "secret",
  BOSS_ROOM: "boss_room",
  STAIRS: "stairs",
} as const
export type RoomType = (typeof RoomType)[keyof typeof RoomType]

export const ItemType = {
  HEAL_POTION: "heal_potion",
  ATK_SCROLL: "atk_scroll",
  DEF_SCROLL: "def_scroll",
  MAP_REVEAL: "map_reveal",
  KEY: "key",
  SHIELD: "shield",
} as const
export type ItemType = (typeof ItemType)[keyof typeof ItemType]

export const Direction = {
  UP: "up",
  DOWN: "down",
  LEFT: "left",
  RIGHT: "right",
} as const
export type Direction = (typeof Direction)[keyof typeof Direction]

export const GamePhase = {
  START: "start",
  EXPLORING: "exploring",
  COMBAT: "combat",
  DIALOGUE: "dialogue",
  QUIZ: "quiz",
  FLOOR_COMPLETE: "floor_complete",
  GAME_OVER: "game_over",
  VICTORY: "victory",
} as const
export type GamePhase = (typeof GamePhase)[keyof typeof GamePhase]

export const WorldEventType = {
  ZOMBIE_HORDE: "zombie_horde",
  CORRIDOR_COLLAPSE: "corridor_collapse",
  FOG_THICKEN: "fog_thicken",
  ROOM_DECAY: "room_decay",
  BOSS_SPAWN: "boss_spawn",
  POISON_TRAPS: "poison_traps",
} as const
export type WorldEventType = (typeof WorldEventType)[keyof typeof WorldEventType]

// --------------- Neuron -> Room mapping ---------------

export const NEURON_TYPE_TO_ROOM: Record<string, RoomType> = {
  concept: RoomType.LIBRARY,
  entity: RoomType.NPC_ROOM,
  action: RoomType.TRAP_ROOM,
  error: RoomType.ENEMY_ROOM,
  insight: RoomType.TREASURE,
  decision: RoomType.FORK,
  pattern: RoomType.PUZZLE,
  reference: RoomType.EMPTY,
  temporal: RoomType.EMPTY,
  spatial: RoomType.EMPTY,
  state: RoomType.EMPTY,
}

// --------------- Core types ---------------

export interface Position {
  x: number
  y: number
}

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export interface Room {
  id: string
  rect: Rect
  center: Position
  type: RoomType
  neuron: GraphNeuron | null
  visited: boolean
  revealed: boolean
  entities: Entity[]
  connectedRoomIds: string[]
  lighting: number // 0.0 (dark) - 1.0 (bright), from neuron activation
}

export interface Corridor {
  from: string // room id
  to: string   // room id
  tiles: Position[]
  weight: number // synapse weight — affects width/stability
  synapseId: string
}

export interface Entity {
  id: string
  type: EntityType
  position: Position
  name: string
  content: string // neuron content for dialogue/lore
  stats: CombatStats | null
  item: Item | null
  defeated: boolean
  neuronId: string | null
}

export interface CombatStats {
  hp: number
  maxHp: number
  atk: number
  def: number
  spd: number
  neuronType: string // for type advantage
}

export interface Item {
  id: string
  type: ItemType
  name: string
  description: string
  value: number // heal amount, buff amount, etc.
  neuronId: string | null
}

export interface ActiveBuff {
  type: "atk" | "def"
  value: number
  turnsRemaining: number
}

export interface Player {
  position: Position
  stats: CombatStats
  inventory: Item[]
  buffs: ActiveBuff[]
  shieldActive: boolean  // one-hit shield from Shield Potion
  score: number
  roomsExplored: number
  enemiesDefeated: number
  turnsElapsed: number
}

export interface DungeonFloor {
  id: string // fiber id
  name: string // fiber summary
  depth: number // floor number (0-indexed)
  width: number
  height: number
  tiles: TileType[][]
  rooms: Room[]
  corridors: Corridor[]
  playerStart: Position
  stairsDown: Position | null
  stairsUp: Position | null
}

// --------------- Memory Chain (synapse-based combo system) --------

export interface MemoryChain {
  currentNeuronId: string | null  // last visited neuron
  chainLength: number             // consecutive connected visits
  maxChain: number                // best chain this run
  multiplier: number              // score multiplier (1 + chainLength * 0.5)
}

// --------------- Danger Escalation ---------------

export interface DangerLevel {
  level: number         // 0-10, increases over time
  turnsOnFloor: number  // turns spent on current floor
  escalationRate: number // turns per danger level increase (default 15)
}

// --------------- Floor Rating ---------------

export const FloorRating = {
  S: "S",
  A: "A",
  B: "B",
  C: "C",
  D: "D",
} as const
export type FloorRating = (typeof FloorRating)[keyof typeof FloorRating]

export interface FloorResult {
  floorName: string
  rating: FloorRating
  explorationPct: number
  killPct: number
  chainBonus: number
  turnsUsed: number
  scoreEarned: number
}

export interface DungeonState {
  floors: DungeonFloor[]
  currentFloorIndex: number
  player: Player
  phase: GamePhase
  exploredTiles: Set<string> // "x,y,floor" keys
  visitedNeuronIds: Set<string>
  defeatedNeuronIds: Set<string>
  turnLog: TurnLogEntry[]
  activeEvent: WorldEvent | null
  chain: MemoryChain
  danger: DangerLevel
  floorResults: FloorResult[]  // completed floor ratings
  killStreak: number           // consecutive kills without taking damage
}

export interface TurnLogEntry {
  turn: number
  message: string
  type: "move" | "combat" | "item" | "event" | "discovery"
}

// --------------- World events (from health data) ---------------

export interface WorldEvent {
  type: WorldEventType
  message: string
  severity: number // 1-3
}

// --------------- Dungeon generation input ---------------

export interface DungeonGenInput {
  neurons: GraphNeuron[]
  synapses: GraphSynapse[]
  fibers: GraphFiber[]
  fiberId: string // which fiber to generate floor from
  floorDepth: number
  seed: number
}

export interface DungeonGenResult {
  floor: DungeonFloor
  entityCount: number
  roomCount: number
}

// --------------- Game actions ---------------

export type GameAction =
  | { type: "move"; direction: Direction }
  | { type: "interact" }
  | { type: "use_item"; itemIndex: number }
  | { type: "drop_item"; itemIndex: number }
  | { type: "attack" }
  | { type: "defend" }
  | { type: "flee" }
  | { type: "answer_quiz"; correct: boolean }
  | { type: "next_floor" }
  | { type: "wait" }

// --------------- Save data (localStorage) ---------------

export interface SaveData {
  brainName: string
  dungeonState: Omit<DungeonState, "exploredTiles" | "visitedNeuronIds" | "defeatedNeuronIds"> & {
    exploredTiles: string[]
    visitedNeuronIds: string[]
    defeatedNeuronIds: string[]
  }
  highScore: number
  achievements: string[]
  savedAt: string
}
