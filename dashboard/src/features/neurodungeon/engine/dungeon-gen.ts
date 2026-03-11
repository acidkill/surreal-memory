/* ------------------------------------------------------------------ */
/*  Dungeon generator — transforms NM graph data into a playable map  */
/*                                                                    */
/*  Algorithm:                                                        */
/*  1. Filter neurons/synapses for target fiber                       */
/*  2. Place rooms on grid (BSP-lite placement)                       */
/*  3. Connect rooms via A* corridors following synapse edges         */
/*  4. Populate rooms with entities based on neuron type              */
/*  5. Place stairs, assign lighting from activation                  */
/* ------------------------------------------------------------------ */

import type {
  DungeonGenInput,
  DungeonGenResult,
  DungeonFloor,
  Room,
  Corridor,
  Entity,
  Player,
  Position,
  Rect,
  CombatStats,
} from "./types"
import {
  TileType,
  RoomType,
  EntityType,
  NEURON_TYPE_TO_ROOM,
  MAP_WIDTH,
  MAP_HEIGHT,
  ROOM_MIN_SIZE,
  ROOM_MAX_SIZE,
  PLAYER_BASE_HP,
  PLAYER_BASE_ATK,
  PLAYER_BASE_DEF,
} from "./types"
import type { GraphNeuron, GraphSynapse } from "@/api/types"
import { findPath } from "../utils/pathfinding"
import { generateItem } from "./items"
import { createRng, randInt, shuffle } from "../utils/noise"

// Max rooms per floor (avoid overcrowding)
const MAX_ROOMS = 25
// Padding between rooms
const ROOM_PADDING = 2

export function generateDungeon(input: DungeonGenInput): DungeonGenResult {
  const rng = createRng(input.seed)

  // 1. Filter data for this fiber
  const { neurons, synapses } = filterByFiber(input)

  // 2. Initialize tile grid (all walls)
  const tiles = createGrid(MAP_WIDTH, MAP_HEIGHT, TileType.WALL)

  // 3. Place rooms
  const rooms = placeRooms(neurons, rng, MAP_WIDTH, MAP_HEIGHT)

  // 4. Carve rooms into tile grid
  for (const room of rooms) {
    carveRect(tiles, room.rect)
  }

  // 5. Connect rooms with corridors
  const corridors = connectRooms(rooms, synapses, tiles, MAP_WIDTH, MAP_HEIGHT, rng)

  // 6. Populate rooms with entities
  populateRooms(rooms, rng)

  // 7. Place stairs
  const stairsDown = rooms.length > 1 ? rooms[rooms.length - 1]!.center : null
  if (stairsDown) {
    tiles[stairsDown.y]![stairsDown.x] = TileType.STAIRS_DOWN
  }

  const stairsUp = input.floorDepth > 0 ? rooms[0]!.center : null
  if (stairsUp && input.floorDepth > 0) {
    tiles[stairsUp.y]![stairsUp.x] = TileType.STAIRS_UP
  }

  // 8. Player starts in first room
  const playerStart = rooms[0]!.center

  const floor: DungeonFloor = {
    id: input.fiberId,
    name: input.fibers.find((f) => f.id === input.fiberId)?.summary ?? `Floor ${input.floorDepth + 1}`,
    depth: input.floorDepth,
    width: MAP_WIDTH,
    height: MAP_HEIGHT,
    tiles,
    rooms,
    corridors,
    playerStart,
    stairsDown,
    stairsUp,
  }

  return {
    floor,
    entityCount: rooms.reduce((sum, r) => sum + r.entities.length, 0),
    roomCount: rooms.length,
  }
}

// --------------- Internal helpers ---------------

function filterByFiber(input: DungeonGenInput): {
  neurons: GraphNeuron[]
  synapses: GraphSynapse[]
} {
  // If fiber has few neurons, use all available (small brain fallback)
  const fiberNeuronIds = new Set<string>()

  // Use all neurons (fiber → neuron mapping not in graph response)
  const neurons = input.neurons.slice(0, MAX_ROOMS)
  for (const n of neurons) fiberNeuronIds.add(n.id)

  const synapses = input.synapses.filter(
    (s) => fiberNeuronIds.has(s.source_id) && fiberNeuronIds.has(s.target_id),
  )

  return { neurons, synapses }
}

function createGrid(width: number, height: number, fill: TileType): TileType[][] {
  return Array.from({ length: height }, () => Array.from({ length: width }, () => fill))
}

function placeRooms(
  neurons: GraphNeuron[],
  rng: () => number,
  mapW: number,
  mapH: number,
): Room[] {
  const rooms: Room[] = []
  const capped = neurons.slice(0, MAX_ROOMS)

  // Shuffle neurons for varied placement each seed
  const shuffled = shuffle(rng, [...capped])

  for (const neuron of shuffled) {
    // Try to place room without overlap
    let placed = false
    for (let attempt = 0; attempt < 30; attempt++) {
      const w = randInt(rng, ROOM_MIN_SIZE, ROOM_MAX_SIZE)
      const h = randInt(rng, ROOM_MIN_SIZE, ROOM_MAX_SIZE)
      const x = randInt(rng, 2, mapW - w - 2)
      const y = randInt(rng, 2, mapH - h - 2)
      const rect: Rect = { x, y, w, h }

      if (!overlapsAny(rect, rooms, ROOM_PADDING)) {
        const roomType = resolveRoomType(neuron)
        const activation = extractActivation(neuron)

        rooms.push({
          id: neuron.id,
          rect,
          center: { x: Math.floor(x + w / 2), y: Math.floor(y + h / 2) },
          type: roomType,
          neuron,
          visited: false,
          revealed: false,
          entities: [],
          connectedRoomIds: [],
          lighting: activation,
        })
        placed = true
        break
      }
    }

    // If we can't place after 30 attempts, skip this neuron
    if (!placed && rooms.length >= 5) continue
  }

  // Ensure at least a few rooms exist (fallback for tiny brains)
  if (rooms.length < 3) {
    for (let i = rooms.length; i < 3; i++) {
      const w = randInt(rng, ROOM_MIN_SIZE, ROOM_MAX_SIZE)
      const h = randInt(rng, ROOM_MIN_SIZE, ROOM_MAX_SIZE)
      const x = randInt(rng, 2, mapW - w - 2)
      const y = randInt(rng, 2, mapH - h - 2)
      rooms.push({
        id: `filler-${i}`,
        rect: { x, y, w, h },
        center: { x: Math.floor(x + w / 2), y: Math.floor(y + h / 2) },
        type: RoomType.EMPTY,
        neuron: null,
        visited: false,
        revealed: false,
        entities: [],
        connectedRoomIds: [],
        lighting: 0.5,
      })
    }
  }

  return rooms
}

function overlapsAny(rect: Rect, rooms: Room[], padding: number): boolean {
  return rooms.some((r) => {
    const a = r.rect
    return !(
      rect.x + rect.w + padding <= a.x ||
      a.x + a.w + padding <= rect.x ||
      rect.y + rect.h + padding <= a.y ||
      a.y + a.h + padding <= rect.y
    )
  })
}

function resolveRoomType(neuron: GraphNeuron): RoomType {
  const base = NEURON_TYPE_TO_ROOM[neuron.type] ?? RoomType.EMPTY

  // Promote high-priority errors to boss rooms
  const priority = (neuron.metadata?.priority as number) ?? 5
  if (base === RoomType.ENEMY_ROOM && priority >= 8) {
    return RoomType.BOSS_ROOM
  }

  return base
}

function extractActivation(neuron: GraphNeuron): number {
  const activation = neuron.metadata?.activation_level
  if (typeof activation === "number") return Math.max(0, Math.min(1, activation))
  return 0.5
}

function carveRect(tiles: TileType[][], rect: Rect): void {
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    for (let x = rect.x; x < rect.x + rect.w; x++) {
      if (tiles[y]?.[x] !== undefined) {
        tiles[y]![x] = TileType.FLOOR
      }
    }
  }
}

function connectRooms(
  rooms: Room[],
  synapses: GraphSynapse[],
  tiles: TileType[][],
  width: number,
  height: number,
  _rng: () => number,
): Corridor[] {
  const corridors: Corridor[] = []
  const roomById = new Map(rooms.map((r) => [r.id, r]))
  const connected = new Set<string>()

  // First: connect rooms that have actual synapses
  for (const synapse of synapses) {
    const from = roomById.get(synapse.source_id)
    const to = roomById.get(synapse.target_id)
    if (!from || !to) continue

    const pairKey = [from.id, to.id].sort().join(":")
    if (connected.has(pairKey)) continue

    const path = findPath(tiles, from.center, to.center, width, height)
    carveCorridor(tiles, path)

    from.connectedRoomIds.push(to.id)
    to.connectedRoomIds.push(from.id)
    connected.add(pairKey)

    corridors.push({
      from: from.id,
      to: to.id,
      tiles: path,
      weight: synapse.weight,
      synapseId: synapse.id,
    })
  }

  // Second: ensure all rooms are reachable (MST fallback)
  // Connect each unconnected room to its nearest connected room
  if (rooms.length > 1) {
    const connectedRooms = new Set<string>()
    connectedRooms.add(rooms[0]!.id)

    // Mark rooms already connected via synapses
    for (const c of corridors) {
      connectedRooms.add(c.from)
      connectedRooms.add(c.to)
    }
    if (connectedRooms.size === 0) connectedRooms.add(rooms[0]!.id)

    for (const room of rooms) {
      if (connectedRooms.has(room.id)) continue

      // Find nearest connected room
      let nearest: Room | null = null
      let nearestDist = Infinity
      for (const rid of connectedRooms) {
        const other = roomById.get(rid)
        if (!other) continue
        const dist = Math.abs(room.center.x - other.center.x) + Math.abs(room.center.y - other.center.y)
        if (dist < nearestDist) {
          nearestDist = dist
          nearest = other
        }
      }

      if (nearest) {
        const path = findPath(tiles, room.center, nearest.center, width, height)
        carveCorridor(tiles, path)
        room.connectedRoomIds.push(nearest.id)
        nearest.connectedRoomIds.push(room.id)
        connectedRooms.add(room.id)

        corridors.push({
          from: room.id,
          to: nearest.id,
          tiles: path,
          weight: 0.3,
          synapseId: `mst-${room.id}`,
        })
      }
    }
  }

  return corridors
}

function carveCorridor(tiles: TileType[][], path: Position[]): void {
  for (const pos of path) {
    if (tiles[pos.y]?.[pos.x] === TileType.WALL) {
      tiles[pos.y]![pos.x] = TileType.CORRIDOR
    }
  }
}

function populateRooms(rooms: Room[], rng: () => number): void {
  for (const room of rooms) {
    switch (room.type) {
      case RoomType.ENEMY_ROOM:
      case RoomType.BOSS_ROOM:
        room.entities.push(createEnemy(room, rng))
        break
      case RoomType.TREASURE:
      case RoomType.FORK:
      case RoomType.LIBRARY:
        room.entities.push(createItemEntity(room))
        break
      case RoomType.NPC_ROOM:
        room.entities.push(createNPC(room))
        break
      case RoomType.TRAP_ROOM:
        placeTrap(room)
        break
      default:
        break
    }
  }
}

function createEnemy(room: Room, rng: () => number): Entity {
  const neuron = room.neuron
  const content = neuron?.content ?? "Unknown threat"
  const contentLen = content.length
  const activation = extractActivation(neuron ?? { id: "", type: "", content: "", metadata: {} })
  const isBoss = room.type === RoomType.BOSS_ROOM
  const multiplier = isBoss ? 2 : 1

  const stats: CombatStats = {
    hp: Math.max(10, Math.floor(contentLen / 10) * multiplier),
    maxHp: Math.max(10, Math.floor(contentLen / 10) * multiplier),
    atk: Math.max(3, Math.floor(activation * 10 * multiplier)),
    def: Math.max(1, (neuron?.metadata?.synapse_count as number ?? 2) * multiplier),
    spd: Math.max(1, Math.floor((1 / (activation + 0.1)) * 2)),
    neuronType: neuron?.type ?? "error",
  }

  const offset = randInt(rng, 0, room.rect.w - 1)
  return {
    id: `enemy-${room.id}`,
    type: isBoss ? EntityType.BOSS : EntityType.ENEMY,
    position: { x: room.rect.x + offset, y: room.rect.y + Math.floor(room.rect.h / 2) },
    name: isBoss ? `Boss: ${truncate(content, 30)}` : truncate(content, 30),
    content,
    stats,
    item: null,
    defeated: false,
    neuronId: neuron?.id ?? null,
  }
}

function createNPC(room: Room): Entity {
  const neuron = room.neuron
  return {
    id: `npc-${room.id}`,
    type: EntityType.NPC,
    position: { ...room.center },
    name: truncate(neuron?.content ?? "Stranger", 30),
    content: neuron?.content ?? "...",
    stats: null,
    item: null,
    defeated: false,
    neuronId: neuron?.id ?? null,
  }
}

function createItemEntity(room: Room): Entity {
  const neuron = room.neuron
  const item = generateItem(room, neuron)

  return {
    id: `item-entity-${room.id}`,
    type: EntityType.ITEM,
    position: { ...room.center },
    name: item.name,
    content: neuron?.content ?? "",
    stats: null,
    item,
    defeated: false,
    neuronId: neuron?.id ?? null,
  }
}

function placeTrap(room: Room): void {
  // Mark center tile as trap
  // (actual trap tile set during render — room.type is enough)
  // Entity not needed, trap is tile-based
  void room
}

/** Create initial player stats */
export function createPlayer(startPos: Position): Player {
  return {
    position: { ...startPos },
    stats: {
      hp: PLAYER_BASE_HP,
      maxHp: PLAYER_BASE_HP,
      atk: PLAYER_BASE_ATK,
      def: PLAYER_BASE_DEF,
      spd: 5,
      neuronType: "player",
    },
    inventory: [],
    buffs: [],
    shieldActive: false,
    score: 0,
    roomsExplored: 0,
    enemiesDefeated: 0,
    turnsElapsed: 0,
  }
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max - 1) + "\u2026" : str
}
