/* ------------------------------------------------------------------ */
/*  Tile rendering — colors and glyphs for each tile type             */
/*  Dark theme palette matching NM dashboard                          */
/* ------------------------------------------------------------------ */

import { TileType, RoomType, EntityType } from "../engine/types"

// Tile colors (dark theme)
export const TILE_COLORS: Record<TileType, string> = {
  [TileType.VOID]: "#0c1419",
  [TileType.WALL]: "#1a2332",
  [TileType.FLOOR]: "#2a3a4a",
  [TileType.CORRIDOR]: "#232f3e",
  [TileType.DOOR]: "#3a4f5f",
  [TileType.STAIRS_DOWN]: "#2196f3",
  [TileType.STAIRS_UP]: "#4caf50",
  [TileType.TRAP]: "#ff6b6b",
  [TileType.CHEST]: "#ffd700",
  [TileType.WATER]: "#1565c0",
}

// Fog color overlay (unexplored)
export const FOG_COLOR = "#0c1419"
export const FOG_EXPLORED_ALPHA = 0.4 // dimmed but visible
export const FOG_VISIBLE_ALPHA = 0.0 // fully visible

// Room type accent colors (tint on floor tiles)
export const ROOM_TINT: Record<RoomType, string> = {
  [RoomType.LIBRARY]: "rgba(100, 181, 246, 0.15)",
  [RoomType.NPC_ROOM]: "rgba(129, 199, 132, 0.15)",
  [RoomType.TRAP_ROOM]: "rgba(255, 107, 107, 0.15)",
  [RoomType.ENEMY_ROOM]: "rgba(239, 83, 80, 0.2)",
  [RoomType.TREASURE]: "rgba(255, 215, 0, 0.15)",
  [RoomType.FORK]: "rgba(171, 71, 188, 0.15)",
  [RoomType.PUZZLE]: "rgba(255, 167, 38, 0.15)",
  [RoomType.EMPTY]: "rgba(255, 255, 255, 0.03)",
  [RoomType.SECRET]: "rgba(0, 230, 118, 0.15)",
  [RoomType.BOSS_ROOM]: "rgba(244, 67, 54, 0.25)",
  [RoomType.STAIRS]: "rgba(33, 150, 243, 0.15)",
}

// Entity colors
export const ENTITY_COLORS: Record<EntityType, string> = {
  [EntityType.PLAYER]: "#00d084",
  [EntityType.ENEMY]: "#ff6b6b",
  [EntityType.NPC]: "#64b5f6",
  [EntityType.ITEM]: "#ffd700",
  [EntityType.BOSS]: "#f44336",
}

// Entity glyphs (drawn as text on canvas)
export const ENTITY_GLYPHS: Record<EntityType, string> = {
  [EntityType.PLAYER]: "@",
  [EntityType.ENEMY]: "e",
  [EntityType.NPC]: "?",
  [EntityType.ITEM]: "*",
  [EntityType.BOSS]: "B",
}

// Tile glyphs (fallback for very small tiles)
export const TILE_GLYPHS: Record<TileType, string> = {
  [TileType.VOID]: " ",
  [TileType.WALL]: "#",
  [TileType.FLOOR]: ".",
  [TileType.CORRIDOR]: ".",
  [TileType.DOOR]: "+",
  [TileType.STAIRS_DOWN]: ">",
  [TileType.STAIRS_UP]: "<",
  [TileType.TRAP]: "^",
  [TileType.CHEST]: "$",
  [TileType.WATER]: "~",
}
