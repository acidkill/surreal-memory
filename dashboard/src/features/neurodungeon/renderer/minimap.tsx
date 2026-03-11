/* ------------------------------------------------------------------ */
/*  Minimap — small overview of explored dungeon floor                */
/* ------------------------------------------------------------------ */

import { useRef, useEffect } from "react"
import type { DungeonState } from "../engine/types"
import { TileType } from "../engine/types"
import { currentFloor } from "../engine/game-loop"
import { ENTITY_COLORS } from "./tiles"
import { EntityType } from "../engine/types"

interface MinimapProps {
  state: DungeonState
  size?: number
}

export function Minimap({ state, size = 160 }: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const floor = currentFloor(state)
    const scale = size / Math.max(floor.width, floor.height)
    const floorIdx = state.currentFloorIndex

    // Clear
    ctx.fillStyle = "rgba(12, 20, 25, 0.9)"
    ctx.fillRect(0, 0, size, size)

    // Draw explored tiles
    for (let y = 0; y < floor.height; y++) {
      for (let x = 0; x < floor.width; x++) {
        const tileKey = `${x},${y},${floorIdx}`
        if (!state.exploredTiles.has(tileKey)) continue

        const tile = floor.tiles[y]![x]!
        if (tile === TileType.WALL || tile === TileType.VOID) continue

        const sx = x * scale
        const sy = y * scale

        ctx.fillStyle =
          tile === TileType.STAIRS_DOWN
            ? "#2196f3"
            : tile === TileType.STAIRS_UP
              ? "#4caf50"
              : "rgba(160, 174, 184, 0.4)"
        ctx.fillRect(sx, sy, Math.max(scale, 1), Math.max(scale, 1))
      }
    }

    // Draw rooms (if visited)
    for (const room of floor.rooms) {
      if (!room.visited) continue
      ctx.strokeStyle = "rgba(160, 174, 184, 0.3)"
      ctx.lineWidth = 0.5
      ctx.strokeRect(
        room.rect.x * scale,
        room.rect.y * scale,
        room.rect.w * scale,
        room.rect.h * scale,
      )
    }

    // Draw player
    const px = state.player.position.x * scale
    const py = state.player.position.y * scale
    ctx.fillStyle = ENTITY_COLORS[EntityType.PLAYER]
    ctx.fillRect(px - 1.5, py - 1.5, 3, 3)
  }, [state, size])

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{
        position: "absolute",
        top: 8,
        right: 8,
        borderRadius: "6px",
        border: "1px solid rgba(255, 255, 255, 0.1)",
        opacity: 0.85,
      }}
    />
  )
}
