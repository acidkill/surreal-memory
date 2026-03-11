/* ------------------------------------------------------------------ */
/*  Game Canvas — 2D tile renderer with fog of war + camera follow    */
/* ------------------------------------------------------------------ */

import { useRef, useEffect, useCallback } from "react"
import type { DungeonState } from "../engine/types"
import { TileType, EntityType, TILE_SIZE, FOG_RADIUS } from "../engine/types"
import { currentFloor } from "../engine/game-loop"
import {
  TILE_COLORS,
  FOG_COLOR,
  ROOM_TINT,
  ENTITY_COLORS,
  ENTITY_GLYPHS,
} from "./tiles"
import type { EffectsState } from "./effects"
import { tickEffects, renderParticles, renderFlash } from "./effects"

interface GameCanvasProps {
  state: DungeonState
  width: number
  height: number
  fogModifier?: number
  effects?: EffectsState
  onEffectsTick?: (effects: EffectsState) => void
}

export function GameCanvas({ state, width, height, fogModifier = 0, effects, onEffectsTick }: GameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const floor = currentFloor(state)
    const player = state.player
    const floorIdx = state.currentFloorIndex

    // Fog radius (modified by world events)
    const effectiveRadius = Math.max(3, FOG_RADIUS + fogModifier)

    // Camera: center on player (with screen shake offset)
    const shakeX = effects?.shake?.dx ?? 0
    const shakeY = effects?.shake?.dy ?? 0
    const camX = player.position.x * TILE_SIZE - width / 2 + shakeX
    const camY = player.position.y * TILE_SIZE - height / 2 + shakeY

    // Clear
    ctx.fillStyle = FOG_COLOR
    ctx.fillRect(0, 0, width, height)

    // Visible tile range (only render what's on screen)
    const startCol = Math.max(0, Math.floor(camX / TILE_SIZE) - 1)
    const endCol = Math.min(floor.width, Math.ceil((camX + width) / TILE_SIZE) + 1)
    const startRow = Math.max(0, Math.floor(camY / TILE_SIZE) - 1)
    const endRow = Math.min(floor.height, Math.ceil((camY + height) / TILE_SIZE) + 1)

    // Draw tiles
    for (let y = startRow; y < endRow; y++) {
      for (let x = startCol; x < endCol; x++) {
        const screenX = x * TILE_SIZE - camX
        const screenY = y * TILE_SIZE - camY

        const tileKey = `${x},${y},${floorIdx}`
        const isExplored = state.exploredTiles.has(tileKey)
        const isVisible = isInFogRadius(x, y, player.position.x, player.position.y, effectiveRadius)

        if (!isExplored) continue // completely hidden

        const tile = floor.tiles[y]![x]!

        // Base tile color
        ctx.fillStyle = TILE_COLORS[tile] ?? TILE_COLORS[TileType.VOID]
        ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE)

        // Room tint
        if (tile === TileType.FLOOR) {
          const room = floor.rooms.find(
            (r) =>
              x >= r.rect.x &&
              x < r.rect.x + r.rect.w &&
              y >= r.rect.y &&
              y < r.rect.y + r.rect.h,
          )
          if (room) {
            ctx.fillStyle = ROOM_TINT[room.type]
            ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE)

            // Lighting variation (darker rooms = lower activation)
            if (room.lighting < 0.4) {
              ctx.fillStyle = `rgba(0, 0, 0, ${0.3 - room.lighting * 0.5})`
              ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE)
            }
          }
        }

        // Stairs indicators
        if (tile === TileType.STAIRS_DOWN || tile === TileType.STAIRS_UP) {
          ctx.fillStyle = TILE_COLORS[tile]
          ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE)
          // Draw arrow glyph
          ctx.fillStyle = "#ffffff"
          ctx.font = `bold ${TILE_SIZE - 2}px monospace`
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(
            tile === TileType.STAIRS_DOWN ? ">" : "<",
            screenX + TILE_SIZE / 2,
            screenY + TILE_SIZE / 2,
          )
        }

        // Fog dimming for explored-but-not-visible tiles
        if (!isVisible) {
          ctx.fillStyle = `rgba(12, 20, 25, 0.5)`
          ctx.fillRect(screenX, screenY, TILE_SIZE, TILE_SIZE)
        }
      }
    }

    // Draw entities
    for (const room of floor.rooms) {
      for (const entity of room.entities) {
        if (entity.defeated) continue
        const { x, y } = entity.position
        const tileKey = `${x},${y},${floorIdx}`
        if (!state.exploredTiles.has(tileKey)) continue

        const isVisible = isInFogRadius(x, y, player.position.x, player.position.y, effectiveRadius)
        if (!isVisible) continue

        const screenX = x * TILE_SIZE - camX
        const screenY = y * TILE_SIZE - camY

        // Entity glyph
        ctx.fillStyle = ENTITY_COLORS[entity.type]
        ctx.font = `bold ${TILE_SIZE - 2}px monospace`
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(
          ENTITY_GLYPHS[entity.type],
          screenX + TILE_SIZE / 2,
          screenY + TILE_SIZE / 2,
        )
      }
    }

    // Draw player
    {
      const px = player.position.x * TILE_SIZE - camX
      const py = player.position.y * TILE_SIZE - camY

      // Glow effect
      ctx.shadowColor = ENTITY_COLORS[EntityType.PLAYER]
      ctx.shadowBlur = 8
      ctx.fillStyle = ENTITY_COLORS[EntityType.PLAYER]
      ctx.font = `bold ${TILE_SIZE}px monospace`
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("@", px + TILE_SIZE / 2, py + TILE_SIZE / 2)
      ctx.shadowBlur = 0
    }

    // Grid lines (subtle)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)"
    ctx.lineWidth = 0.5
    for (let y = startRow; y < endRow; y++) {
      const screenY = y * TILE_SIZE - camY
      ctx.beginPath()
      ctx.moveTo(0, screenY)
      ctx.lineTo(width, screenY)
      ctx.stroke()
    }
    for (let x = startCol; x < endCol; x++) {
      const screenX = x * TILE_SIZE - camX
      ctx.beginPath()
      ctx.moveTo(screenX, 0)
      ctx.lineTo(screenX, height)
      ctx.stroke()
    }
    // Draw particles
    if (effects && effects.particles.length > 0) {
      renderParticles(ctx, effects.particles, camX, camY)
    }

    // Draw screen flash
    if (effects?.flash) {
      renderFlash(ctx, effects.flash, width, height)
    }

    // Tick effects for next frame
    if (effects && onEffectsTick) {
      onEffectsTick(tickEffects(effects))
    }
  }, [state, width, height, fogModifier, effects, onEffectsTick])

  useEffect(() => {
    cancelAnimationFrame(animFrameRef.current)
    animFrameRef.current = requestAnimationFrame(render)
    return () => cancelAnimationFrame(animFrameRef.current)
  }, [render])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        display: "block",
        background: FOG_COLOR,
        imageRendering: "pixelated",
        borderRadius: "8px",
      }}
    />
  )
}

function isInFogRadius(x: number, y: number, px: number, py: number, radius: number = FOG_RADIUS): boolean {
  const dx = x - px
  const dy = y - py
  return dx * dx + dy * dy <= radius * radius
}
