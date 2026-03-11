/* ------------------------------------------------------------------ */
/*  Share card — generate PNG summary of a dungeon run                 */
/* ------------------------------------------------------------------ */

import type { DungeonState } from "../engine/types"
import { GamePhase } from "../engine/types"

const CARD_W = 600
const CARD_H = 340

interface RunSummary {
  score: number
  roomsExplored: number
  enemiesDefeated: number
  turnsElapsed: number
  floorsReached: number
  totalFloors: number
  memoriesVisited: number
  isVictory: boolean
}

function extractSummary(state: DungeonState): RunSummary {
  return {
    score: state.player.score,
    roomsExplored: state.player.roomsExplored,
    enemiesDefeated: state.player.enemiesDefeated,
    turnsElapsed: state.player.turnsElapsed,
    floorsReached: state.currentFloorIndex + 1,
    totalFloors: state.floors.length,
    memoriesVisited: state.visitedNeuronIds.size,
    isVictory: state.phase === GamePhase.VICTORY,
  }
}

/** Generate a share card canvas and return as Blob */
export async function generateShareCard(state: DungeonState): Promise<Blob> {
  const summary = extractSummary(state)
  const canvas = document.createElement("canvas")
  canvas.width = CARD_W
  canvas.height = CARD_H
  const ctx = canvas.getContext("2d")!

  // Background
  const grad = ctx.createLinearGradient(0, 0, CARD_W, CARD_H)
  grad.addColorStop(0, "#0c1419")
  grad.addColorStop(1, "#121a20")
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, CARD_W, CARD_H)

  // Border
  ctx.strokeStyle = summary.isVictory ? "#00d084" : "#ff6b6b"
  ctx.lineWidth = 2
  ctx.strokeRect(1, 1, CARD_W - 2, CARD_H - 2)

  // Title
  ctx.font = "bold 28px 'Space Grotesk', sans-serif"
  ctx.textAlign = "center"
  ctx.fillStyle = "#ffffff"
  ctx.fillText("NEURODUNGEON", CARD_W / 2, 44)

  // Result badge
  ctx.font = "bold 16px 'Space Grotesk', sans-serif"
  ctx.fillStyle = summary.isVictory ? "#00d084" : "#ff6b6b"
  ctx.fillText(
    summary.isVictory ? "DUNGEON CONQUERED" : "FALLEN IN THE DEPTHS",
    CARD_W / 2,
    70,
  )

  // Score
  ctx.font = "bold 48px 'JetBrains Mono', monospace"
  ctx.fillStyle = "#ffd700"
  ctx.fillText(String(summary.score), CARD_W / 2, 128)
  ctx.font = "12px 'Inter', sans-serif"
  ctx.fillStyle = "#a0aeb8"
  ctx.fillText("SCORE", CARD_W / 2, 148)

  // Stats grid
  const stats = [
    { label: "Rooms", value: String(summary.roomsExplored), color: "#64b5f6" },
    { label: "Enemies", value: String(summary.enemiesDefeated), color: "#ff6b6b" },
    { label: "Memories", value: String(summary.memoriesVisited), color: "#00d084" },
    { label: "Floors", value: `${summary.floorsReached}/${summary.totalFloors}`, color: "#2196f3" },
    { label: "Turns", value: String(summary.turnsElapsed), color: "#a0aeb8" },
  ]

  const startX = 60
  const spacing = (CARD_W - 120) / (stats.length - 1)
  const statsY = 200

  for (let i = 0; i < stats.length; i++) {
    const s = stats[i]!
    const x = startX + i * spacing

    ctx.font = "bold 22px 'JetBrains Mono', monospace"
    ctx.fillStyle = s.color
    ctx.textAlign = "center"
    ctx.fillText(s.value, x, statsY)

    ctx.font = "10px 'Inter', sans-serif"
    ctx.fillStyle = "#a0aeb8"
    ctx.fillText(s.label, x, statsY + 18)
  }

  // Decorative line
  ctx.strokeStyle = "#2a3f52"
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(40, 260)
  ctx.lineTo(CARD_W - 40, 260)
  ctx.stroke()

  // Footer
  ctx.font = "11px 'Inter', sans-serif"
  ctx.fillStyle = "#4a5568"
  ctx.textAlign = "center"
  ctx.fillText("Generated from real Neural Memory brain data", CARD_W / 2, 286)

  ctx.font = "bold 12px 'Space Grotesk', sans-serif"
  ctx.fillStyle = "#2a3f52"
  ctx.fillText("neuralmemory.dev", CARD_W / 2, 310)

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error("Failed to generate card"))
    }, "image/png")
  })
}

/** Copy share card to clipboard */
export async function copyShareCard(state: DungeonState): Promise<void> {
  const blob = await generateShareCard(state)
  await navigator.clipboard.write([
    new ClipboardItem({ "image/png": blob }),
  ])
}

/** Download share card as PNG file */
export async function downloadShareCard(state: DungeonState): Promise<void> {
  const blob = await generateShareCard(state)
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `neurodungeon-run-${Date.now()}.png`
  a.click()
  URL.revokeObjectURL(url)
}
