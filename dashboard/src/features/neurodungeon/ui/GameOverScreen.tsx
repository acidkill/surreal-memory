/* ------------------------------------------------------------------ */
/*  Game Over / Victory screen with score summary + high score         */
/* ------------------------------------------------------------------ */

import { useEffect, useState } from "react"
import { GamePhase } from "../engine/types"
import type { DungeonState } from "../engine/types"
import { getAchievementList, ACHIEVEMENTS } from "../engine/achievements"
import { copyShareCard, downloadShareCard } from "../utils/share-card"

const HIGH_SCORE_KEY = "neurodungeon_highscore"

function getHighScore(): number {
  try {
    return Number(localStorage.getItem(HIGH_SCORE_KEY)) || 0
  } catch {
    return 0
  }
}

function saveHighScore(score: number): void {
  try {
    localStorage.setItem(HIGH_SCORE_KEY, String(score))
  } catch {
    // localStorage unavailable
  }
}

interface GameOverScreenProps {
  state: DungeonState
  unlockedAchievements?: Set<string>
  onRestart: () => void
}

export function GameOverScreen({ state, unlockedAchievements, onRestart }: GameOverScreenProps) {
  const isVictory = state.phase === GamePhase.VICTORY
  const { player } = state
  const [isNewHighScore, setIsNewHighScore] = useState(false)
  const [prevHighScore, setPrevHighScore] = useState(0)
  const [shareStatus, setShareStatus] = useState<"idle" | "copied" | "error">("idle")

  useEffect(() => {
    const prev = getHighScore()
    setPrevHighScore(prev)
    if (player.score > prev) {
      setIsNewHighScore(true)
      saveHighScore(player.score)
    }
  }, [player.score])

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.85)",
        zIndex: 30,
      }}
    >
      <div
        style={{
          textAlign: "center",
          color: "#f0f0f0",
          fontFamily: "Inter, sans-serif",
          maxWidth: 420,
        }}
      >
        <h2
          style={{
            fontSize: 36,
            fontWeight: 700,
            fontFamily: "Space Grotesk, sans-serif",
            color: isVictory ? "#00d084" : "#ff6b6b",
            margin: "0 0 4px 0",
            letterSpacing: "0.05em",
          }}
        >
          {isVictory ? "DUNGEON CONQUERED" : "YOU DIED"}
        </h2>

        <p style={{ color: "#a0aeb8", fontSize: 14, marginBottom: 8 }}>
          {isVictory
            ? "Your memories guided you through the depths."
            : "The dungeon claims another explorer..."}
        </p>

        {isNewHighScore && (
          <p
            style={{
              color: "#ffd700",
              fontSize: 14,
              fontWeight: 700,
              fontFamily: "Space Grotesk, sans-serif",
              margin: "0 0 16px 0",
              animation: "pulse 1s ease infinite",
            }}
          >
            NEW HIGH SCORE!
          </p>
        )}

        {/* Score highlight */}
        <div
          style={{
            fontSize: 48,
            fontWeight: 700,
            fontFamily: "JetBrains Mono, monospace",
            color: "#ffd700",
            marginBottom: 4,
          }}
        >
          {player.score}
        </div>
        <div style={{ fontSize: 11, color: "#a0aeb8", marginBottom: 16 }}>
          {!isNewHighScore && prevHighScore > 0 && `Best: ${prevHighScore}`}
        </div>

        {/* Stats grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "8px 24px",
            padding: "14px 28px",
            background: "rgba(26, 35, 50, 0.8)",
            borderRadius: 12,
            border: "1px solid #2a3f52",
            marginBottom: 24,
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 13,
          }}
        >
          <StatRow label="Rooms" value={String(player.roomsExplored)} color="#64b5f6" />
          <StatRow label="Enemies" value={String(player.enemiesDefeated)} color="#ff6b6b" />
          <StatRow label="Turns" value={String(player.turnsElapsed)} color="#a0aeb8" />
          <StatRow label="Memories" value={String(state.visitedNeuronIds.size)} color="#00d084" />
          <StatRow
            label="Floor"
            value={`${state.currentFloorIndex + 1}/${state.floors.length}`}
            color="#2196f3"
          />
          <StatRow label="Items" value={String(player.inventory.length)} color="#ffd700" />
        </div>

        {/* Achievements unlocked this run */}
        {unlockedAchievements && unlockedAchievements.size > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 10, color: "#ffd700", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
              Achievements ({unlockedAchievements.size}/{ACHIEVEMENTS.length})
            </div>
            <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap" }}>
              {getAchievementList(unlockedAchievements).map((a) => (
                <span
                  key={a.id}
                  title={`${a.name}: ${a.description}`}
                  style={{
                    fontSize: 16,
                    width: 28,
                    height: 28,
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    borderRadius: 4,
                    background: a.isUnlocked ? "rgba(255, 215, 0, 0.15)" : "rgba(255, 255, 255, 0.03)",
                    border: `1px solid ${a.isUnlocked ? "#ffd700" : "#2a3f52"}`,
                    color: a.isUnlocked ? "#ffd700" : "#2a3f52",
                    fontFamily: "JetBrains Mono, monospace",
                    fontWeight: 700,
                    cursor: "default",
                  }}
                >
                  {a.icon}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Share buttons */}
        <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16 }}>
          <ShareButton
            label={shareStatus === "copied" ? "Copied!" : "Copy Card"}
            onClick={async () => {
              try {
                await copyShareCard(state)
                setShareStatus("copied")
                setTimeout(() => setShareStatus("idle"), 2000)
              } catch {
                setShareStatus("error")
              }
            }}
          />
          <ShareButton
            label="Download"
            onClick={() => downloadShareCard(state)}
          />
        </div>

        <button
          onClick={onRestart}
          style={{
            background: "linear-gradient(135deg, #00d084, #2196f3)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "12px 40px",
            fontSize: 16,
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: "Space Grotesk, sans-serif",
            letterSpacing: "0.05em",
            transition: "transform 150ms ease, box-shadow 150ms ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "scale(1.05)"
            e.currentTarget.style.boxShadow = "0 4px 20px rgba(0, 208, 132, 0.3)"
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "scale(1)"
            e.currentTarget.style.boxShadow = "none"
          }}
        >
          PLAY AGAIN
        </button>

        <p style={{ color: "#4a5568", fontSize: 11, marginTop: 12 }}>
          Press Enter to restart
        </p>
      </div>
    </div>
  )
}

function StatRow({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color: string
}) {
  return (
    <>
      <span style={{ color: "#a0aeb8", textAlign: "right" }}>{label}</span>
      <span style={{ color, textAlign: "left", fontWeight: 700 }}>{value}</span>
    </>
  )
}

function ShareButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "rgba(255, 255, 255, 0.05)",
        border: "1px solid #2a3f52",
        borderRadius: 6,
        padding: "6px 16px",
        color: "#a0aeb8",
        fontSize: 12,
        cursor: "pointer",
        fontFamily: "JetBrains Mono, monospace",
        transition: "background 150ms",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.1)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.05)")}
    >
      {label}
    </button>
  )
}
