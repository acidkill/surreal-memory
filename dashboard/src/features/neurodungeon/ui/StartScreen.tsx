/* ------------------------------------------------------------------ */
/*  Start screen — brain stats, high score, world events, controls    */
/* ------------------------------------------------------------------ */

import { useEffect, useState } from "react"
import type { WorldEvent } from "../engine/types"

const HIGH_SCORE_KEY = "neurodungeon_highscore"

interface StartScreenProps {
  totalNeurons: number
  totalFibers: number
  isLoading: boolean
  onStart: () => void
  worldEvents?: WorldEvent[]
}

export function StartScreen({
  totalNeurons,
  totalFibers,
  isLoading,
  onStart,
  worldEvents = [],
}: StartScreenProps) {
  const [highScore, setHighScore] = useState(0)

  useEffect(() => {
    try {
      setHighScore(Number(localStorage.getItem(HIGH_SCORE_KEY)) || 0)
    } catch {
      // ignore
    }
  }, [])

  // Enter key to start
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !isLoading && totalNeurons > 0) {
        onStart()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [isLoading, totalNeurons, onStart])

  const difficulty = worldEvents.length === 0 ? "Normal" : worldEvents.length <= 2 ? "Hard" : "Nightmare"
  const diffColor = worldEvents.length === 0 ? "#00d084" : worldEvents.length <= 2 ? "#ffa726" : "#ff6b6b"

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        gap: 20,
        color: "#f0f0f0",
        fontFamily: "Inter, sans-serif",
      }}
    >
      {/* Title */}
      <div style={{ textAlign: "center" }}>
        <h1
          style={{
            fontSize: 40,
            fontWeight: 700,
            fontFamily: "Space Grotesk, sans-serif",
            margin: 0,
            background: "linear-gradient(135deg, #00d084, #2196f3)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "0.08em",
          }}
        >
          NEURODUNGEON
        </h1>
        <p style={{ color: "#a0aeb8", fontSize: 13, marginTop: 6 }}>
          Explore the dungeon of your mind
        </p>
      </div>

      {/* Brain stats + difficulty */}
      <div
        style={{
          display: "flex",
          gap: 28,
          padding: "14px 28px",
          background: "rgba(26, 35, 50, 0.8)",
          borderRadius: 12,
          border: "1px solid #2a3f52",
        }}
      >
        <Stat label="Neurons" value={totalNeurons} color="#64b5f6" />
        <Stat label="Floors" value={totalFibers} color="#00d084" />
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: diffColor,
              fontFamily: "JetBrains Mono, monospace",
            }}
          >
            {difficulty}
          </div>
          <div style={{ fontSize: 11, color: "#a0aeb8", marginTop: 2 }}>
            Difficulty
          </div>
        </div>
        {highScore > 0 && (
          <div style={{ textAlign: "center" }}>
            <div
              style={{
                fontSize: 20,
                fontWeight: 700,
                color: "#ffd700",
                fontFamily: "JetBrains Mono, monospace",
              }}
            >
              {highScore}
            </div>
            <div style={{ fontSize: 11, color: "#a0aeb8", marginTop: 2 }}>
              Best
            </div>
          </div>
        )}
      </div>

      {/* World events (difficulty modifiers) */}
      {worldEvents.length > 0 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 4,
            padding: "10px 20px",
            background: "rgba(244, 67, 54, 0.08)",
            borderRadius: 8,
            border: "1px solid rgba(244, 67, 54, 0.2)",
            maxWidth: 400,
          }}
        >
          <span style={{ fontSize: 10, color: "#ff6b6b", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>
            Active World Events
          </span>
          {worldEvents.map((ev, i) => (
            <span key={i} style={{ fontSize: 12, color: "#a0aeb8" }}>
              {"!".repeat(ev.severity)} {ev.message.split("(")[0]?.trim()}
            </span>
          ))}
        </div>
      )}

      {/* Controls info */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto auto",
          gap: "3px 14px",
          fontSize: 11,
          color: "#a0aeb8",
          fontFamily: "JetBrains Mono, monospace",
        }}
      >
        <span style={{ color: "#64b5f6" }}>WASD/Arrows</span>
        <span>Move</span>
        <span style={{ color: "#64b5f6" }}>E / Enter</span>
        <span>Interact</span>
        <span style={{ color: "#64b5f6" }}>1-5</span>
        <span>Use item</span>
        <span style={{ color: "#64b5f6" }}>Shift+1-5</span>
        <span>Drop item</span>
        <span style={{ color: "#64b5f6" }}>A/D/F</span>
        <span>Attack/Defend/Flee</span>
        <span style={{ color: "#64b5f6" }}>&gt;</span>
        <span>Descend stairs</span>
      </div>

      {/* Start button */}
      <button
        onClick={onStart}
        disabled={isLoading || totalNeurons === 0}
        style={{
          background: isLoading
            ? "#334155"
            : "linear-gradient(135deg, #00d084, #2196f3)",
          color: "#fff",
          border: "none",
          borderRadius: 8,
          padding: "12px 40px",
          fontSize: 16,
          fontWeight: 700,
          cursor: isLoading ? "not-allowed" : "pointer",
          fontFamily: "Space Grotesk, sans-serif",
          letterSpacing: "0.05em",
          transition: "transform 150ms ease, box-shadow 150ms ease",
        }}
        onMouseEnter={(e) => {
          if (!isLoading) {
            e.currentTarget.style.transform = "scale(1.05)"
            e.currentTarget.style.boxShadow = "0 4px 20px rgba(0, 208, 132, 0.3)"
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = "scale(1)"
          e.currentTarget.style.boxShadow = "none"
        }}
      >
        {isLoading ? "Loading brain..." : totalNeurons === 0 ? "No memories yet" : "ENTER DUNGEON"}
      </button>

      {totalNeurons === 0 && (
        <p style={{ color: "#ff6b6b", fontSize: 13 }}>
          Use Neural Memory to create some memories first!
        </p>
      )}

      <p style={{ color: "#4a5568", fontSize: 11 }}>
        Press Enter to start
      </p>
    </div>
  )
}

function Stat({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) {
  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          fontSize: 24,
          fontWeight: 700,
          color,
          fontFamily: "JetBrains Mono, monospace",
        }}
      >
        {value}
      </div>
      <div style={{ fontSize: 11, color: "#a0aeb8", marginTop: 2 }}>
        {label}
      </div>
    </div>
  )
}
