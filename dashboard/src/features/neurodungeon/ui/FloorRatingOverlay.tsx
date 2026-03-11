/* ------------------------------------------------------------------ */
/*  Floor Rating overlay — shown when completing a floor               */
/* ------------------------------------------------------------------ */

import type { FloorResult } from "../engine/types"
import { ratingColor } from "../engine/engagement"

interface FloorRatingOverlayProps {
  result: FloorResult
  onContinue: () => void
}

export function FloorRatingOverlay({ result, onContinue }: FloorRatingOverlayProps) {
  const color = ratingColor(result.rating)

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.85)",
        zIndex: 25,
      }}
    >
      <div
        style={{
          textAlign: "center",
          color: "#f0f0f0",
          fontFamily: "Inter, sans-serif",
          maxWidth: 380,
        }}
      >
        <div style={{ fontSize: 12, color: "#a0aeb8", textTransform: "uppercase", letterSpacing: 2, marginBottom: 4 }}>
          Floor Complete
        </div>
        <div style={{ fontSize: 14, color: "#64b5f6", marginBottom: 16 }}>
          {result.floorName.length > 40 ? result.floorName.slice(0, 40) + "\u2026" : result.floorName}
        </div>

        {/* Big rating letter */}
        <div
          style={{
            fontSize: 72,
            fontWeight: 700,
            fontFamily: "Space Grotesk, sans-serif",
            color,
            lineHeight: 1,
            marginBottom: 8,
            textShadow: `0 0 30px ${color}44`,
          }}
        >
          {result.rating}
        </div>

        <div
          style={{
            fontSize: 24,
            fontWeight: 700,
            fontFamily: "JetBrains Mono, monospace",
            color: "#ffd700",
            marginBottom: 20,
          }}
        >
          +{result.scoreEarned} score
        </div>

        {/* Breakdown */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr auto",
            gap: "6px 24px",
            padding: "12px 24px",
            background: "rgba(26, 35, 50, 0.6)",
            borderRadius: 8,
            fontSize: 13,
            fontFamily: "JetBrains Mono, monospace",
            marginBottom: 20,
          }}
        >
          <BarStat label="Explored" pct={result.explorationPct} color="#64b5f6" />
          <BarStat label="Kills" pct={result.killPct} color="#ff6b6b" />
          <span style={{ color: "#a0aeb8", textAlign: "left" }}>Best Chain</span>
          <span style={{ color: "#ffa726", textAlign: "right", fontWeight: 700 }}>x{result.chainBonus}</span>
          <span style={{ color: "#a0aeb8", textAlign: "left" }}>Turns</span>
          <span style={{ color: "#a0aeb8", textAlign: "right" }}>{result.turnsUsed}</span>
        </div>

        <button
          onClick={onContinue}
          style={{
            background: "linear-gradient(135deg, #00d084, #2196f3)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "10px 32px",
            fontSize: 14,
            fontWeight: 700,
            cursor: "pointer",
            fontFamily: "Space Grotesk, sans-serif",
          }}
        >
          DESCEND DEEPER
        </button>

        <p style={{ color: "#4a5568", fontSize: 11, marginTop: 8 }}>
          Press Enter to continue
        </p>
      </div>
    </div>
  )
}

function BarStat({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <>
      <span style={{ color: "#a0aeb8", textAlign: "left" }}>{label}</span>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div
          style={{
            width: 60,
            height: 6,
            background: "rgba(255,255,255,0.1)",
            borderRadius: 3,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${Math.round(pct * 100)}%`,
              height: "100%",
              background: color,
              borderRadius: 3,
            }}
          />
        </div>
        <span style={{ color, fontWeight: 700, fontSize: 12, minWidth: 36, textAlign: "right" }}>
          {Math.round(pct * 100)}%
        </span>
      </div>
    </>
  )
}
