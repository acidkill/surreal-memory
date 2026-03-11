/* ------------------------------------------------------------------ */
/*  HUD — Player stats overlay (HP, floor, score, turn log)          */
/* ------------------------------------------------------------------ */

import type { DungeonState, WorldEvent } from "../engine/types"
import { currentFloor } from "../engine/game-loop"
import { dangerColor, dangerLabel } from "../engine/engagement"

interface HudProps {
  state: DungeonState
  worldEvents?: WorldEvent[]
}

export function Hud({ state, worldEvents = [] }: HudProps) {
  const { player } = state
  const floor = currentFloor(state)
  const hpPct = Math.max(0, (player.stats.hp / player.stats.maxHp) * 100)
  const hpColor = hpPct > 60 ? "#00d084" : hpPct > 30 ? "#ffa726" : "#ff6b6b"

  // Last 4 log entries
  const recentLog = state.turnLog.slice(-4)

  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        padding: "8px 12px",
        background: "linear-gradient(transparent, rgba(12, 20, 25, 0.95))",
        color: "#a0aeb8",
        fontFamily: "JetBrains Mono, monospace",
        fontSize: "12px",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        pointerEvents: "none",
      }}
    >
      {/* Stats row */}
      <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
        {/* HP bar */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ color: hpColor, fontWeight: 700 }}>HP</span>
          <div
            style={{
              width: 80,
              height: 8,
              background: "rgba(255, 255, 255, 0.1)",
              borderRadius: 4,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${hpPct}%`,
                height: "100%",
                background: hpColor,
                borderRadius: 4,
                transition: "width 200ms ease",
              }}
            />
          </div>
          <span style={{ color: hpColor }}>
            {player.stats.hp}/{player.stats.maxHp}
          </span>
        </div>

        <span style={{ color: "#2196f3" }}>
          F{floor.depth + 1} {floor.name.length > 20 ? floor.name.slice(0, 20) + "\u2026" : floor.name}
        </span>

        <span>Rooms: {player.roomsExplored}</span>
        <span>Kills: {player.enemiesDefeated}</span>
        <span>Turn: {player.turnsElapsed}</span>
        <span style={{ color: "#ffd700", fontWeight: 700 }}>
          Score: {player.score}
        </span>

        {/* Active buffs */}
        {player.buffs.length > 0 && (
          <span style={{ color: "#ffa726" }}>
            {player.buffs.map((b) => `${b.type.toUpperCase()}+${b.value}(${b.turnsRemaining}t)`).join(" ")}
          </span>
        )}

        {/* Shield indicator */}
        {player.shieldActive && (
          <span style={{ color: "#e040fb", fontWeight: 700 }}>SHIELD</span>
        )}

        {/* Memory chain */}
        {state.chain.chainLength >= 2 && (
          <span style={{ color: "#ffa726", fontWeight: 700 }}>
            Chain x{state.chain.chainLength} ({state.chain.multiplier.toFixed(1)}x)
          </span>
        )}

        {/* Danger level */}
        <span style={{ color: dangerColor(state.danger.level), fontWeight: state.danger.level >= 4 ? 700 : 400 }}>
          {dangerLabel(state.danger.level)}
        </span>

        {/* Kill streak */}
        {state.killStreak >= 2 && (
          <span style={{ color: "#f44336", fontWeight: 700 }}>
            Streak x{state.killStreak}
          </span>
        )}

        {/* Inventory indicators */}
        {player.inventory.length > 0 && (
          <span style={{ color: "#ffd700" }}>
            Items: {player.inventory.map((it, i) => `[${i + 1}]${it.name.slice(0, 8)}`).join(" ")}
          </span>
        )}
      </div>

      {/* World events */}
      {worldEvents.length > 0 && (
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {worldEvents.map((ev, i) => (
            <span
              key={i}
              style={{
                fontSize: 10,
                padding: "1px 6px",
                borderRadius: 3,
                background: ev.severity >= 3 ? "rgba(244, 67, 54, 0.2)" : ev.severity >= 2 ? "rgba(255, 167, 38, 0.2)" : "rgba(100, 181, 246, 0.15)",
                color: ev.severity >= 3 ? "#ff6b6b" : ev.severity >= 2 ? "#ffa726" : "#64b5f6",
                border: `1px solid ${ev.severity >= 3 ? "#ff6b6b33" : ev.severity >= 2 ? "#ffa72633" : "#64b5f633"}`,
              }}
            >
              {ev.message.split(".")[0]}
            </span>
          ))}
        </div>
      )}

      {/* Turn log */}
      <div style={{ display: "flex", flexDirection: "column", gap: "1px" }}>
        {recentLog.map((entry, i) => (
          <span
            key={`${entry.turn}-${i}`}
            style={{
              color: logColor(entry.type),
              opacity: 0.5 + (i / recentLog.length) * 0.5,
              fontSize: "11px",
            }}
          >
            {entry.message}
          </span>
        ))}
      </div>
    </div>
  )
}

function logColor(type: string): string {
  switch (type) {
    case "combat":
      return "#ff6b6b"
    case "item":
      return "#ffd700"
    case "discovery":
      return "#64b5f6"
    case "event":
      return "#ffa726"
    default:
      return "#a0aeb8"
  }
}
