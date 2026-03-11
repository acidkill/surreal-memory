/* ------------------------------------------------------------------ */
/*  Combat overlay — turn-based combat UI with enemy info + actions    */
/* ------------------------------------------------------------------ */

import { useCallback } from "react"
import type { Entity, Player } from "../engine/types"
import { EntityType } from "../engine/types"
import type { CombatResult } from "../engine/combat"
import { itemColor } from "../engine/items"

interface CombatOverlayProps {
  player: Player
  enemy: Entity
  lastResult: CombatResult | null
  onAction: (action: "attack" | "defend" | "flee") => void
  onUseItem: (index: number) => void
}

export function CombatOverlay({
  player,
  enemy,
  lastResult,
  onAction,
  onUseItem,
}: CombatOverlayProps) {
  const stats = enemy.stats
  if (!stats) return null

  const enemyHpPct = (stats.hp / stats.maxHp) * 100
  const playerHpPct = (player.stats.hp / player.stats.maxHp) * 100
  const isBoss = enemy.type === EntityType.BOSS

  const handleKey = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "a":
        case "A":
        case "1":
          onAction("attack")
          break
        case "d":
        case "D":
        case "2":
          onAction("defend")
          break
        case "f":
        case "F":
        case "3":
          onAction("flee")
          break
        case "4":
        case "5":
        case "6":
        case "7":
        case "8": {
          const idx = parseInt(e.key) - 4
          if (idx < player.inventory.length) onUseItem(idx)
          break
        }
      }
    },
    [onAction, onUseItem, player.inventory.length],
  )

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.7)",
        zIndex: 20,
      }}
      onKeyDown={handleKey}
      tabIndex={0}
      // eslint-disable-next-line jsx-a11y/no-autofocus
      autoFocus
    >
      <div
        style={{
          background: "#121a20",
          border: `2px solid ${isBoss ? "#f44336" : "#ff6b6b"}`,
          borderRadius: 12,
          padding: "20px 28px",
          maxWidth: 500,
          width: "90%",
          color: "#f0f0f0",
          fontFamily: "JetBrains Mono, monospace",
        }}
      >
        {/* Enemy section */}
        <div style={{ textAlign: "center", marginBottom: 16 }}>
          <div
            style={{
              fontSize: isBoss ? 20 : 16,
              fontWeight: 700,
              color: isBoss ? "#f44336" : "#ff6b6b",
              marginBottom: 4,
            }}
          >
            {isBoss ? "BOSS " : ""}{enemy.name}
          </div>
          <div style={{ fontSize: 11, color: "#a0aeb8", marginBottom: 8 }}>
            Type: {stats.neuronType} | ATK: {stats.atk} | DEF: {stats.def}
          </div>
          <HpBar
            current={stats.hp}
            max={stats.maxHp}
            pct={enemyHpPct}
            color="#ff6b6b"
            label="Enemy"
          />
        </div>

        {/* VS divider */}
        <div
          style={{
            textAlign: "center",
            fontSize: 12,
            color: "#2a3f52",
            margin: "8px 0",
            letterSpacing: 4,
          }}
        >
          ---- VS ----
        </div>

        {/* Player section */}
        <div style={{ textAlign: "center", marginBottom: 12 }}>
          <HpBar
            current={player.stats.hp}
            max={player.stats.maxHp}
            pct={playerHpPct}
            color="#00d084"
            label="You"
          />
          <div style={{ fontSize: 11, color: "#a0aeb8", marginTop: 4 }}>
            ATK: {player.stats.atk} | DEF: {player.stats.def}
          </div>
        </div>

        {/* Combat log */}
        {lastResult && (
          <div
            style={{
              fontSize: 12,
              padding: "8px 12px",
              background: "rgba(255, 255, 255, 0.05)",
              borderRadius: 6,
              marginBottom: 12,
              color: lastResult.isCritical ? "#ffd700" : "#a0aeb8",
              textAlign: "center",
              lineHeight: 1.5,
            }}
          >
            {lastResult.message}
          </div>
        )}

        {/* Actions */}
        <div
          style={{
            display: "flex",
            gap: 8,
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <ActionButton
            label="Attack"
            hotkey="A"
            color="#ff6b6b"
            onClick={() => onAction("attack")}
          />
          <ActionButton
            label="Defend"
            hotkey="D"
            color="#64b5f6"
            onClick={() => onAction("defend")}
          />
          <ActionButton
            label="Flee"
            hotkey="F"
            color="#a0aeb8"
            onClick={() => onAction("flee")}
          />
        </div>

        {/* Inventory during combat */}
        {player.inventory.length > 0 && (
          <div style={{ marginTop: 10, textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "#2a3f52", marginBottom: 4 }}>
              ITEMS (press 4-8)
            </div>
            <div style={{ display: "flex", gap: 6, justifyContent: "center" }}>
              {player.inventory.map((item, i) => (
                <button
                  key={item.id}
                  onClick={() => onUseItem(i)}
                  style={{
                    background: "rgba(255, 255, 255, 0.05)",
                    border: `1px solid ${itemColor(item.type)}`,
                    borderRadius: 4,
                    padding: "2px 8px",
                    color: itemColor(item.type),
                    fontSize: 10,
                    cursor: "pointer",
                    fontFamily: "JetBrains Mono, monospace",
                  }}
                  title={item.description}
                >
                  [{i + 4}] {item.name.slice(0, 10)}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function HpBar({
  current,
  max,
  pct,
  color,
  label,
}: {
  current: number
  max: number
  pct: number
  color: string
  label: string
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "center" }}>
      <span style={{ fontSize: 11, color, fontWeight: 700, width: 40, textAlign: "right" }}>
        {label}
      </span>
      <div
        style={{
          width: 140,
          height: 10,
          background: "rgba(255, 255, 255, 0.08)",
          borderRadius: 5,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: 5,
            transition: "width 300ms ease",
          }}
        />
      </div>
      <span style={{ fontSize: 11, color, width: 60 }}>
        {current}/{max}
      </span>
    </div>
  )
}

function ActionButton({
  label,
  hotkey,
  color,
  onClick,
}: {
  label: string
  hotkey: string
  color: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "rgba(255, 255, 255, 0.05)",
        border: `1px solid ${color}`,
        borderRadius: 6,
        padding: "6px 16px",
        color,
        fontSize: 13,
        fontWeight: 600,
        cursor: "pointer",
        fontFamily: "JetBrains Mono, monospace",
        transition: "background 150ms",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = `${color}22`)}
      onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.05)")}
    >
      [{hotkey}] {label}
    </button>
  )
}
