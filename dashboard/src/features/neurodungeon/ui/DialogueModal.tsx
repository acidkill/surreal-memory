/* ------------------------------------------------------------------ */
/*  Dialogue modal — different styles per room type                   */
/*  Library=scroll, NPC=speech, Trap=warning, Puzzle=riddle           */
/* ------------------------------------------------------------------ */

import type { DungeonState, Room } from "../engine/types"
import { GamePhase, EntityType, RoomType } from "../engine/types"
import { currentFloor } from "../engine/game-loop"

interface DialogueModalProps {
  state: DungeonState
  onClose: () => void
}

const ROOM_STYLE: Record<string, { icon: string; title: string; borderColor: string; iconColor: string }> = {
  [RoomType.LIBRARY]: { icon: "\u{1F4DC}", title: "Ancient Knowledge", borderColor: "#64b5f6", iconColor: "#64b5f6" },
  [RoomType.NPC_ROOM]: { icon: "\u{1F5E3}", title: "Memory Speaks", borderColor: "#00d084", iconColor: "#00d084" },
  [RoomType.TRAP_ROOM]: { icon: "\u26A0", title: "WARNING", borderColor: "#ff6b6b", iconColor: "#ff6b6b" },
  [RoomType.TREASURE]: { icon: "\u2728", title: "Discovery", borderColor: "#ffd700", iconColor: "#ffd700" },
  [RoomType.FORK]: { icon: "\u{1F500}", title: "A Decision Was Made", borderColor: "#ab47bc", iconColor: "#ab47bc" },
  [RoomType.PUZZLE]: { icon: "\u{1F9E9}", title: "Pattern Detected", borderColor: "#ffa726", iconColor: "#ffa726" },
  [RoomType.SECRET]: { icon: "\u{1F441}", title: "Hidden Memory", borderColor: "#00e676", iconColor: "#00e676" },
  [RoomType.EMPTY]: { icon: "\u{1F4AD}", title: "A Faint Echo", borderColor: "#546e7a", iconColor: "#546e7a" },
  [RoomType.BOSS_ROOM]: { icon: "\u{1F480}", title: "Guardian's Domain", borderColor: "#f44336", iconColor: "#f44336" },
  [RoomType.ENEMY_ROOM]: { icon: "\u2694", title: "Hostile Territory", borderColor: "#ff6b6b", iconColor: "#ff6b6b" },
  [RoomType.STAIRS]: { icon: "\u{1F6AA}", title: "Passage", borderColor: "#2196f3", iconColor: "#2196f3" },
}

const DEFAULT_STYLE = { icon: "\u{1F4AD}", title: "Memory", borderColor: "#546e7a", iconColor: "#546e7a" }

export function DialogueModal({ state, onClose }: DialogueModalProps) {
  if (state.phase !== GamePhase.DIALOGUE) return null

  const floor = currentFloor(state)
  const pos = state.player.position

  const room = floor.rooms.find(
    (r) =>
      pos.x >= r.rect.x &&
      pos.x < r.rect.x + r.rect.w &&
      pos.y >= r.rect.y &&
      pos.y < r.rect.y + r.rect.h,
  )

  if (!room) {
    onClose()
    return null
  }

  const npc = room.entities.find((e) => e.type === EntityType.NPC)
  const content = npc?.content ?? room.neuron?.content ?? "..."
  const style = ROOM_STYLE[room.type] ?? DEFAULT_STYLE
  const neuronType = room.neuron?.type ?? "unknown"
  const tags = (room.neuron?.metadata?.tags as string[]) ?? []

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.6)",
        zIndex: 20,
      }}
      onClick={onClose}
      onKeyDown={(e) => {
        if (e.key === "e" || e.key === "E" || e.key === "Escape" || e.key === "Enter") {
          onClose()
        }
      }}
      tabIndex={0}
      // eslint-disable-next-line jsx-a11y/no-autofocus
      autoFocus
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#121a20",
          border: `2px solid ${style.borderColor}`,
          borderRadius: 12,
          padding: "20px 24px",
          maxWidth: 520,
          width: "90%",
          color: "#f0f0f0",
          fontFamily: "Inter, sans-serif",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          <span style={{ fontSize: 24 }}>{style.icon}</span>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: style.iconColor }}>
              {style.title}
            </div>
            <div style={{ fontSize: 11, color: "#546e7a" }}>
              {neuronType}
              {tags.length > 0 && ` · ${tags.slice(0, 3).join(", ")}`}
            </div>
          </div>
        </div>

        {/* Content */}
        <div
          style={{
            fontSize: 14,
            lineHeight: 1.7,
            color: "#c8d6e0",
            maxHeight: 220,
            overflowY: "auto",
            marginBottom: 16,
            padding: "12px 16px",
            background: "rgba(255, 255, 255, 0.03)",
            borderRadius: 8,
            borderLeft: `3px solid ${style.borderColor}`,
            fontFamily: room.type === RoomType.LIBRARY
              ? "Georgia, serif"
              : "Inter, sans-serif",
            fontStyle: room.type === RoomType.LIBRARY ? "italic" : "normal",
          }}
        >
          {content}
        </div>

        {/* Room hint */}
        <RoomHint room={room} />

        {/* Close */}
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={onClose}
            style={{
              background: `${style.borderColor}22`,
              color: style.borderColor,
              border: `1px solid ${style.borderColor}`,
              borderRadius: 6,
              padding: "6px 16px",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "JetBrains Mono, monospace",
            }}
          >
            Continue [E]
          </button>
        </div>
      </div>
    </div>
  )
}

function RoomHint({ room }: { room: Room }) {
  let hint = ""
  switch (room.type) {
    case RoomType.LIBRARY:
      hint = "This knowledge may help you navigate the depths."
      break
    case RoomType.TREASURE:
      hint = "Press E to collect the memory fragment."
      break
    case RoomType.TRAP_ROOM:
      hint = "Tread carefully. This memory holds pain."
      break
    case RoomType.FORK:
      hint = "A crossroads of thought. The choice echoes still."
      break
    case RoomType.PUZZLE:
      hint = "A recurring pattern. Understanding it grants power."
      break
    case RoomType.SECRET:
      hint = "An orphan memory, disconnected from the network."
      break
    default:
      return null
  }

  return (
    <div
      style={{
        fontSize: 11,
        color: "#546e7a",
        fontStyle: "italic",
        marginBottom: 12,
        textAlign: "center",
      }}
    >
      {hint}
    </div>
  )
}
