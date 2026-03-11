/* ------------------------------------------------------------------ */
/*  Keyboard input hook for turn-based movement                       */
/* ------------------------------------------------------------------ */

import { useEffect, useCallback, useRef } from "react"
import { Direction, type GameAction, GamePhase } from "../engine/types"

type ActionHandler = (action: GameAction) => void

/**
 * Captures keyboard input and dispatches game actions.
 * Only fires in EXPLORING or COMBAT phase.
 * Debounced to prevent held-key spam in turn-based game.
 */
export function useKeyboard(
  onAction: ActionHandler,
  phase: GamePhase,
  enabled: boolean = true,
) {
  const lastKeyTime = useRef(0)
  const DEBOUNCE_MS = 120

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled) return

      const now = Date.now()
      if (now - lastKeyTime.current < DEBOUNCE_MS) return
      lastKeyTime.current = now

      // Exploring phase — movement + interaction
      if (phase === GamePhase.EXPLORING) {
        switch (e.key) {
          case "ArrowUp":
          case "w":
          case "W":
            e.preventDefault()
            onAction({ type: "move", direction: Direction.UP })
            return
          case "ArrowDown":
          case "s":
          case "S":
            e.preventDefault()
            onAction({ type: "move", direction: Direction.DOWN })
            return
          case "ArrowLeft":
          case "a":
          case "A":
            e.preventDefault()
            onAction({ type: "move", direction: Direction.LEFT })
            return
          case "ArrowRight":
          case "d":
          case "D":
            e.preventDefault()
            onAction({ type: "move", direction: Direction.RIGHT })
            return
          case "e":
          case "E":
          case "Enter":
            e.preventDefault()
            onAction({ type: "interact" })
            return
          case ".":
          case " ":
            e.preventDefault()
            onAction({ type: "wait" })
            return
          case ">":
            e.preventDefault()
            onAction({ type: "next_floor" })
            return
          case "1":
          case "2":
          case "3":
          case "4":
          case "5": {
            e.preventDefault()
            const idx = parseInt(e.key) - 1
            if (e.shiftKey) {
              onAction({ type: "drop_item", itemIndex: idx })
            } else {
              onAction({ type: "use_item", itemIndex: idx })
            }
            return
          }
        }
      }

      // Combat phase — attack/defend/flee
      if (phase === GamePhase.COMBAT) {
        switch (e.key) {
          case "a":
          case "A":
          case "1":
            e.preventDefault()
            onAction({ type: "attack" })
            return
          case "d":
          case "D":
          case "2":
            e.preventDefault()
            onAction({ type: "defend" })
            return
          case "f":
          case "F":
          case "3":
          case "Escape":
            e.preventDefault()
            onAction({ type: "flee" })
            return
        }
      }
    },
    [onAction, phase, enabled],
  )

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [handleKeyDown])
}
