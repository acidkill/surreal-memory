/* ------------------------------------------------------------------ */
/*  Neurodungeon — Main page composing all game components            */
/*  Phase 2: combat engine, items, world events, dialogue polish      */
/* ------------------------------------------------------------------ */

import { useCallback, useRef, useEffect, useState } from "react"
import { GamePhase, ItemType } from "./engine/types"
import type { GameAction, WorldEvent } from "./engine/types"
import { generateDungeon } from "./engine/dungeon-gen"
import { getActiveEnemy, defeatEnemy, currentFloor } from "./engine/game-loop"
import { resolveCombat, defeatScore } from "./engine/combat"
import type { CombatResult } from "./engine/combat"
import { useItem } from "./engine/items"
import { generateEvents, applyEvents, getFogModifier } from "./engine/events"
import { GameCanvas } from "./renderer/GameCanvas"
import { Minimap } from "./renderer/minimap"
import { Hud } from "./renderer/hud"
import type { EffectsState } from "./renderer/effects"
import {
  createEffectsState,
  emitDamageParticles,
  emitPickupParticles,
  emitDeathParticles,
  emitLevelUpParticles,
  createShake,
  createFlash,
} from "./renderer/effects"
import { StartScreen } from "./ui/StartScreen"
import { DialogueModal } from "./ui/DialogueModal"
import { CombatOverlay } from "./ui/CombatOverlay"
import { GameOverScreen } from "./ui/GameOverScreen"
import { QuizModal } from "./ui/QuizModal"
import { AchievementToast } from "./ui/AchievementToast"
import { FloorRatingOverlay } from "./ui/FloorRatingOverlay"
import { killStreakBonus, killStreakMessage } from "./engine/engagement"
import {
  checkAchievements,
  getUnlockedAchievements,
  saveAchievements,
} from "./engine/achievements"
import type { Achievement } from "./engine/achievements"
import { useGameStore } from "./hooks/useGameState"
import { useKeyboard } from "./hooks/useKeyboard"
import { useDungeonData } from "./hooks/useDungeonData"
import { useHealth } from "@/api/hooks/useDashboard"

export default function NeurodungeonPage() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 })
  const [combatResult, setCombatResult] = useState<CombatResult | null>(null)
  const [worldEvents, setWorldEvents] = useState<WorldEvent[]>([])
  const [effects, setEffects] = useState<EffectsState>(createEffectsState)
  const [quizNeuron, setQuizNeuron] = useState<{ content: string; type: string } | null>(null)
  const [unlockedAchievements, setUnlockedAchievements] = useState(() => getUnlockedAchievements())
  const [toastAchievement, setToastAchievement] = useState<Achievement | null>(null)
  const toastQueueRef = useRef<Achievement[]>([])

  const { inputs, isLoading, totalNeurons, totalFibers } = useDungeonData()
  const { data: healthData } = useHealth()
  const { dungeon, startGame, dispatch, resumeExploring, updateState, reset } =
    useGameStore()

  // Resize canvas to fill container
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const { width, height } = entry.contentRect
      setCanvasSize({ width: Math.floor(width), height: Math.floor(height) })
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  // Check achievements when state changes
  useEffect(() => {
    if (!dungeon) return
    const newlyUnlocked = checkAchievements(dungeon, unlockedAchievements)
    if (newlyUnlocked.length > 0) {
      const updated = new Set(unlockedAchievements)
      for (const a of newlyUnlocked) updated.add(a.id)
      setUnlockedAchievements(updated)
      saveAchievements(updated)
      toastQueueRef.current.push(...newlyUnlocked)
      if (!toastAchievement) {
        setToastAchievement(toastQueueRef.current.shift() ?? null)
      }
    }
  }, [dungeon, unlockedAchievements, toastAchievement])

  const dismissToast = useCallback(() => {
    const next = toastQueueRef.current.shift() ?? null
    setToastAchievement(next)
  }, [])

  // Generate world events from health data
  useEffect(() => {
    if (healthData) {
      setWorldEvents(generateEvents(healthData))
    }
  }, [healthData])

  // Start game handler
  const handleStart = useCallback(() => {
    if (inputs.length === 0) return
    const events = healthData ? generateEvents(healthData) : []
    const floors = inputs.map((input) => {
      const { floor } = generateDungeon(input)
      // Apply world events to each floor
      return events.length > 0 ? applyEvents(floor, events) : floor
    })
    startGame(floors)
    setCombatResult(null)
  }, [inputs, startGame, healthData])

  // Effects helper
  const addEffects = useCallback((partial: Partial<EffectsState>) => {
    setEffects((prev) => ({
      particles: [...prev.particles, ...(partial.particles ?? [])],
      shake: partial.shake ?? prev.shake,
      flash: partial.flash ?? prev.flash,
    }))
  }, [])

  // Combat handler
  const handleCombatAction = useCallback(
    (action: "attack" | "defend" | "flee") => {
      if (!dungeon) return
      const enemy = getActiveEnemy(dungeon)
      if (!enemy) return

      const result = resolveCombat(dungeon.player, enemy, { type: action })
      setCombatResult(result)

      // Consume shield if it absorbed a hit
      if (result.shieldAbsorbed) {
        updateState((s) => ({
          ...s,
          player: { ...s.player, shieldActive: false },
          turnLog: [
            ...s.turnLog,
            { turn: s.player.turnsElapsed, message: "Shield shattered!", type: "combat" as const },
          ],
        }))
        addEffects({ flash: createFlash("#e040fb", 0.2) })
      }

      // Visual effects for combat
      if (result.enemyDamage > 0) {
        addEffects({ particles: emitDamageParticles(enemy.position.x, enemy.position.y, 6, "#ff6b6b") })
      }
      if (result.playerDamage > 0) {
        addEffects({
          shake: createShake(result.isCritical ? 6 : 3, result.isCritical ? 10 : 6),
          flash: createFlash("#ff6b6b", result.isCritical ? 0.25 : 0.12),
        })
      }

      if (result.fled) {
        resumeExploring()
        setCombatResult(null)
        return
      }

      if (result.enemyDefeated) {
        addEffects({
          particles: emitDeathParticles(enemy.position.x, enemy.position.y),
          flash: createFlash("#00d084", 0.15),
        })
        const baseScore = defeatScore(enemy)
        updateState((s) => {
          const defeated = defeatEnemy(s, enemy.id)
          const newStreak = s.killStreak + 1
          const streakBonus = killStreakBonus(newStreak)
          const chainMultiplier = s.chain.multiplier
          const totalScore = Math.floor((baseScore + streakBonus) * chainMultiplier)
          const streakMsg = killStreakMessage(newStreak)
          const logs = [...defeated.turnLog]
          if (streakMsg) {
            logs.push({ turn: s.player.turnsElapsed, message: streakMsg, type: "combat" as const })
          }
          if (chainMultiplier > 1) {
            logs.push({ turn: s.player.turnsElapsed, message: `Chain x${s.chain.chainLength} bonus! Score x${chainMultiplier.toFixed(1)}`, type: "discovery" as const })
          }
          return {
            ...defeated,
            player: {
              ...defeated.player,
              stats: { ...defeated.player.stats, hp: result.playerHp },
              score: defeated.player.score + totalScore,
            },
            killStreak: newStreak,
            turnLog: logs,
          }
        })
        setTimeout(() => setCombatResult(null), 800)
        return
      }

      if (result.playerDied) {
        addEffects({
          shake: createShake(8, 15),
          flash: createFlash("#ff0000", 0.4),
        })
        updateState((s) => ({
          ...s,
          phase: GamePhase.GAME_OVER,
          player: { ...s.player, stats: { ...s.player.stats, hp: 0 } },
          turnLog: [
            ...s.turnLog,
            { turn: s.player.turnsElapsed, message: "You have been slain...", type: "combat" as const },
          ],
        }))
        return
      }

      // Update HP values after combat round
      updateState((s) => {
        const floor = currentFloor(s)
        const updatedRooms = floor.rooms.map((r) => ({
          ...r,
          entities: r.entities.map((e) =>
            e.id === enemy.id && e.stats
              ? { ...e, stats: { ...e.stats, hp: result.enemyHp } }
              : e,
          ),
        }))
        const updatedFloor = { ...floor, rooms: updatedRooms }
        const newFloors = [...s.floors]
        newFloors[s.currentFloorIndex] = updatedFloor

        return {
          ...s,
          floors: newFloors,
          player: {
            ...s.player,
            stats: { ...s.player.stats, hp: result.playerHp },
            turnsElapsed: s.player.turnsElapsed + 1,
          },
          killStreak: result.playerDamage > 0 ? 0 : s.killStreak,
          turnLog: [
            ...s.turnLog,
            { turn: s.player.turnsElapsed, message: result.message, type: "combat" as const },
          ],
        }
      })
    },
    [dungeon, updateState, resumeExploring, addEffects],
  )

  // Combat item usage
  const handleCombatItem = useCallback(
    (index: number) => {
      if (!dungeon) return
      const result = useItem(dungeon.player, index)
      if (!result) return

      // Map reveal during combat — reveal all rooms
      const item = dungeon.player.inventory[index]
      if (item?.type === ItemType.MAP_REVEAL) {
        updateState((s) => {
          const floor = currentFloor(s)
          const updatedRooms = floor.rooms.map((r) => ({ ...r, revealed: true }))
          const updatedFloor = { ...floor, rooms: updatedRooms }
          const newFloors = [...s.floors]
          newFloors[s.currentFloorIndex] = updatedFloor
          return {
            ...s,
            floors: newFloors,
            player: result.player,
            turnLog: [...s.turnLog, { turn: s.player.turnsElapsed, message: result.message, type: "item" as const }],
          }
        })
      } else {
        updateState((s) => ({
          ...s,
          player: result.player,
          turnLog: [...s.turnLog, { turn: s.player.turnsElapsed, message: result.message, type: "item" as const }],
        }))
      }
    },
    [dungeon, updateState],
  )

  // Action handler (from keyboard)
  const handleAction = useCallback(
    (action: GameAction) => {
      if (!dungeon) return

      // Combat actions -> combat engine
      if (dungeon.phase === GamePhase.COMBAT) {
        if (action.type === "attack") handleCombatAction("attack")
        else if (action.type === "defend") handleCombatAction("defend")
        else if (action.type === "flee") handleCombatAction("flee")
        return
      }

      // Item usage during exploration
      if (action.type === "use_item") {
        const result = useItem(dungeon.player, action.itemIndex)
        if (!result) return
        addEffects({ particles: emitPickupParticles(dungeon.player.position.x, dungeon.player.position.y) })

        const item = dungeon.player.inventory[action.itemIndex]
        if (item?.type === ItemType.MAP_REVEAL) {
          updateState((s) => {
            const floor = currentFloor(s)
            const updatedRooms = floor.rooms.map((r) => ({ ...r, revealed: true }))
            const updatedFloor = { ...floor, rooms: updatedRooms }
            const newFloors = [...s.floors]
            newFloors[s.currentFloorIndex] = updatedFloor
            return {
              ...s,
              floors: newFloors,
              player: result.player,
              turnLog: [...s.turnLog, { turn: s.player.turnsElapsed, message: result.message, type: "item" as const }],
            }
          })
        } else {
          updateState((s) => ({
            ...s,
            player: result.player,
            turnLog: [...s.turnLog, { turn: s.player.turnsElapsed, message: result.message, type: "item" as const }],
          }))
        }
        return
      }

      // After dispatching movement, check for quiz trigger
      dispatch(action)

      if (action.type === "move" && dungeon) {
        // 20% chance of quiz in puzzle rooms after move
        const floor = currentFloor(dungeon)
        const pos = dungeon.player.position
        const room = floor.rooms.find(
          (r) =>
            pos.x >= r.rect.x &&
            pos.x < r.rect.x + r.rect.w &&
            pos.y >= r.rect.y &&
            pos.y < r.rect.y + r.rect.h,
        )
        if (room?.type === "puzzle" && room.neuron && room.visited && Math.random() < 0.2) {
          setQuizNeuron({ content: room.neuron.content, type: room.neuron.type })
        }
      }
    },
    [dungeon, dispatch, updateState, handleCombatAction],
  )

  // Quiz answer handler
  const handleQuizAnswer = useCallback(
    (correct: boolean) => {
      setQuizNeuron(null)
      if (correct) {
        addEffects({ particles: emitLevelUpParticles(dungeon?.player.position.x ?? 0, dungeon?.player.position.y ?? 0) })
        updateState((s) => ({
          ...s,
          player: {
            ...s.player,
            score: s.player.score + 100,
            stats: { ...s.player.stats, hp: Math.min(s.player.stats.maxHp, s.player.stats.hp + 20) },
          },
          turnLog: [
            ...s.turnLog,
            { turn: s.player.turnsElapsed, message: "Memory recalled! +100 score, +20 HP", type: "item" as const },
          ],
        }))
      } else {
        addEffects({ shake: createShake(4, 8), flash: createFlash("#ff6b6b", 0.2) })
        updateState((s) => ({
          ...s,
          player: {
            ...s.player,
            stats: { ...s.player.stats, hp: Math.max(0, s.player.stats.hp - 15) },
          },
          turnLog: [
            ...s.turnLog,
            { turn: s.player.turnsElapsed, message: "Memory failed! Took 15 damage from ambush.", type: "combat" as const },
          ],
        }))
      }
    },
    [dungeon, updateState, addEffects],
  )

  // Effects tick callback
  const handleEffectsTick = useCallback((newEffects: EffectsState) => {
    setEffects(newEffects)
  }, [])

  // Keyboard input
  useKeyboard(
    handleAction,
    dungeon?.phase ?? GamePhase.START,
    dungeon !== null,
  )

  // Fog modifier from world events
  const fogMod = getFogModifier(worldEvents)

  // Not started — show start screen
  if (!dungeon) {
    return (
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          background: "#0c1419",
          borderRadius: 8,
          overflow: "hidden",
        }}
      >
        <StartScreen
          totalNeurons={totalNeurons}
          totalFibers={totalFibers}
          isLoading={isLoading}
          onStart={handleStart}
          worldEvents={worldEvents}
        />
      </div>
    )
  }

  const activeEnemy = dungeon.phase === GamePhase.COMBAT ? getActiveEnemy(dungeon) : null

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        background: "#0c1419",
        borderRadius: 8,
        overflow: "hidden",
      }}
      tabIndex={0}
    >
      {/* Game canvas */}
      <GameCanvas
        state={dungeon}
        width={canvasSize.width}
        height={canvasSize.height}
        fogModifier={fogMod}
        effects={effects}
        onEffectsTick={handleEffectsTick}
      />

      {/* Minimap */}
      <Minimap state={dungeon} />

      {/* HUD */}
      <Hud state={dungeon} worldEvents={worldEvents} />

      {/* Combat overlay */}
      {dungeon.phase === GamePhase.COMBAT && activeEnemy && (
        <CombatOverlay
          player={dungeon.player}
          enemy={activeEnemy}
          lastResult={combatResult}
          onAction={handleCombatAction}
          onUseItem={handleCombatItem}
        />
      )}

      {/* Quiz modal */}
      {quizNeuron && (
        <QuizModal
          neuronContent={quizNeuron.content}
          neuronType={quizNeuron.type}
          onAnswer={handleQuizAnswer}
        />
      )}

      {/* Dialogue modal */}
      <DialogueModal state={dungeon} onClose={resumeExploring} />

      {/* Floor rating overlay */}
      {dungeon.phase === GamePhase.FLOOR_COMPLETE && dungeon.floorResults.length > 0 && (
        <FloorRatingOverlay
          result={dungeon.floorResults[dungeon.floorResults.length - 1]!}
          onContinue={resumeExploring}
        />
      )}

      {/* Achievement toast */}
      <AchievementToast achievement={toastAchievement} onDismiss={dismissToast} />

      {/* Game over / Victory */}
      {(dungeon.phase === GamePhase.GAME_OVER ||
        dungeon.phase === GamePhase.VICTORY) && (
        <GameOverScreen
          state={dungeon}
          unlockedAchievements={unlockedAchievements}
          onRestart={() => {
            reset()
            handleStart()
          }}
        />
      )}
    </div>
  )
}
