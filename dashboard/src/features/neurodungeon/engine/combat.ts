/* ------------------------------------------------------------------ */
/*  Combat engine — type advantages, damage formulas, resolution      */
/*  Pure functions: no side effects, returns new state                 */
/* ------------------------------------------------------------------ */

import type { Entity, Player } from "./types"
import { EntityType } from "./types"

// --------------- Type advantage chart ---------------
// insight > error > pattern > decision > insight (rock-paper-scissors+)
// concept, entity, reference, etc. are neutral

const ADVANTAGE_MAP: Record<string, string[]> = {
  insight: ["error"],
  error: ["pattern"],
  pattern: ["decision"],
  decision: ["insight"],
  concept: [],
  entity: [],
  action: ["state"],
  state: [],
  reference: [],
  temporal: [],
  spatial: [],
}

const TYPE_BONUS = 1.5
const TYPE_PENALTY = 0.7

export interface CombatAction {
  type: "attack" | "defend" | "flee"
}

export interface CombatResult {
  playerDamage: number    // damage dealt TO player
  enemyDamage: number     // damage dealt TO enemy
  playerHp: number        // player HP after
  enemyHp: number         // enemy HP after
  fled: boolean
  enemyDefeated: boolean
  playerDied: boolean
  message: string
  isCritical: boolean
  shieldAbsorbed: boolean // shield blocked the hit
}

/** Resolve one round of combat */
export function resolveCombat(
  player: Player,
  enemy: Entity,
  action: CombatAction,
): CombatResult {
  if (!enemy.stats) {
    return {
      playerDamage: 0,
      enemyDamage: 0,
      playerHp: player.stats.hp,
      enemyHp: 0,
      fled: false,
      enemyDefeated: true,
      playerDied: false,
      message: "The shadow fades away...",
      isCritical: false,
      shieldAbsorbed: false,
    }
  }

  if (action.type === "flee") {
    // 70% chance to flee, 30% take hit while running
    const escaped = Math.random() < 0.7
    if (escaped) {
      return {
        playerDamage: 0,
        enemyDamage: 0,
        playerHp: player.stats.hp,
        enemyHp: enemy.stats.hp,
        fled: true,
        enemyDefeated: false,
        playerDied: false,
        message: "You escape into the shadows!",
        isCritical: false,
        shieldAbsorbed: false,
      }
    }
    // Failed flee — shield can absorb
    if (player.shieldActive) {
      return {
        playerDamage: 0,
        enemyDamage: 0,
        playerHp: player.stats.hp,
        enemyHp: enemy.stats.hp,
        fled: false,
        enemyDefeated: false,
        playerDied: false,
        message: "Failed to flee! Shield absorbed the hit!",
        isCritical: false,
        shieldAbsorbed: true,
      }
    }
    const fleeDmg = Math.max(1, Math.floor(enemy.stats.atk * 0.5))
    const newPlayerHp = player.stats.hp - fleeDmg
    return {
      playerDamage: fleeDmg,
      enemyDamage: 0,
      playerHp: Math.max(0, newPlayerHp),
      enemyHp: enemy.stats.hp,
      fled: false,
      enemyDefeated: false,
      playerDied: newPlayerHp <= 0,
      message: `Failed to flee! Took ${fleeDmg} damage while retreating.`,
      isCritical: false,
      shieldAbsorbed: false,
    }
  }

  if (action.type === "defend") {
    const blocked = Math.floor(player.stats.def * 1.5)
    const incomingDmg = Math.max(0, enemy.stats.atk - blocked)
    if (incomingDmg > 0 && player.shieldActive) {
      return {
        playerDamage: 0,
        enemyDamage: 0,
        playerHp: player.stats.hp,
        enemyHp: enemy.stats.hp,
        fled: false,
        enemyDefeated: false,
        playerDied: false,
        message: "Shield absorbed the remaining damage!",
        isCritical: false,
        shieldAbsorbed: true,
      }
    }
    const newPlayerHp = player.stats.hp - incomingDmg
    return {
      playerDamage: incomingDmg,
      enemyDamage: 0,
      playerHp: Math.max(0, newPlayerHp),
      enemyHp: enemy.stats.hp,
      fled: false,
      enemyDefeated: false,
      playerDied: newPlayerHp <= 0,
      message: incomingDmg === 0
        ? "You block all incoming damage!"
        : `You brace yourself. Took ${incomingDmg} reduced damage.`,
      isCritical: false,
      shieldAbsorbed: false,
    }
  }

  // Attack
  const typeMultiplier = getTypeMultiplier("player", enemy.stats.neuronType)
  const isCritical = Math.random() < 0.15 // 15% crit chance
  const critMultiplier = isCritical ? 1.8 : 1.0
  const rawDmg = player.stats.atk * typeMultiplier * critMultiplier
  const enemyDmg = Math.max(1, Math.floor(rawDmg - enemy.stats.def * 0.3))

  const enemyTypeMultiplier = getTypeMultiplier(enemy.stats.neuronType, "player")
  const enemyRawDmg = enemy.stats.atk * enemyTypeMultiplier
  const rawPlayerDmg = Math.max(1, Math.floor(enemyRawDmg - player.stats.def * 0.3))

  // Shield absorbs enemy counter-attack
  const shieldAbsorbed = rawPlayerDmg > 0 && player.shieldActive
  const playerDmg = shieldAbsorbed ? 0 : rawPlayerDmg

  const newEnemyHp = enemy.stats.hp - enemyDmg
  const newPlayerHp = player.stats.hp - playerDmg

  const parts: string[] = []
  if (isCritical) parts.push("CRITICAL HIT!")
  if (typeMultiplier > 1) parts.push("Super effective!")
  if (typeMultiplier < 1) parts.push("Not very effective...")
  parts.push(`You deal ${enemyDmg} dmg.`)
  if (shieldAbsorbed) {
    parts.push("Shield blocked the counter-attack!")
  } else {
    parts.push(`${enemy.name} deals ${playerDmg} dmg.`)
  }
  if (newEnemyHp <= 0) parts.push("Enemy defeated!")

  return {
    playerDamage: playerDmg,
    enemyDamage: enemyDmg,
    playerHp: Math.max(0, newPlayerHp),
    enemyHp: Math.max(0, newEnemyHp),
    fled: false,
    enemyDefeated: newEnemyHp <= 0,
    playerDied: newPlayerHp <= 0,
    message: parts.join(" "),
    isCritical,
    shieldAbsorbed,
  }
}

function getTypeMultiplier(attackerType: string, defenderType: string): number {
  const advantages = ADVANTAGE_MAP[attackerType]
  if (advantages?.includes(defenderType)) return TYPE_BONUS

  // Check if defender has advantage (penalty for attacker)
  const defAdvantages = ADVANTAGE_MAP[defenderType]
  if (defAdvantages?.includes(attackerType)) return TYPE_PENALTY

  return 1.0
}

/** Calculate score bonus for defeating an enemy */
export function defeatScore(enemy: Entity): number {
  const base = 50
  const bossBonus = enemy.type === EntityType.BOSS ? 200 : 0
  const hpBonus = Math.floor((enemy.stats?.maxHp ?? 0) / 5)
  return base + bossBonus + hpBonus
}
