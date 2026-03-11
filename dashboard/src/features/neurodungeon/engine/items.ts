/* ------------------------------------------------------------------ */
/*  Item system — generation, usage, drop mechanics                    */
/*                                                                      */
/*  Design intent:                                                      */
/*  - Heal potions: save for critical moments (HP management)          */
/*  - ATK/DEF scrolls: temporary buffs (10 turns), timing matters      */
/*  - Map reveal: strategic — use before exploring or save for later   */
/*  - Shield: block one hit completely, clutch save                    */
/*  - Drop: recycle for +15 score, forces inventory prioritization     */
/* ------------------------------------------------------------------ */

import type { Item, Player, Room } from "./types"
import { ItemType, RoomType } from "./types"
import type { GraphNeuron } from "@/api/types"

const BUFF_DURATION = 10 // turns

/** Generate an item appropriate for the room/neuron type */
export function generateItem(room: Room, neuron: GraphNeuron | null): Item {
  const content = neuron?.content ?? "A mysterious artifact"
  const activation = (neuron?.metadata?.activation_level as number) ?? 0.5
  const priority = (neuron?.metadata?.priority as number) ?? 5

  switch (room.type) {
    case RoomType.TREASURE:
      // High-activation treasures = shield, otherwise heal
      return activation > 0.7
        ? {
            id: `item-${room.id}`,
            type: ItemType.SHIELD,
            name: "Memory Barrier",
            description: `A vivid memory forms a protective shell. Blocks one hit.`,
            value: 1,
            neuronId: neuron?.id ?? null,
          }
        : {
            id: `item-${room.id}`,
            type: ItemType.HEAL_POTION,
            name: "Memory Essence",
            description: truncate(content, 60),
            value: Math.max(15, Math.floor(activation * 50)),
            neuronId: neuron?.id ?? null,
          }

    case RoomType.FORK:
      // Decision neurons -> offensive or defensive buff (temporary!)
      return priority >= 6
        ? {
            id: `item-${room.id}`,
            type: ItemType.ATK_SCROLL,
            name: "Scroll of Resolve",
            description: `ATK +${Math.max(3, Math.floor(activation * 8))} for ${BUFF_DURATION} turns.`,
            value: Math.max(3, Math.floor(activation * 8)),
            neuronId: neuron?.id ?? null,
          }
        : {
            id: `item-${room.id}`,
            type: ItemType.DEF_SCROLL,
            name: "Scroll of Caution",
            description: `DEF +${Math.max(2, Math.floor(activation * 6))} for ${BUFF_DURATION} turns.`,
            value: Math.max(2, Math.floor(activation * 6)),
            neuronId: neuron?.id ?? null,
          }

    case RoomType.LIBRARY:
      // Varied loot from libraries — not always map reveal
      return pickLibraryItem(room, neuron, content, activation)

    case RoomType.SECRET:
      // Secret rooms = rare shield
      return {
        id: `item-${room.id}`,
        type: ItemType.SHIELD,
        name: "Forgotten Ward",
        description: "An ancient protection, found in a hidden place. Blocks one hit.",
        value: 1,
        neuronId: neuron?.id ?? null,
      }

    case RoomType.EMPTY:
      return {
        id: `item-${room.id}`,
        type: ItemType.HEAL_POTION,
        name: "Forgotten Fragment",
        description: `A faint memory, barely holding together. ${truncate(content, 30)}`,
        value: 10,
        neuronId: neuron?.id ?? null,
      }

    default:
      return {
        id: `item-${room.id}`,
        type: ItemType.HEAL_POTION,
        name: "Memory Shard",
        description: truncate(content, 50),
        value: 15,
        neuronId: neuron?.id ?? null,
      }
  }
}

/** Libraries have a varied loot table based on neuron properties */
function pickLibraryItem(
  room: Room,
  neuron: GraphNeuron | null,
  content: string,
  activation: number,
): Item {
  // Use neuron id hash to deterministically pick item type
  const hash = simpleHash(room.id)
  const roll = hash % 4

  switch (roll) {
    case 0:
      return {
        id: `item-${room.id}`,
        type: ItemType.MAP_REVEAL,
        name: "Scholar's Map",
        description: `Ancient knowledge reveals hidden paths. ${truncate(content, 30)}`,
        value: 1,
        neuronId: neuron?.id ?? null,
      }
    case 1:
      return {
        id: `item-${room.id}`,
        type: ItemType.HEAL_POTION,
        name: "Tome of Restoration",
        description: `Reading this brings clarity and peace. +${Math.max(20, Math.floor(activation * 40))} HP`,
        value: Math.max(20, Math.floor(activation * 40)),
        neuronId: neuron?.id ?? null,
      }
    case 2:
      return {
        id: `item-${room.id}`,
        type: ItemType.ATK_SCROLL,
        name: "Treatise on War",
        description: `Strategic insights sharpen your attacks. ATK +${Math.max(2, Math.floor(activation * 5))} for ${BUFF_DURATION} turns.`,
        value: Math.max(2, Math.floor(activation * 5)),
        neuronId: neuron?.id ?? null,
      }
    default:
      return {
        id: `item-${room.id}`,
        type: ItemType.DEF_SCROLL,
        name: "Manual of Defense",
        description: `Defensive techniques from ancient scholars. DEF +${Math.max(2, Math.floor(activation * 5))} for ${BUFF_DURATION} turns.`,
        value: Math.max(2, Math.floor(activation * 5)),
        neuronId: neuron?.id ?? null,
      }
  }
}

/** Apply item effect to player, return new player (immutable) */
export function useItem(player: Player, itemIndex: number): { player: Player; message: string } | null {
  const item = player.inventory[itemIndex]
  if (!item) return null

  const newInventory = player.inventory.filter((_, i) => i !== itemIndex)

  switch (item.type) {
    case ItemType.HEAL_POTION: {
      const healed = Math.min(item.value, player.stats.maxHp - player.stats.hp)
      return {
        player: {
          ...player,
          stats: { ...player.stats, hp: player.stats.hp + healed },
          inventory: newInventory,
        },
        message: healed > 0
          ? `Used ${item.name}. Restored ${healed} HP.`
          : `Used ${item.name}. Already at full HP!`,
      }
    }

    case ItemType.ATK_SCROLL:
      return {
        player: {
          ...player,
          stats: { ...player.stats, atk: player.stats.atk + item.value },
          buffs: [...player.buffs, { type: "atk", value: item.value, turnsRemaining: BUFF_DURATION }],
          inventory: newInventory,
        },
        message: `Used ${item.name}. ATK +${item.value} for ${BUFF_DURATION} turns!`,
      }

    case ItemType.DEF_SCROLL:
      return {
        player: {
          ...player,
          stats: { ...player.stats, def: player.stats.def + item.value },
          buffs: [...player.buffs, { type: "def", value: item.value, turnsRemaining: BUFF_DURATION }],
          inventory: newInventory,
        },
        message: `Used ${item.name}. DEF +${item.value} for ${BUFF_DURATION} turns!`,
      }

    case ItemType.MAP_REVEAL:
      // Map reveal handled at floor level — caller must reveal rooms
      return {
        player: { ...player, inventory: newInventory },
        message: `Used ${item.name}. The floor layout is revealed!`,
      }

    case ItemType.SHIELD:
      return {
        player: { ...player, shieldActive: true, inventory: newInventory },
        message: `Used ${item.name}. Next hit will be completely blocked!`,
      }

    case ItemType.KEY:
      return {
        player: { ...player, inventory: newInventory },
        message: `Used ${item.name}. A sealed passage opens...`,
      }

    default:
      return null
  }
}

/** Tick down buff durations. Call once per turn. Returns new player with expired buffs removed. */
export function tickBuffs(player: Player): Player {
  const updatedBuffs = player.buffs
    .map((b) => ({ ...b, turnsRemaining: b.turnsRemaining - 1 }))
    .filter((b) => b.turnsRemaining > 0)

  // Remove expired buff values from stats
  const expiredBuffs = player.buffs.filter(
    (b) => b.turnsRemaining <= 1, // will be 0 after tick
  )

  let { atk, def } = player.stats
  for (const b of expiredBuffs) {
    if (b.type === "atk") atk -= b.value
    if (b.type === "def") def -= b.value
  }

  if (expiredBuffs.length === 0) return { ...player, buffs: updatedBuffs }

  return {
    ...player,
    stats: { ...player.stats, atk: Math.max(1, atk), def: Math.max(0, def) },
    buffs: updatedBuffs,
  }
}

/** Score bonus when dropping (recycling) an item */
export const DROP_SCORE_BONUS = 15

/** Item rarity color based on type */
export function itemColor(type: ItemType): string {
  switch (type) {
    case ItemType.HEAL_POTION:
      return "#00d084"
    case ItemType.ATK_SCROLL:
      return "#ff6b6b"
    case ItemType.DEF_SCROLL:
      return "#64b5f6"
    case ItemType.MAP_REVEAL:
      return "#ffa726"
    case ItemType.KEY:
      return "#ffd700"
    case ItemType.SHIELD:
      return "#e040fb"
    default:
      return "#a0aeb8"
  }
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max - 1) + "\u2026" : str
}

function simpleHash(str: string): number {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0
  }
  return Math.abs(hash)
}
