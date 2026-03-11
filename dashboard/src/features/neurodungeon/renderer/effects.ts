/* ------------------------------------------------------------------ */
/*  Visual effects — particles, screen shake, flash                    */
/*  Stateless renderer: pass particles array, draws them on canvas     */
/* ------------------------------------------------------------------ */

import { TILE_SIZE } from "../engine/types"

// --------------- Types ---------------

export interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  life: number    // 0..1 (1 = just spawned, 0 = dead)
  decay: number   // how fast life decreases per frame
  color: string
  size: number
  type: "spark" | "float" | "splat"
}

export interface ScreenShake {
  intensity: number  // pixels
  duration: number   // frames remaining
  dx: number
  dy: number
}

export interface ScreenFlash {
  color: string
  alpha: number
  decay: number  // per frame
}

export interface EffectsState {
  particles: Particle[]
  shake: ScreenShake | null
  flash: ScreenFlash | null
}

// --------------- Initial state ---------------

export function createEffectsState(): EffectsState {
  return { particles: [], shake: null, flash: null }
}

// --------------- Emitters ---------------

export function emitDamageParticles(
  x: number,
  y: number,
  count: number,
  color: string,
): Particle[] {
  return Array.from({ length: count }, () => ({
    x: x * TILE_SIZE + TILE_SIZE / 2,
    y: y * TILE_SIZE + TILE_SIZE / 2,
    vx: (Math.random() - 0.5) * 4,
    vy: (Math.random() - 0.5) * 4 - 1,
    life: 1,
    decay: 0.03 + Math.random() * 0.02,
    color,
    size: 2 + Math.random() * 2,
    type: "spark" as const,
  }))
}

export function emitPickupParticles(x: number, y: number): Particle[] {
  return Array.from({ length: 8 }, (_, i) => {
    const angle = (i / 8) * Math.PI * 2
    return {
      x: x * TILE_SIZE + TILE_SIZE / 2,
      y: y * TILE_SIZE + TILE_SIZE / 2,
      vx: Math.cos(angle) * 2,
      vy: Math.sin(angle) * 2 - 1,
      life: 1,
      decay: 0.04,
      color: "#ffd700",
      size: 3,
      type: "float" as const,
    }
  })
}

export function emitDeathParticles(x: number, y: number): Particle[] {
  return Array.from({ length: 16 }, () => ({
    x: x * TILE_SIZE + TILE_SIZE / 2,
    y: y * TILE_SIZE + TILE_SIZE / 2,
    vx: (Math.random() - 0.5) * 6,
    vy: (Math.random() - 0.5) * 6,
    life: 1,
    decay: 0.02,
    color: "#ff6b6b",
    size: 2 + Math.random() * 3,
    type: "splat" as const,
  }))
}

export function emitLevelUpParticles(x: number, y: number): Particle[] {
  return Array.from({ length: 12 }, (_, i) => {
    const angle = (i / 12) * Math.PI * 2
    return {
      x: x * TILE_SIZE + TILE_SIZE / 2,
      y: y * TILE_SIZE + TILE_SIZE / 2,
      vx: Math.cos(angle) * 1.5,
      vy: -2 - Math.random() * 2,
      life: 1,
      decay: 0.025,
      color: "#00d084",
      size: 3,
      type: "float" as const,
    }
  })
}

// --------------- Shake & flash ---------------

export function createShake(intensity: number = 4, duration: number = 8): ScreenShake {
  return { intensity, duration, dx: 0, dy: 0 }
}

export function createFlash(color: string = "#ff6b6b", alpha: number = 0.3): ScreenFlash {
  return { color, alpha, decay: 0.05 }
}

// --------------- Update (per animation frame) ---------------

export function tickEffects(state: EffectsState): EffectsState {
  // Update particles
  const particles = state.particles
    .map((p) => ({
      ...p,
      x: p.x + p.vx,
      y: p.y + p.vy,
      vy: p.type === "float" ? p.vy : p.vy + 0.1, // gravity for sparks
      life: p.life - p.decay,
    }))
    .filter((p) => p.life > 0)

  // Update shake
  let shake = state.shake
  if (shake) {
    shake = {
      ...shake,
      duration: shake.duration - 1,
      dx: (Math.random() - 0.5) * shake.intensity,
      dy: (Math.random() - 0.5) * shake.intensity,
    }
    if (shake.duration <= 0) shake = null
  }

  // Update flash
  let flash = state.flash
  if (flash) {
    flash = { ...flash, alpha: flash.alpha - flash.decay }
    if (flash.alpha <= 0) flash = null
  }

  return { particles, shake, flash }
}

// --------------- Render (draw on canvas context) ---------------

export function renderParticles(
  ctx: CanvasRenderingContext2D,
  particles: Particle[],
  camX: number,
  camY: number,
): void {
  for (const p of particles) {
    ctx.globalAlpha = p.life
    ctx.fillStyle = p.color
    ctx.beginPath()
    ctx.arc(p.x - camX, p.y - camY, p.size * p.life, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

export function renderFlash(
  ctx: CanvasRenderingContext2D,
  flash: ScreenFlash,
  width: number,
  height: number,
): void {
  ctx.globalAlpha = flash.alpha
  ctx.fillStyle = flash.color
  ctx.fillRect(0, 0, width, height)
  ctx.globalAlpha = 1
}
