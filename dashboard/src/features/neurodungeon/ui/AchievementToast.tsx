/* ------------------------------------------------------------------ */
/*  Achievement toast — popup when achievement unlocked               */
/* ------------------------------------------------------------------ */

import { useEffect, useState } from "react"
import type { Achievement } from "../engine/achievements"

interface AchievementToastProps {
  achievement: Achievement | null
  onDismiss: () => void
}

export function AchievementToast({ achievement, onDismiss }: AchievementToastProps) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (!achievement) {
      setVisible(false)
      return
    }
    setVisible(true)
    const timer = setTimeout(() => {
      setVisible(false)
      setTimeout(onDismiss, 300)
    }, 3000)
    return () => clearTimeout(timer)
  }, [achievement, onDismiss])

  if (!achievement) return null

  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        left: "50%",
        transform: `translateX(-50%) translateY(${visible ? "0" : "-60px"})`,
        opacity: visible ? 1 : 0,
        transition: "transform 300ms ease, opacity 300ms ease",
        zIndex: 40,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "10px 20px",
          background: "linear-gradient(135deg, rgba(26, 35, 50, 0.95), rgba(18, 26, 32, 0.95))",
          border: "1px solid #ffd700",
          borderRadius: 10,
          boxShadow: "0 4px 20px rgba(255, 215, 0, 0.2)",
        }}
      >
        <span
          style={{
            fontSize: 22,
            fontFamily: "JetBrains Mono, monospace",
            color: "#ffd700",
            fontWeight: 700,
            width: 32,
            height: 32,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(255, 215, 0, 0.1)",
            borderRadius: 6,
          }}
        >
          {achievement.icon}
        </span>
        <div>
          <div
            style={{
              fontSize: 11,
              color: "#ffd700",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: 1,
              fontFamily: "Space Grotesk, sans-serif",
            }}
          >
            Achievement Unlocked
          </div>
          <div
            style={{
              fontSize: 14,
              color: "#f0f0f0",
              fontWeight: 600,
              fontFamily: "Inter, sans-serif",
            }}
          >
            {achievement.name}
          </div>
          <div
            style={{
              fontSize: 11,
              color: "#a0aeb8",
              fontFamily: "Inter, sans-serif",
            }}
          >
            {achievement.description}
          </div>
        </div>
      </div>
    </div>
  )
}
