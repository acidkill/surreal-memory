import { Sparkles, Shuffle, Swords } from "lucide-react"
import type { OracleMode } from "../engine/types"

const MODES: { key: OracleMode; label: string; icon: typeof Sparkles }[] = [
  { key: "daily", label: "Daily Reading", icon: Sparkles },
  { key: "whatif", label: "What If", icon: Shuffle },
  { key: "matchup", label: "Matchup", icon: Swords },
]

interface ModeSelectorProps {
  mode: OracleMode
  onModeChange: (mode: OracleMode) => void
}

export function ModeSelector({ mode, onModeChange }: ModeSelectorProps) {
  return (
    <div className="flex gap-2">
      {MODES.map(({ key, label, icon: Icon }) => (
        <button
          key={key}
          onClick={() => onModeChange(key)}
          className={`flex cursor-pointer items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
            mode === key
              ? "bg-primary/15 text-primary ring-1 ring-primary/30"
              : "text-muted-foreground hover:bg-accent hover:text-foreground"
          }`}
        >
          <Icon className="size-4" />
          {label}
        </button>
      ))}
    </div>
  )
}
