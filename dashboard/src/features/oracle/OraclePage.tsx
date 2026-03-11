import { useState } from "react"
import { Sparkles } from "lucide-react"
import { ModeSelector } from "./components/ModeSelector"
import { FlipCard } from "./components/FlipCard"
import { useOracleData } from "./hooks/useOracleData"
import type { OracleMode } from "./engine/types"

export default function OraclePage() {
  const [mode, setMode] = useState<OracleMode>("daily")
  const { cards, isLoading } = useOracleData()

  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <h1 className="font-display text-2xl font-bold">Brain Oracle</h1>
        <p className="text-muted-foreground">Channeling your memories...</p>
      </div>
    )
  }

  if (cards.length < 3) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 p-6 pt-24">
        <Sparkles className="size-12 text-muted-foreground/40" />
        <h2 className="font-display text-xl font-semibold text-muted-foreground">
          Your brain needs more memories
        </h2>
        <p className="max-w-md text-center text-sm text-muted-foreground/70">
          The Oracle requires at least 3 memories to unlock. Ask your AI agent
          to remember decisions, insights, and learnings — then return for your
          reading.
        </p>
      </div>
    )
  }

  // Phase 1: show 3 sample cards with staggered flip
  const sampleCards = cards.slice(0, 3)

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold">Brain Oracle</h1>
        <ModeSelector mode={mode} onModeChange={setMode} />
      </div>

      <div className="flex flex-col items-center gap-8">
        {mode === "daily" && (
          <>
            <p className="text-sm text-muted-foreground">
              Tap each card to reveal your reading
            </p>
            <div className="flex flex-wrap justify-center gap-6">
              {sampleCards.map((card, i) => (
                <div key={card.id} className="flex flex-col items-center gap-2">
                  <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    {["Past", "Present", "Future"][i]}
                  </span>
                  <FlipCard
                    card={card}
                    autoFlipDelay={800 + i * 500}
                    className="h-[340px] w-[240px]"
                  />
                </div>
              ))}
            </div>
          </>
        )}

        {mode === "whatif" && (
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-muted-foreground">
              What if these memories collided?
            </p>
            <div className="flex flex-wrap justify-center gap-6">
              {sampleCards.map((card) => (
                <FlipCard
                  key={card.id}
                  card={card}
                  className="h-[340px] w-[240px]"
                />
              ))}
            </div>
          </div>
        )}

        {mode === "matchup" && (
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-muted-foreground">
              Which memory is stronger? Tap to reveal, then decide.
            </p>
            <div className="flex flex-wrap justify-center gap-8">
              {sampleCards.slice(0, 2).map((card) => (
                <FlipCard
                  key={card.id}
                  card={card}
                  className="h-[340px] w-[240px]"
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
