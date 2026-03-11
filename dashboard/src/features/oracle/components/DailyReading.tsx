import { useEffect, useState } from "react"
import { FlipCard } from "./FlipCard"
import { generateDailyReading } from "../engine/reading-engine"
import type { OracleCard, DailyReading as DailyReadingType } from "../engine/types"
import { useTranslation } from "react-i18next"

interface DailyReadingProps {
  cards: OracleCard[]
  brainName: string
}

const POSITIONS = ["past", "present", "future"] as const

export function DailyReading({ cards, brainName }: DailyReadingProps) {
  const { t } = useTranslation()
  const [reading, setReading] = useState<DailyReadingType | null>(null)
  const [revealed, setRevealed] = useState(0)

  useEffect(() => {
    const result = generateDailyReading(cards, brainName)
    setReading(result)
    setRevealed(0)
  }, [cards, brainName])

  if (!reading) return null

  const positionCards = [reading.past, reading.present, reading.future]

  return (
    <div className="flex flex-col items-center gap-8">
      <p className="text-sm text-muted-foreground">
        {t("oracle.dailyHint")}
      </p>

      <div className="flex flex-wrap justify-center gap-6">
        {positionCards.map((card, i) => (
          <div key={card.id} className="flex flex-col items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {t(`oracle.${POSITIONS[i]}`)}
            </span>
            <FlipCard
              card={card}
              autoFlipDelay={800 + i * 500}
              onFlip={() => setRevealed((prev) => Math.max(prev, i + 1))}
              className="h-[340px] w-[240px]"
            />
          </div>
        ))}
      </div>

      {/* Interpretation appears after all 3 cards revealed */}
      {revealed >= 3 && (
        <div className="max-w-xl animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-5">
            <p className="text-center text-sm leading-relaxed text-foreground/80">
              {reading.interpretation}
            </p>
            <p className="mt-3 text-center text-xs text-muted-foreground">
              {t("oracle.readingDate", { date: reading.date, brain: reading.brainName })}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
