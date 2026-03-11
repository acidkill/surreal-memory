import type { OracleCard } from "../engine/types"

interface CardFaceProps {
  card: OracleCard
  className?: string
}

export function CardFace({ card, className = "" }: CardFaceProps) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-[#16140f] p-5 ${className}`}
      style={{ backfaceVisibility: "hidden" }}
    >
      {/* Background gradient */}
      <div
        className={`absolute inset-0 bg-gradient-to-b ${card.suit.bg} opacity-60`}
      />

      {/* Geometric art per suit */}
      <div className="absolute right-3 top-3 opacity-10">
        <span className="text-7xl font-bold" style={{ color: card.suit.color }}>
          {card.suit.symbol}
        </span>
      </div>

      {/* Content */}
      <div className="relative z-10 flex h-full flex-col">
        {/* Title */}
        <div className="mb-1 text-center">
          <span
            className="text-xs font-semibold uppercase tracking-[0.2em]"
            style={{ color: card.suit.color }}
          >
            ✦ {card.title} ✦
          </span>
        </div>

        {/* Suit symbol */}
        <div className="my-4 flex justify-center">
          <div
            className="flex size-16 items-center justify-center rounded-full border-2"
            style={{
              borderColor: card.suit.color + "40",
              backgroundColor: card.suit.color + "15",
            }}
          >
            <span className="text-3xl" style={{ color: card.suit.color }}>
              {card.suit.symbol}
            </span>
          </div>
        </div>

        {/* Divider */}
        <div
          className="mx-auto mb-3 h-px w-3/4"
          style={{ backgroundColor: card.suit.color + "30" }}
        />

        {/* Memory content */}
        <div className="mb-auto flex-1">
          <p className="line-clamp-3 text-center text-sm leading-relaxed text-white/80">
            &ldquo;{card.content}&rdquo;
          </p>
        </div>

        {/* Stats row */}
        <div className="mt-4 flex items-center justify-between text-xs text-white/50">
          <span title="Activation">⚡ {card.activation}</span>
          <span title="Connections">🔗 {card.connectionCount}</span>
          <span title="Age">📅 {card.age}</span>
        </div>

        {/* Suit badge */}
        <div className="mt-2 text-center">
          <span
            className="inline-block rounded-full px-3 py-0.5 text-xs font-medium"
            style={{
              backgroundColor: card.suit.color + "20",
              color: card.suit.color,
            }}
          >
            {card.suit.symbol} {card.suitKey}
          </span>
        </div>
      </div>
    </div>
  )
}
