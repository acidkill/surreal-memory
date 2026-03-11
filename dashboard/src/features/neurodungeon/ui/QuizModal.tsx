/* ------------------------------------------------------------------ */
/*  Memory Recall Quiz — fill-the-blank from neuron content            */
/*  Correct = bonus loot. Wrong = enemy ambush.                        */
/* ------------------------------------------------------------------ */

import { useState, useMemo } from "react"

interface QuizModalProps {
  neuronContent: string
  neuronType: string
  onAnswer: (correct: boolean) => void
}

export function QuizModal({ neuronContent, neuronType, onAnswer }: QuizModalProps) {
  const quiz = useMemo(() => generateQuiz(neuronContent), [neuronContent])
  const [selected, setSelected] = useState<number | null>(null)
  const [answered, setAnswered] = useState(false)

  if (!quiz) {
    onAnswer(true)
    return null
  }

  const isCorrect = selected === quiz.correctIndex

  const handleSubmit = () => {
    if (selected === null) return
    setAnswered(true)
    setTimeout(() => onAnswer(isCorrect), 1200)
  }

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.85)",
        zIndex: 25,
      }}
    >
      <div
        style={{
          maxWidth: 460,
          padding: "24px 32px",
          background: "#121a20",
          border: "1px solid #2a3f52",
          borderRadius: 12,
          color: "#f0f0f0",
          fontFamily: "Inter, sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
          <span style={{ fontSize: 20 }}>{"?"}</span>
          <h3
            style={{
              margin: 0,
              fontSize: 16,
              fontWeight: 700,
              fontFamily: "Space Grotesk, sans-serif",
              color: "#64b5f6",
            }}
          >
            Memory Recall Challenge
          </h3>
        </div>

        <p style={{ color: "#a0aeb8", fontSize: 11, margin: "0 0 12px 0" }}>
          Type: {neuronType}
        </p>

        {/* Question with blank */}
        <p
          style={{
            fontSize: 14,
            lineHeight: 1.6,
            color: "#e0e0e0",
            background: "rgba(42, 63, 82, 0.3)",
            padding: "12px 16px",
            borderRadius: 8,
            marginBottom: 16,
          }}
        >
          {quiz.questionParts[0]}
          <span
            style={{
              display: "inline-block",
              minWidth: 80,
              borderBottom: "2px solid #2196f3",
              color: answered ? (isCorrect ? "#00d084" : "#ff6b6b") : "#2196f3",
              fontWeight: 700,
              padding: "0 4px",
            }}
          >
            {answered ? quiz.answer : "______"}
          </span>
          {quiz.questionParts[1]}
        </p>

        {/* Options */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
          {quiz.options.map((opt, i) => {
            let bg = "rgba(42, 63, 82, 0.3)"
            let border = "1px solid #2a3f52"
            if (answered && i === quiz.correctIndex) {
              bg = "rgba(0, 208, 132, 0.15)"
              border = "1px solid #00d084"
            } else if (answered && i === selected && !isCorrect) {
              bg = "rgba(255, 107, 107, 0.15)"
              border = "1px solid #ff6b6b"
            } else if (i === selected && !answered) {
              bg = "rgba(33, 150, 243, 0.15)"
              border = "1px solid #2196f3"
            }

            return (
              <button
                key={i}
                onClick={() => !answered && setSelected(i)}
                style={{
                  background: bg,
                  border,
                  borderRadius: 8,
                  padding: "8px 16px",
                  color: "#e0e0e0",
                  fontSize: 13,
                  cursor: answered ? "default" : "pointer",
                  textAlign: "left",
                  fontFamily: "Inter, sans-serif",
                  transition: "background 150ms ease, border 150ms ease",
                }}
              >
                {opt}
              </button>
            )
          })}
        </div>

        {/* Result message */}
        {answered && (
          <p
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: isCorrect ? "#00d084" : "#ff6b6b",
              margin: "0 0 12px 0",
              fontFamily: "Space Grotesk, sans-serif",
            }}
          >
            {isCorrect ? "Correct! Bonus loot awaits." : "Wrong! An enemy emerges from the shadows..."}
          </p>
        )}

        {!answered && (
          <button
            onClick={handleSubmit}
            disabled={selected === null}
            style={{
              background: selected !== null
                ? "linear-gradient(135deg, #2196f3, #00d084)"
                : "#334155",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "8px 24px",
              fontSize: 14,
              fontWeight: 700,
              cursor: selected !== null ? "pointer" : "not-allowed",
              fontFamily: "Space Grotesk, sans-serif",
            }}
          >
            Submit Answer
          </button>
        )}
      </div>
    </div>
  )
}

// --------------- Quiz generation ---------------

interface Quiz {
  questionParts: [string, string] // before and after blank
  answer: string
  options: string[]
  correctIndex: number
}

function generateQuiz(content: string): Quiz | null {
  // Split content into words, pick a meaningful word to blank out
  const words = content.split(/\s+/).filter((w) => w.length > 3)
  if (words.length < 4) return null

  // Pick a word from the middle of content (more meaningful)
  const middleStart = Math.floor(words.length * 0.25)
  const middleEnd = Math.floor(words.length * 0.75)
  const targetIdx = middleStart + Math.floor(Math.random() * (middleEnd - middleStart))
  const answer = words[targetIdx]!.replace(/[.,;:!?'"()]/g, "")

  if (answer.length < 3) return null

  // Find position in original content
  const blankPos = content.indexOf(answer)
  if (blankPos === -1) return null

  const before = content.slice(0, blankPos)
  const after = content.slice(blankPos + answer.length)

  // Generate distractors from other words in content
  const otherWords = words
    .filter((w) => w.replace(/[.,;:!?'"()]/g, "") !== answer && w.length > 2)
    .map((w) => w.replace(/[.,;:!?'"()]/g, ""))

  // Ensure we have enough distractors
  const fallbacks = ["memory", "neuron", "synapse", "pattern", "signal", "data", "model"]
  while (otherWords.length < 3) {
    const fb = fallbacks[Math.floor(Math.random() * fallbacks.length)]!
    if (!otherWords.includes(fb) && fb !== answer) {
      otherWords.push(fb)
    }
  }

  // Pick 3 random distractors
  const shuffled = otherWords.sort(() => Math.random() - 0.5).slice(0, 3)

  // Insert correct answer at random position
  const correctIndex = Math.floor(Math.random() * 4)
  const options = [...shuffled]
  options.splice(correctIndex, 0, answer)

  return {
    questionParts: [before, after],
    answer,
    options,
    correctIndex,
  }
}
