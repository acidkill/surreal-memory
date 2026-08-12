import { useEffect, useState } from "react"

/**
 * Trailing-edge debounce for a value.
 *
 * Used by the graph's node-count slider: dragging it emits a value per pointer
 * move, and each distinct value is a query key, so without this a single drag
 * from 100 to 2000 would fire dozens of `/api/graph` requests — each of which
 * makes the server rebuild a graph it is about to throw away.
 */
export function useDebounce<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])

  return debounced
}
