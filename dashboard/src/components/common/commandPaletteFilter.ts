/**
 * The palette renders with `shouldFilter={false}` because fibers and neurons are
 * matched by hand (neurons server-side, fibers by summary). That also switched
 * off cmdk's filtering for the static items, so pages and Pro hints stayed on
 * screen whatever was typed. Every static item goes through this instead.
 */
export function matchesQuery(label: string, query: string): boolean {
  const q = query.trim().toLowerCase()
  return q.length === 0 || label.toLowerCase().includes(q)
}
