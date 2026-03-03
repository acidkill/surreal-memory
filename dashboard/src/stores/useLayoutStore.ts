import { create } from "zustand"

type Theme = "light" | "dark" | "system"

interface LayoutState {
  sidebarOpen: boolean
  theme: Theme
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setTheme: (theme: Theme) => void
  cycleTheme: () => void
}

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "system"
  const stored = localStorage.getItem("nm-theme")
  if (stored === "light" || stored === "dark" || stored === "system") return stored
  return "system"
}

function applyTheme(theme: Theme) {
  const root = document.documentElement
  const isDark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)

  root.classList.toggle("dark", isDark)
  localStorage.setItem("nm-theme", theme)
}

// Apply on load (before React hydrates) to avoid flash
applyTheme(getInitialTheme())

export const useLayoutStore = create<LayoutState>((set, get) => ({
  sidebarOpen: true,
  theme: getInitialTheme(),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setTheme: (theme) => {
    applyTheme(theme)
    set({ theme })
  },
  cycleTheme: () => {
    const order: Theme[] = ["light", "dark", "system"]
    const current = get().theme
    const next = order[(order.indexOf(current) + 1) % order.length]
    applyTheme(next)
    set({ theme: next })
  },
}))

// Listen for OS theme changes when in "system" mode
if (typeof window !== "undefined") {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    const { theme } = useLayoutStore.getState()
    if (theme === "system") applyTheme("system")
  })
}
