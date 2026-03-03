import { PanelLeftClose, PanelLeft, Sun, Moon, Monitor } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { useStats, useHealthCheck } from "@/api/hooks/useDashboard"
import { Badge } from "@/components/ui/badge"

const themeIcons = {
  light: Sun,
  dark: Moon,
  system: Monitor,
} as const

const themeLabels = {
  light: "Light mode",
  dark: "Dark mode",
  system: "System theme",
} as const

export function TopBar() {
  const { sidebarOpen, toggleSidebar, theme, cycleTheme } = useLayoutStore()
  const { data: stats } = useStats()
  const { data: healthCheck } = useHealthCheck()

  const ThemeIcon = themeIcons[theme]

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-4 border-b border-border bg-background/80 px-4 backdrop-blur-sm">
      {/* Sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleSidebar}
        aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
      >
        {sidebarOpen ? (
          <PanelLeftClose className="size-5" />
        ) : (
          <PanelLeft className="size-5" />
        )}
      </Button>

      {/* Active brain indicator */}
      {stats?.active_brain && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Brain:</span>
          <Badge variant="secondary" className="font-mono text-xs">
            {stats.active_brain}
          </Badge>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Theme toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={cycleTheme}
        aria-label={themeLabels[theme]}
        title={themeLabels[theme]}
      >
        <ThemeIcon className="size-4" />
      </Button>

      {/* Version */}
      {healthCheck?.version && (
        <span className="text-xs text-muted-foreground font-mono">
          v{healthCheck.version}
        </span>
      )}
    </header>
  )
}
