import { SidebarSimple, Sun, Moon, Monitor, Question } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { useStats, useHealthCheck } from "@/api/hooks/useDashboard"
import { Badge } from "@/components/ui/badge"
import { useTranslation } from "react-i18next"
import { CommandPalette } from "@/components/common/CommandPalette"

const themeIcons = {
  light: Sun,
  dark: Moon,
  system: Monitor,
} as const

const themeKeys = {
  light: "common.lightMode",
  dark: "common.darkMode",
  system: "common.systemTheme",
} as const

export function TopBar() {
  const { sidebarOpen, toggleSidebar, theme, cycleTheme } = useLayoutStore()
  const { data: stats } = useStats()
  const { data: healthCheck } = useHealthCheck()
  const { t } = useTranslation()

  const ThemeIcon = themeIcons[theme]

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-4 border-b border-border bg-background/80 px-4 backdrop-blur-sm">
      {/* Sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleSidebar}
        aria-label={sidebarOpen ? t("common.collapseSidebar") : t("common.expandSidebar")}
      >
        {sidebarOpen ? (
          <SidebarSimple className="size-5" weight="bold" />
        ) : (
          <SidebarSimple className="size-5" />
        )}
      </Button>

      {/* Active brain indicator */}
      {stats?.active_brain && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{t("common.brain")}:</span>
          <Badge variant="secondary" className="font-mono text-xs">
            {stats.active_brain}
          </Badge>
        </div>
      )}

      {/* Command palette trigger */}
      <CommandPalette />

      {/* Spacer */}
      <div className="flex-1" />

      {/* Help / Guide link */}
      <Button
        variant="ghost"
        size="icon"
        asChild
        aria-label="Quickstart Guide"
        title="Quickstart Guide"
      >
        <a
          href="https://github.com/acidkill/surreal-memory/blob/main/docs/guides/quickstart-guide.md"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Question className="size-4" />
        </a>
      </Button>

      {/* Theme toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={cycleTheme}
        aria-label={t(themeKeys[theme])}
        title={t(themeKeys[theme])}
        data-testid="theme-toggle"
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
