let opener: (() => void) | null = null

export function openUpgradeModal() {
  opener?.()
}

export function setUpgradeModalOpener(value: (() => void) | null) {
  opener = value
}
