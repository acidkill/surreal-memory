# Phase 4: Dashboard Guide Card

## Goal
Small card in dashboard homepage that links to the quickstart guide.
New users landing on dashboard immediately know where docs are.

## Tasks

- [ ] 4.1 Create `dashboard/src/components/GuideCard.tsx`:
  - Small card (not full-width, fits in sidebar or top banner area)
  - Icon: 📖 or book icon
  - Title: "Quickstart Guide"
  - Subtitle: "Learn setup, recall, cognitive tools & more"
  - CTA button: "Open Guide →" (opens in new tab)
  - Dismissible (localStorage flag `guide-card-dismissed`)
  - Shows only when brain has <50 neurons (new user detection)
- [ ] 4.2 Add GuideCard to dashboard main layout:
  - Position: top of Overview page, above stats cards
  - Collapses to thin banner after first dismiss
  - Re-shows if user clicks "Help" or "?" button
- [ ] 4.3 Style with existing dashboard design tokens:
  - Subtle gradient border (accent color)
  - Dark mode compatible
  - Responsive (full-width on mobile)
- [ ] 4.4 Add `GUIDE_URL` constant to dashboard config
- [ ] 4.5 Add "?" help button in header that reopens guide card

## Acceptance Criteria
- [ ] Card shows for new users (<50 neurons)
- [ ] Dismiss persists across page reloads
- [ ] Link opens correct docs URL in new tab
- [ ] Works in dark mode
- [ ] Mobile responsive
- [ ] "?" button in header always available

## Files Touched
- `dashboard/src/components/GuideCard.tsx` — new
- `dashboard/src/pages/Overview.tsx` (or equivalent main page) — modify
- `dashboard/src/components/Header.tsx` (or layout) — modify (add ? button)

## Dependencies
- Phase 2 (guide page URL must be live)
