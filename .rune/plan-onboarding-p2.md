# Phase 2: Guide Page — MkDocs Interactive Quickstart

## Goal
Interactive quickstart guide with animated terminal demo on existing docs site.
Pre-marketing asset: users see value before reading technical docs.

## Design Reference
Inspired by VividKit CLI guide (vividkit.dev/guides/cli):
- Animated terminal hero (typing + checkmarks)
- 3-step quick start (not 26 steps)
- Feature cards with mini-demos
- Pro tips callouts
- Dark mode native

## Tasks

### Page Structure
- [ ] 2.1 Create `docs/guides/quickstart.md` — MkDocs page with custom HTML/JS blocks
- [ ] 2.2 Hero section: animated terminal showing `nmem init --full` output
  - Typing animation (25ms/char)
  - Checkmarks appear sequentially (300ms stagger)
  - Replay button
- [ ] 2.3 "3 Steps" section:
  - `pip install neural-memory` → Install
  - `nmem init --full` → Everything configured
  - Restart IDE → Brain is live
  - Each step: number badge + code block with copy button + 1-line explanation
- [ ] 2.4 "What Your Brain Looks Like" section:
  - Animated `nmem doctor` output
  - Show healthy brain with real-looking stats
- [ ] 2.5 "Core Features" cards (4 cards):
  - Remember: `nmem_remember` with rich context example
  - Recall: spreading activation description + example
  - Doctor: health check output
  - Surface: .nm knowledge surface explanation
- [ ] 2.6 "Power User" section:
  - Cognitive tools overview (hypothesize → evidence → predict → verify)
  - Cloud sync mention
  - `nmem serve` background mode
- [ ] 2.7 "Pro Tips" callout boxes (3-4 tips)
- [ ] 2.8 Footer CTA: GitHub star + Discord/Discussions link

### Styling
- [ ] 2.9 Create `docs/assets/css/quickstart.css`:
  - Terminal container (dark bg, monospace, border-radius)
  - Typing cursor animation (blink)
  - Step number badges (circular, accent color)
  - Feature cards (grid, hover scale)
  - Callout boxes (tip/warning/info variants)
- [ ] 2.10 Create `docs/assets/js/quickstart.js`:
  - Typing animation engine (char-by-char with configurable speed)
  - Intersection Observer for scroll-triggered animations
  - Replay button handler
  - Copy button with checkmark feedback

### Integration
- [ ] 2.11 Add page to `mkdocs.yml` nav under Guides
- [ ] 2.12 Add custom CSS/JS to mkdocs.yml `extra_css` / `extra_javascript`
- [ ] 2.13 Verify dark/light mode works with Material theme

## Acceptance Criteria
- [ ] Page loads in <2s, no external dependencies (no CDN)
- [ ] Terminal animation plays on scroll into view
- [ ] All code blocks have working copy buttons
- [ ] Dark mode looks correct
- [ ] Mobile responsive (stacks cards, terminal scrolls horizontally)
- [ ] Replay button resets and replays animation

## Files Touched
- `docs/guides/quickstart.md` — new
- `docs/assets/css/quickstart.css` — new
- `docs/assets/js/quickstart.js` — new
- `mkdocs.yml` — modify (nav + extra_css/js)

## Dependencies
- Phase 1 done (so guide shows real `--full` and `doctor` commands)
