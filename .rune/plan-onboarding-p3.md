# Phase 3: CLI → Guide Link

## Goal
After `nmem init` (both regular and --full), show a small banner pointing to the quickstart guide.
User knows exactly where to go for help.

## Tasks

- [ ] 3.1 Add guide URL constant to `cli/setup.py`:
  ```python
  QUICKSTART_URL = "https://nhadaututtheky.github.io/neural-memory/guides/quickstart/"
  ```
- [ ] 3.2 After init completes, print styled banner:
  ```
  ╭──────────────────────────────────────────────╮
  │  🧠 Neural Memory is ready!                  │
  │                                               │
  │  📖 Quickstart Guide:                         │
  │  https://nhadaututtheky.github.io/.../quickstart │
  │                                               │
  │  🩺 Run `nmem doctor` to check your setup     │
  ╰──────────────────────────────────────────────╯
  ```
- [ ] 3.3 After `nmem doctor` with issues, suggest guide:
  ```
  💡 See full setup guide: <URL>
  ```
- [ ] 3.4 Add `--quiet` flag to suppress banner (for CI/scripting)

## Acceptance Criteria
- [ ] Banner displays after successful init
- [ ] URL is correct and clickable in terminal
- [ ] `--quiet` suppresses banner
- [ ] Banner works on Windows (no broken box-drawing chars in cmd.exe — use ASCII fallback)

## Files Touched
- `src/neural_memory/cli/commands/tools.py` — modify
- `src/neural_memory/cli/setup.py` — modify (add constant + banner function)
- `src/neural_memory/cli/doctor.py` — modify (add guide suggestion)

## Dependencies
- Phase 1 (doctor command exists)
- Phase 2 (guide URL is live)
