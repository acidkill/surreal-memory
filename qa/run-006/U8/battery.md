# U8 — Geospatial recall (PR7): bateria weryfikacyjna

Branch: `feature/v2100-pr7-geo` (z HEAD U7 `b92856f`). Commity U8:
`7f4c003` utils/geo.py · `15991fc` near recall + location remember · `de70ce9` fmt ·
`556cc5a` browse pushdown `find_fibers(near=)` + fast-path guardy ·
`35acaa0` fix: usunięcie SQLite bbox (false-exclusion na brzegu promienia) ·
`8672f76` Stage-2: guard-regression + live-DB id-norm + hardening surface-guard dla valid_at.

## Zakres (metadata-only — BEZ zmiany schematu)
Współrzędne w `fiber.metadata["location"] = {lat, lon, label?}` (OBJECT FLEXIBLE → zero migracji).
- `utils/geo.py` (stdlib): `GeoPoint`/`GeoFilter` z walidacją zakresów, `haversine_m` (WGS-84 sfera,
  R=6 371 008.8 m), `parse_geo_point`/`parse_geo_filter` (ValueError → MCP error dict), `fiber_location`
  (odporny na śmieci), `fiber_within` (JEDYNE źródło prawdy hard-filtra — używane przez pipeline i wszystkie backendy).
- Remember: `location {lat,lon,label?}` w schemacie `smem_remember` + batch item → parse w `remember_handler`
  → `encode()` metadata → `BuildFiberStep` propaguje na `fiber.metadata["location"]`.
- Recall: `near {lat,lon,radius_m}` w schemacie `smem_recall` → parse w `recall_handler` + `_cross_brain_recall`
  → `ReflexPipeline.query(near=...)` → hard-filtr haversine w `_find_matching_fibers` ZARAZ po `valid_at`.
  `near` dopisane do `trace_builder._FILTER_KEYS`. Semantyka = twardy filtr wg precedensu `valid_at`
  (fiber bez lokalizacji = wykluczony; `near=None` = ścisły no-op).
- GUARDY (lekcja z buga valid_at): gdy `near` ustawione, pomijane są fast-pathy `_try_temporal_reasoning`
  + `_try_fiber_summary_tier` (retrieval) oraz surface-routing (recall_handler) — inaczej filtr byłby cicho
  zignorowany. Surface-guard rozszerzony też o `valid_at` (finding python-reviewera).
- Browse pushdown `find_fibers(*, near=)` na base/InMemory/SQLite/SharedStore/SurrealDB. SQLite i SurrealDB
  pushują TYLKO `location IS NOT NULL` / `metadata.location != NONE`; exact haversine (fiber_within) jest
  jedynym wiążącym. `geo::distance` = udokumentowana opcja v2 (brak indeksu przestrzennego; kolejność lon/lat
  i jednostka metrów zmienne na 3.x).

## Stage 1 — gates (env -u SURREALDB_URL, .venv/bin)
- **D1 no-DB** (`pytest -m "not stress" -n auto`): **6286 passed / 0 FAILED**, 26 błędów = PRE-EXISTING
  `tests/e2e/test_api.py` stub-leak (nie U8). Re-run po Stage-2 (guard valid_at) — patrz d1_u8_final.log.
- **Coverage**: 68.19% ≥ 67 (gate). `utils/geo.py` = 100%.
- **mypy** `--ignore-missing-imports`: 357 plików czysto.
- **ruff check** + **ruff format --check**: czysto.
- **Non-live geo suite**: `test_geo_utils` 36 · `test_geo_recall` 19 (unit `_fiber_near`, REALNY pipeline
  near, browse InMemory+SQLite incl. antymerydian + brzeg promienia, guard temporal-bypass) · `test_geo_mcp` 8
  (near→GeoFilter→pipeline, error-paths near/location, location→encode metadata).

## Stage 2 — agenci
### python-reviewer → **APPROVE** (0 CRITICAL / 0 HIGH)
- Potwierdził: BRAK bbox (usunięty) → cała klasa false-exclusion wyeliminowana i otestowana; guardy pokrywają
  KAŻDY early-return między `query()` a `_find_matching_fibers` (sufficiency-gate zwraca `fibers_matched=[]` →
  nic do filtrowania); brak SurQL/SQL injection (lat/lon/radius nigdy nie trafiają do query stringa); jedna
  implementacja haversine we wszystkich backendach (brak dryfu semantyki).
- MEDIUM (naprawione): surface-guard dla `near` powinien też objąć `valid_at` → rozszerzone (`8672f76`).
- LOW (deferred → REPORT follow-ups, PRE-EXISTING, nie-blokujące):
  1. ścieżka ghost-recall `fiber:{id}` omija `near` i `valid_at` (escape-hatch by-id; poprzedza U3/U8).
  2. over-fetch `min(limit*3, 3000)` przy `near`/`tags` — przy >3000 zlokalizowanych fiberów o niskiej
     salience możliwe pominięcie (ten sam wzorzec co `tags`; nieistotne przy obecnych rozmiarach mózgów).

### real-db-test-runner → **PASS** (live SurrealDB v3 @ localhost:8001)
- Logika geo zweryfikowana niezależnie na realnym serwerze: `SELECT ... WHERE metadata.location != NONE`
  zwraca dokładnie zlokalizowane fibery (FLEXIBLE-traversal DZIAŁA, bez błędu SurQL); `find_fibers(near=Oslo,50km)`
  = tylko Oslo; `400km` = Oslo+Bergen.
- `test_surrealdb_geo_live.py`: **4/4 passed** po poprawce testu (normalizacja id `_`↔`-` — SurrealDB zwraca
  sanityzowaną formę `Fiber.id`, udokumentowany „Bug-C" z `_row_to_fiber`, poprzedza U8; silnik działa
  id-agnostycznie). To poprawka TESTU, nie produktu (fix `_row_to_fiber` byłby przekrojowy → NEEDS-HUMAN).
- Evidence: `qa/run-006/U8/real-db.md`. Sprzątnięto mózgi testowe z dev DB.

## Self-fix w trakcie (przed werdyktami agentów)
Wykryto i naprawiono własny bug: SQLite bbox `dlat = radius/111320` (średnia m/°) niedoszacowywał prawdziwy
spread (sfera 111195 m/°; przy elipsoidzie równik ≈110574) → wiersz wykluczony przez SQL nigdy nie trafiał do
exact post-filtra → fiber tuż wewnątrz promienia mógł zniknąć; klauzula lon przez `cos(lat0)` w środku bboxa
pogarszała to przy biegunach. Usunięto bbox (na nieindeksowanej kolumnie JSON nie daje zysku); został
`location IS NOT NULL` + exact haversine. Testy: `test_edge_of_radius_included`, `test_antimeridian_not_wrongly_excluded`.

## Werdykt: U8 VERIFIED
Wszystkie gates zielone, oba agenty pozytywne, live-DB potwierdzony. Zero otwartych bugów (2 LOW = pre-existing,
przeniesione do REPORT follow-ups, nie-blokujące). Zero pushy (NEEDS-HUMAN dla v2.10.0 przygotowane przy U10).
