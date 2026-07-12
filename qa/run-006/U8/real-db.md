# U8 (geospatial recall) — weryfikacja na żywej SurrealDB v3

**Worktree:** `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
**Live DB:** kontener `surreal-memory-surrealdb-1`, `SURREALDB_URL=http://localhost:8001` (potwierdzone: `curl http://localhost:8001/health` → `200`)
**Dane logowania:** wyłącznie z `.env` w worktree (skopiowany z głównego repo — worktree run-006 nie miał własnego `.env`; NS/DB/URL identyczne z main repo, hasło nieujawniane w tym raporcie)
**Python:** `.venv/bin/python -m pytest` (bez `uv run`)

## Komenda testowa

```bash
set -a; source .env; set +a
.venv/bin/python -m pytest tests/unit/test_surrealdb_geo_live.py -p no:cacheprovider -v
```

Test uruchomiony samodzielnie (plik `*_live.py`), zgodnie z zaleceniem.

## Wynik pytest

```
tests/unit/test_surrealdb_geo_live.py::TestSurrealGeoBrowse::test_location_roundtrips PASSED
tests/unit/test_surrealdb_geo_live.py::TestSurrealGeoBrowse::test_find_fibers_near_filters_on_real_db FAILED
tests/unit/test_surrealdb_geo_live.py::TestSurrealGeoBrowse::test_wider_radius_keeps_bergen FAILED
tests/unit/test_surrealdb_geo_live.py::TestSurrealGeoRecallPipeline::test_near_through_real_pipeline FAILED

3 failed, 1 passed in 1.64s
```

## Diagnoza porażek — dokładny ślad błędu

Wszystkie 3 porażki mają **identyczny kształt** — nie błąd SurrealQL, tylko `AssertionError` na porównaniu ID:

```
AssertionError: assert 'da317e0b-7c19-4a11-b69a-17103d359b2f' in {'da317e0b_7c19_4a11_b69a_17103d359b2f'}
AssertionError: assert ('4e6be48f-5105-459b-a04c-7194a305e8a0' in
    {'4e6be48f_5105_459b_a04c_7194a305e8a0', '59e358fa_2061_4486_b620_cc177b4ed8b6'})
AssertionError: assert 'cd840f71-e689-4383-87f3-cd25368f0dd9' in
    ['cd840f71_e689_4383_87f3_cd25368f0dd9']
```

Jedyne komunikaty z logów SurrealDB to kosmetyczne, znane wcześniej ostrzeżenia schematu
(pojawiają się przy KAŻDYM `storage.initialize()`, niezależnie od geo-testu):

```
WARNING  Schema statement failed: DEFINE FIELD metadata ON fiber TYPE object FLEXIBLE DEFAULT {}
         (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
```

**Nie ma żadnego błędu SurQL związanego z filtrem `near`/haversine ani z traversalem
`metadata.location != NONE` na polu FLEXIBLE.**

### Root cause: pre-istniejący "Bug C" (round-trip ID fibrów), NIE regresja U8

`_row_to_fiber()` (`src/surreal_memory/storage/surrealdb/store.py:302-330`, niezmieniona od
`1f6fe80` z 2026-05-27 — czyli **przed** całą pracą U8) odtwarza `Fiber.id` z rekordu SurrealDB,
ale **nie konwertuje `_` z powrotem na `-`** — w przeciwieństwie do analogicznej `_row_to_neuron()`
(linia 217: `neuron_id = neuron_id.replace("_", "-")`), która tę konwersję ma.

Skutek: `fiber.id` utworzony lokalnie w Pythonie ma myślniki (`da317e0b-7c19-...`), ale każdy
fiber odczytany przez `find_fibers`/`get_fiber`/pipeline z powrotem z SurrealDB ma podkreślniki
(`da317e0b_7c19_...`) — bo `_to_surreal_id()` (single-source sanitizer z PR #58) mapuje `-`→`_`
przy zapisie do record-id, a `_row_to_fiber` tego nie odwraca.

To jest **znany, udokumentowany** problem — plik `tests/unit/test_surrealdb_supersession_live.py`
(linie 126-130) opisuje go wprost jako *"Bug C root"* i pokazuje, że reszta silnika (np.
`get_typed_memory`) jest już celowo "id-agnostic" (akceptuje obie formy), żeby to obejść. Nowy
test geo `test_surrealdb_geo_live.py` po prostu **nie zastosował tej samej normalizacji** przy
porównywaniu ID w `assert oslo.id in ids`.

### Niezależny dowód, że logika geo-filtrowania jest poprawna

Aby odseparować bug ID-formatu od faktycznej logiki U8, wykonałam bezpośrednie zapytanie do
żywej bazy (ten sam brain, te same 3 fibry: Oslo/Bergen/bez lokalizacji):

```sql
SELECT id, metadata.location FROM fiber
WHERE brain_id = $bid AND metadata.location != NONE
```
→ **2 wiersze** (Oslo + Bergen), fiber bez lokalizacji poprawnie odrzucony, **brak błędu**.

```python
find_fibers(near=GeoFilter(Oslo, 50_000))   # promień 50 km
→ 1 fiber = Oslo   (Bergen ~305 km odrzucony, zgodnie z oczekiwaniem)

find_fibers(near=GeoFilter(Oslo, 400_000))  # promień 400 km
→ 2 fibery = Oslo + Bergen   (oba w zasięgu, zgodnie z oczekiwaniem)
```

Po znormalizowaniu ID (`.replace("-", "_")` po obu stronach) zwrócone zbiory ID **dokładnie**
odpowiadają oczekiwanym fiberom w każdym przypadku — czyli:
1. FLEXIBLE-owy traversal `metadata.location != NONE` **działa bez błędu** na żywym serwerze.
2. Dokładny post-filter haversine **poprawnie odczytuje** `location` i filtruje po promieniu.
3. Pipeline `ReflexPipeline.query(near=...)` (test 4) zwraca dokładnie 1 dopasowany fiber (Oslo),
   Bergen poprawnie wykluczony — end-to-end też działa.

Oba krytyczne ryzyka z planu (a) i (b) są więc **potwierdzone jako działające poprawnie**.

## Smoke `smem` CLI (best-effort)

`smem remember --help` i `smem recall --help` nie mają żadnych opcji `--location`/`--near`/geo —
CLI nie ma wpiętych argumentów geo. Zgodnie z instrukcją: **pominięto** ten krok (nie blokuje
weryfikacji U8, bo funkcja geo jest wywoływana programowo przez `find_fibers`/`ReflexPipeline`,
nie przez CLI).

## Sprzątanie

Wszystkie brainy testowe utworzone podczas weryfikacji (`u8-geo-live` ×8 z uruchomień pytest,
`u8-geo-manual-verify` ×1 z ręcznej weryfikacji) zostały usunięte (`store.clear()` +
`DELETE brain:...`). Weryfikacja: `SELECT count() FROM brain WHERE name IN [...] GROUP ALL` →
`0` pozostałych wierszy. Żadne dane produkcyjne nie zostały naruszone.

## WERDYKT

**U8 live-DB: PASS** (logika geo-filtrowania potwierdzona poprawna na żywym serwerze;
3 porażki pytest to pre-istniejący, znany, udokumentowany defekt formatu ID w
`_row_to_fiber` — "Bug C" — sprzed prac U8, niezwiązany z logiką geo, tylko z tym, że
nowy test porównuje ID bez normalizacji, której używają inne testy live w tym repo)

### Rekomendacja

Nie robić ślepego fixu produktu. Dwie opcje do decyzji człowieka:

1. **Niskie ryzyko (zalecane):** popraw `tests/unit/test_surrealdb_geo_live.py`, żeby
   normalizował ID przed porównaniem (`f.id.replace("-", "_")` po obu stronach), analogicznie
   do wzorca już użytego w `test_surrealdb_supersession_live.py` (linie 88-93, 121-135).
2. **Wyższe ryzyko (wymaga przeglądu, nie robić bez decyzji):** dodać
   `fiber_id = fiber_id.replace("_", "-")` do `_row_to_fiber`, żeby zrównać z `_row_to_neuron`
   — ale to zmienia zachowanie ID fibrów globalnie (poza U8), może wpłynąć na kod, który już
   świadomie zakłada formę z podkreślnikami (np. `resolve_fibers_for_neurons`,
   `get_typed_memory` id-agnostic lookup) — to zmiana cross-cutting, nie punktowa.

Nie wymaga eskalacji do Opus — to nie jest kwestia architektury/multi-tenancy, tylko konkretny,
już zdiagnozowany i udokumentowany w repo defekt formatu ID z jasną, punktową poprawką testu.
