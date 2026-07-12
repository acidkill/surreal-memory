# U10 — Release-prep v2.10.0 "Ecosystem": bateria weryfikacyjna

Branch: `feature/v2100-release-prep` (HEAD `70cf1a1`, z U9 HEAD `6dba78f`). Release-prep = mechaniczny
(version parity + CHANGELOG), zero zmian logiki.

## 1. Version parity 2.9.0 → 2.10.0 (9 plików)
`pyproject.toml:7` · `src/surreal_memory/__init__.py:19` · `tests/unit/test_health_fixes.py:488` (assert) ·
`integrations/surrealmemory/{package.json:3, package-lock.json:3,9}` ·
`integrations/surreal-memory-client/{package.json:3, package-lock.json:3,9}` ·
`vscode-extension/{package.json:5, package-lock.json:3,9}`.
- Bump chirurgiczny (line-targeted sed dla lock-files) — **`tinybench@2.9.0` NIE ruszony** (surrealmemory
  package-lock:2066/2067/2267, surreal-memory-client:2011/2012). To była pułapka: U7 miał szczęście (tinybench≠2.8.0),
  teraz kolidował → blanket-replace zbumpowałby dependency. `yauzl@2.10.0` (vscode lock:4631) = pre-existing dep, nietknięty.
- Weryfikacja: `surreal_memory.__version__ == 2.10.0`; `test_health_fixes.py` 17/17.

## 2. CHANGELOG
Sekcja `## [2.10.0] — Ecosystem` między `[Unreleased]` a `[2.9.0]`: Added (geospatial recall — near+location,
no migration; LangChain adapter — optional extra) + Fixed (find_fibers tag-pushdown = LIMIT po filtrze).

## 3. Battery (PLAN "Weryfikacja end-to-end")
- **item 2 — pełny suite (D1 no-DB)**: **6307 passed / 0 FAILED / cov 68.30%**, 26 błędów = PRE-EXISTING test_api.py.
  mypy 359 czysto, ruff+format czysto.
- **item 1 — golden ranking**: `test_golden_ranking.py` 6/6 (ranking bit-for-bit, defaulty niezmienione).
- **item 6 — geo**: haversine Oslo→Bergen + near-filtr — `test_geo_*` (non-live) + LIVE `test_surrealdb_geo_live.py`
  **5/5** na SurrealDB v3 (near + tag-pushdown).
- **item 7 — LangChain**: `examples/langchain_rag.py` uruchamia się end-to-end (LCEL + RunnableWithMessageHistory,
  stub-LLM). Base env bez langchain → `importorskip("langchain_core")` skipuje plik testowy (suite zielony) —
  z konstrukcji; w tym venv langchain-core 1.4.9 zainstalowany, więc adapter-testy się WYKONUJĄ (12+ passed).
- **item 4 — Spectron-demo regresja** (Emma Oslo→Bergen, valid_at, include_superseded, trace, uncertainty): kod
  U3–U7 niezmieniony od U7 (7/7 live); pokryty w D1 przez test_recall_supersession / test_valid_at_pipeline /
  test_uncertainty. Jedyna zmiana dotykająca tej ścieżki (U8 surface-guard dla valid_at) zweryfikowana bez regresji.
- **item 5 — perf**: brak zmian w hot-path recall wpływających na perf (near = filtr na ≤60 kandydatach; tag-pushdown
  = predykat SQL). Bez regresji perf (benchmarki poza D1; zweryfikowane wcześniej U6/U7 — /stats <3s, trace <2%).
- **item 3 — migracja**: N/A — brak DDL od schematu v9 (U8 metadata-only, U9 code-only, U10 version-only).

## 4. Werdykt: U10 VERIFIED → v2.10.0 gotowe do wydania (NEEDS-HUMAN)
Wszystkie gates zielone. Version parity spójny (2.10.0), tinybench nietknięty. NEEDS-HUMAN push/PR/tag v2.10.0 +
registry-verification: `qa/run-006/U10/needs-human.md`. Zero pushy.
