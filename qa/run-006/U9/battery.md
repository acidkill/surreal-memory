# U9 — LangChain adapter (PR8): bateria weryfikacyjna

Branch: `feature/v2100-pr8-langchain` (z HEAD U8 `b893604`). Commity U9:
`10b9562` feat: retriever + chat history + extra + example ·
`3c1cd1d` fix: 4 HIGH + ordering + tag-pushdown (Stage-2 runda 1) ·
`<fork-fix>` fix: fork-safe bridge loop (Stage-2 re-review).

## Zakres (kod-only — nowy opcjonalny extra, BEZ zmiany schematu)
Nowy pakiet `src/surreal_memory/adapters/` (OUTBOUND — odróżniony od `integration/adapters/` = inbound ingestion).
- `adapters/__init__.py`: lazy `__getattr__` → `import surreal_memory.adapters` NIE importuje langchain (import-safe).
- `adapters/langchain.py` (import-guard try/except → `pip install surreal-memory[langchain]`):
  - `SurrealMemoryRetriever(BaseRetriever)`: pola pydantic brain_name/depth/max_tokens/**memory_tags**/permanent_only/
    k=5; `from_storage(...)` DI + leniwe `get_shared_storage`. `_afetch` → ReflexPipeline.query → mapowanie fibrów na
    `Document` (page_content = anchor content → summary → essence; metadata fiber_id/memory_type/tags/salience/
    confidence/created_at/source). k-cap. Fallback: 1 Document z answer gdy brak fibrów i k>0. Async natywny +
    sync bridge.
  - `SurrealMemoryChatMessageHistory(BaseChatMessageHistory)`: zapis przez MemoryEncoder z tagami {langchain,
    lc-session:<id-znormalizowany>}, metadata lc_role/lc_session/lc_content(verbatim)/lc_seq; odczyt find_fibers(
    tags=) posortowane po (time_start, lc_seq) + exact lc_session guard; clear() usuwa fibry sesji.
  - `_run_sync`: JEDNA współdzielona pętla-tło (run_coroutine_threadsafe) — bezpieczna pod batch() i os.fork().
- pyproject: `langchain = ["langchain-core>=0.3,<2"]` (zainstalowane 1.4.9) + do `all`.
- README sekcja LangChain + `examples/langchain_rag.py` (LCEL + RunnableWithMessageHistory, działa ze stub-LLM).

## Stage 1 — gates (env -u SURREALDB_URL, .venv/bin; langchain-core 1.4.9 w venv → adapter-testy się WYKONUJĄ)
- **D1 no-DB**: **6307 passed / 0 FAILED**, 26 błędów = PRE-EXISTING test_api.py (nie U9). Coverage **68.31%** ≥ 67
  (adapters/__init__.py 100%, adapters/langchain.py ~88%). mypy 359 czysto. ruff+format czysto.
- Testy: `test_langchain_adapter.py` (importorskip; retriever real-pipeline mapping, k-cap, fallback, oba tory
  sync-bridge, batch-no-hang, **fork-survives**, mixed-case session izolacja, memory_tags≠tags, ordering tie) +
  `test_find_fibers_tag_pushdown.py` (LIMIT-po-tagu, InMemory+SQLite) + live `test_surrealdb_geo_live.py::
  TestSurrealTagPushdown` (SurQL `$tag IN auto_tags OR $tag IN agent_tags`, 5/5 na SurrealDB v3).

## Stage 2 — python-reviewer (dwie rundy)
### Runda 1 → **BLOCK** (4 HIGH + MEDIUM + 2 LOW) — wszystkie naprawione (commit 3c1cd1d):
1. HIGH session_id case: enkoder lowercase'uje tagi, read był case-preserved → mixed-case sesje puste + wyciek do
   lowercase-bliźniaka. Fix: `_normalize_tag` (lower().strip() = default TagNormalizer) na odczycie + exact
   `lc_session` w metadata (izolacja niezależna od kolizji tagów).
2. HIGH `tags` field cieniował `BaseRetriever.tags` (LangSmith tracing) → wyciek recall-filtra do trace'u. Fix:
   zmiana na `memory_tags`.
3. HIGH `_run_sync` tworzył NOWĄ pętlę per-call → deadlock pod `retriever.batch()` (wątki rywalizujące o
   process-global asyncio.Lock storage'u przypięte do różnych pętli). Fix: jedna współdzielona pętla-tło.
4. HIGH `find_fibers` obcinał LIMIT PRZED filtrem tagów → historia sesji znika na mózgu > okno fetch. Fix:
   pushdown tagu do SQL (SQLite `EXISTS json_each`) i SurQL (`$tag IN auto_tags OR $tag IN agent_tags` — SurrealDB
   NIE ma pola `tags`, tylko auto_tags/agent_tags; naiwne `$tag IN tags` zwracało 0 — złapane LIVE). Zweryfikowane
   na żywym SurrealDB v3.
   - MEDIUM kolejność: `lc_seq = time.time_ns()` tiebreaker. LOW: k=0 bez fallbacku.
### Runda 2 (re-review) → **BLOCK CLEARED** — potwierdził wszystkie 6 fixów; wykrył JEDEN nowy MEDIUM (naprawiony):
- MEDIUM fork-unsafe singleton bridge loop: pętla-tło utworzona PRZED `os.fork()` (gunicorn --preload + warmup) →
  dziecko dziedziczy martwą referencję → sync-call wisi. Reprodukcja empiryczna (exit 124). Fix:
  `os.register_at_fork(after_in_child=_reset_bridge_after_fork)` (reset `_bridge_loop=None`) + test
  `test_bridge_loop_survives_fork` (realny fork, dziecko przeżywa; działa też pod xdist). ruff/mypy/D1 zielone po fixie.
- Deferred (NOT blocking → REPORT follow-ups): (a) chat history limit=1000/sesję (kolizja sesji + >1000 tur = ten
  sam limit-cap co dla pojedynczej długiej sesji, nie regresja). (b) langchain-core floor 0.3.x nietestowany
  (testowany ceiling 1.4.9).

## NEEDS-HUMAN
- Draft issue stale-TS-client (`/api/remember|recall` vs realne `/api/v1/memory/{encode,query}`):
  `qa/run-006/U9/stale-ts-client-issue.md` — do założenia jako GitHub issue.

## Werdykt: U9 VERIFIED
Wszystkie gates zielone; reviewer BLOCK CLEARED po naprawieniu 4 HIGH + MEDIUM + LOW + nowego fork-MEDIUM; live
SurrealDB tag-pushdown potwierdzony. Zero otwartych bugów blokujących (2 LOW deferred → REPORT). Zero pushy.
