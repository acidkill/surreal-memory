

# Surreal-Memory

[![PyPI](https://img.shields.io/pypi/v/surreal-memory.svg)](https://pypi.org/project/surreal-memory/)
[![CI](https://github.com/acidkill/surreal-memory/workflows/CI/badge.svg)](https://github.com/acidkill/surreal-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![SurrealDB](https://img.shields.io/badge/Powered_by-SurrealDB-ff00e5)](https://surrealdb.com/)

**Memoria gráfica persistente para agentes de IA, impulsada por SurrealDB.**

Todas las funciones son gratuitas y de código abierto. Sin claves de licencia. Sin paywalls. No se requiere una API de embeddings para el uso básico.

```bash
pip install surreal-memory[surrealdb]
smem init --full
```

Reinicia tu herramienta de IA. Tu agente ahora recuerda.

---

## ¿Por qué Surreal-Memory?

La mayoría de las herramientas de memoria para IA son bases de datos vectoriales con una API de búsqueda añadida. Surreal-Memory es un **grafo que piensa**: los recuerdos se almacenan como neuronas interconectadas y se recuperan mediante activación propagada, respaldado por el motor multimodelo de SurrealDB (documento + grafo + vector en una sola base de datos).

```
Query: "Why did Tuesday's outage happen?"

Surreal-Memory traces the chain:
outage ← CAUSED_BY ← JWT expiry ← SUGGESTED_BY ← Alice's review
```

**Las relaciones son explícitas** — `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS` — para que tu agente no solo encuentre recuerdos, sino que *razone* a través de ellos.

| | RAG / Búsqueda Vectorial | Surreal-Memory |
|--|---------------------|----------------|
| Backend | Pinecone / Chroma | **SurrealDB** (documento + grafo + vector) |
| Recuperación | Puntuación de similitud | Recorrido de grafo + búsqueda vectorial |
| Relaciones | Ninguna | 41 tipos de sinapsis explícitos |
| LLM requerido | Sí (embeddings) | No — funciona completamente offline |
| Razonamiento multi-salto | Múltiples consultas | Un solo recorrido |
| Ciclo de vida de la memoria | Estático | Decaimiento, reforzamiento, consolidación |
| Costo por 1K consultas | ~$0.02 | **$0.00** |

---

## ¿En qué se diferencia de NeuralMemory?

Surreal-Memory se construye sobre la arquitectura de memoria gráfica de [NeuralMemory](https://github.com/nhadaututtheky/neural-memory), pero reemplaza el modelo SQLite + pago-Pro por **SurrealDB + plugin comunitario gratuito**:

| | NeuralMemory (origen) | Surreal-Memory |
|--|---------------|----------------|
| Motor de almacenamiento | SQLite (limitado) | **SurrealDB** (todas las funciones gratis) |
| Búsqueda vectorial | Función de pago Pro | **Integrada** vía HNSW de SurrealDB |
| Recuperación semántica | Función de pago Pro | **Gratuita** vía plugin comunitario |
| Consolidación inteligente | Función de pago Pro | **Gratuita** vía plugin comunitario |
| Compresión | Función de pago Pro | **Gratuita** vía plugin comunitario |
| Licencia requerida | Sí para funciones Pro | **No** — todo es gratuito |
| Multimodelo | No | **Sí** — documento + grafo + vector |

---

## Inicio Rápido

### Configuración automatizada (Claude Code)

Dale esto a Claude Code en cualquier máquina y se encargará de todo: prerequisitos, Docker, registro de MCP y verificación:

```
Please read INSTALL_PROMPT.md and follow the instructions to set up Surreal-Memory on this machine.
```

O clona y apunta Claude Code al archivo:

```bash
git clone https://github.com/acidkill/surreal-memory.git
# then in Claude Code:
# "Read INSTALL_PROMPT.md and follow the setup instructions"
```

### Docker (manual)

```bash
cp .env.example .env    # edit with your keys
docker compose -f docker-compose.surrealdb.yml up -d
```

Panel de control en http://localhost:8000/ui, SurrealDB en localhost:8001.

> **Requiere SurrealDB ≥ 3.2.0** (el archivo compose usa `surrealdb/surrealdb:v3.2.0`).
> ¿Actualizando desde una versión anterior de SurrealDB? Haz una copia de seguridad del volumen `surrealdb_data` primero; el grafo de sinapsis se migra automáticamente a bordes RELATE nativos en la primera conexión tras la actualización.

### Manual

```bash
pip install surreal-memory[surrealdb]
smem init --full
```

### Primer recuerdo

```bash
smem remember "Fixed auth bug with null check in login.py:42"
smem recall "auth bug"
# → "Fixed auth bug with null check in login.py:42"
```

---

## 3 herramientas. Eso es todo.

Hay 58 herramientas MCP disponibles, pero solo necesitas tres:

| Herramienta | Qué hace |
|------|-------------|
| `smem_remember` | Almacena un recuerdo — detecta automáticamente tipo, etiquetas y conexiones |
| `smem_recall` | Recupera mediante activación propagada + búsqueda vectorial |
| `smem_health` | Puntuación de salud cerebral (A–F) con sugerencias de corrección accionables |

Todo lo demás — sesiones, carga de contexto, seguimiento de hábitos, mantenimiento — funciona transparentemente en segundo plano.

---

## Arquitectura

```
                    ┌──────────────────────────────┐
                    │       MCP Server (57 tools)   │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     Engine (encoding +        │
                    │     retrieval pipeline)       │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │        SurrealDB Backend         │
              │  ┌─────────┬─────────┬────────┐ │
              │  │ Document │  Graph  │ Vector │ │
              │  │  Store   │ Queries │  HNSW  │ │
              │  └─────────┴─────────┴────────┘ │
              └─────────────────────────────────┘
```

### Modelo de datos central

- **Brain (Cerebro)** — contenedor de nivel superior con configuración
- **Neuron (Neurona)** — nodo de conocimiento atómico (entidad, concepto, tiempo, acción, intención, estado)
- **Synapse (Sinapsis)** — borde tipado y dirigido entre neuronas (41 tipos: `CAUSED_BY`, `LEADS_TO`, etc.)
- **Fiber (Fibra)** — un registro de memoria: contenido tipado con metadatos, prioridad, etiquetas y etapa del ciclo de vida

### Motor

- **Pipeline de codificación** — pasos asíncronos componibles: extraer entidades → crear neuronas → vincular sinapsis → empaquetar en fibras
- **Recuperación reflexiva** — activación propagada a través del grafo de neuronas, combinada con búsqueda vectorial de SurrealDB cuando está disponible
- **Reclasificación** — paso opcional de codificador cruzado que recupera en exceso los candidatos de SA y mezcla la puntuación de relevancia con el nivel de activación para una mayor precisión en la recuperación
- **Consolidación** — fusiona neuronas similares, refuerza rutas fuertes, poda las débiles
- **Compresión** — ciclo de vida en 5 niveles: completo → resumen → esencia → fantasma → metadatos

### Plugin comunitario

El `CommunityPlugin` integrado habilita las capacidades de nivel Pro sin costo:

- **Compresión direccional** — preservación semántica multieje, usada por el paso de compresión
- **Auto-tier durante la consolidación** — requiere un plugin registrado, incluido de forma predeterminada

---

## Sincronización en la nube

Sincroniza tu cerebro en cada máquina a través de tu propio Cloudflare Worker:

```
Laptop ←→ Your Cloudflare Worker ←→ Desktop
                  ↕
              Your Phone
```

Despliegas el centro de sincronización en **tu propia cuenta de Cloudflare**. Tu base de datos D1, tu clave de cifrado, tus datos.

```bash
smem sync sync --direction both   # sincronización bidireccional (también: push, pull)
```

Define `SURREAL_MEMORY_SYNC_AUTO=true` (o `auto_sync` en `[sync]` de
`config.toml`) para sincronizar automáticamente tras cada remember/recall.

La sincronización usa **delta de Merkle** — solo viajan las diferencias, no todo el cerebro.

---

## Características

#### Memoria y recuperación
- **15 tipos de memoria** — hecho, decisión, error, insight, preferencia, flujo de trabajo, instrucción y más
- **Activación propagada** — los recuerdos aparecen por asociación, no por coincidencia de palabras clave
- **Búsqueda vectorial** — HNSW de SurrealDB para similitud semántica (cuando los embeddings están configurados)
- **Reclasificación con codificador cruzado** — paso de precisión opcional basado en configuración, HTTP (servidor de inferencia compartido) o en proceso, mezclado con la puntuación de activación
- **Razonamiento cognitivo** — hipotetizar, presentar evidencia, hacer predicciones, verificar con confianza bayesiana
- **Suplantación por hecho** — cuando un nuevo recuerdo reemplaza a uno antiguo, la recuperación resuelve automáticamente el conflicto (el hecho antiguo queda obsoleto, no se elimina); recupera el estado del mundo en un momento pasado con `valid_at`
- **Puntuación de confianza y antigüedad** — pondera la recuperación según cuánto confíes en una fuente y qué tan fresco es un recuerdo, en lugar de tratar todo como igualmente confiable para siempre
- **Visualización de incertidumbre** — pregunta a `smem_uncertainty` cuánto confiar en una respuesta: contradicciones, deriva, recuerdos por expirar pronto, hechos con poca evidencia
- **Trazas de recuperación consultables** — registro opcional de *por qué* una recuperación devolvió lo que devolvió, para depuración y auditoría
- **Recuperación geoespacial** — adjunta una ubicación a un recuerdo y filtra la recuperación a un radio alrededor de un punto (`near`)

#### Ingesta de conocimiento
- **Entrenamiento desde documentos** — PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV ingeridos en conocimiento cerebral permanente
- **Entrenamiento desde esquemas de base de datos** — extrae estructuras de tablas y relaciones FK
- **Entrenamiento masivo rápido** — `smem train` agrupa escrituras en DB (un viaje de ida y vuelta por N sinapsis vía `add_synapses_batch`) y ahora se usa el índice `brain_id` de `find_neurons`, por lo que los documentos grandes permanecen económicos por fragmento incluso en cerebros grandes (previamente 7–15 s/fragmento en un cerebro de 68k neuronas; ~10× menos ops de DB/fragmento). Muestra progreso en vivo con `tqdm`.
- **Adaptadores de importación** — migración desde ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex
- **Entrenamiento de razonamiento** (opcional) — extrae el propio `thinking` de un modelo desde transcripciones de `~/.claude`, lo destila en fibras de patrones de razonamiento reutilizables e inyecta las estrategias aprendidas en las sesiones de otros modelos (panel + CLI `smem reasoning` + herramienta MCP `smem_reasoning`). Desactivado por defecto; las trazas se redactan antes de almacenarse. La **cobertura por categoría** por modelo — cuántas de las 8 categorías tienen suficientes patrones — es el número a vigilar; `pattern_targets` limita cuántos patrones puede acumular cada modelo, aunque el backlog de trazas propias del modelo suele agotarse primero.
- **Reconstrucción de patrones perdidos** — la destilación marca cada traza que consume como procesada, incluidas las que descarta, por lo que los patrones no pueden derivarse normalmente de un backlog ya minado. `reprocess` lo reabre: una casilla junto a *Backfill* en el panel, `smem reasoning mine --reprocess`, o `reprocess: true` en `smem_reasoning`. Repetirlo es seguro: las firmas de patrón hacen que una segunda pasada sea una operación nula en lugar de un duplicador. Respeta `mining_models` (los globs se resuelven contra los modelos realmente presentes, y un filtro que no coincida con nada no reabre nada), y `dry_run` aún prevalece, ya que reabrir el backlog es una escritura. Las trazas ya eliminadas por `retention_days` necesitan `--backfill` para reingerirlas desde las transcripciones primero.
- **Nombramiento de patrones con LLM** (opcional, solo local) — por defecto, un patrón destilado se nombra según su propia mecánica (`debugging: restate-goal, gather-evidence, verify`) y se describe con un fragmento crudo de la traza mediana. Apunta `distill_use_llm` a un punto final compatible con OpenAI en **loopback** y un modelo local reescribe el título, la descripción y la estrategia para que sean legibles, dejando intacta la identidad y las estadísticas. Se rechaza por completo un punto final no loopback — las trazas de razonamiento nunca salen de la máquina — y cualquier falla vuelve al nombramiento mecánico. `distill_llm_load_cmd` carga el modelo una vez antes de la primera solicitud, para que controlas *cómo* se carga (tamaño de contexto, capas GPU, sin proyector de visión) en lugar de aceptar lo que el punto final haga implícitamente; `distill_llm_unload_cmd` lo libera cuando termina la ejecución, para que no ocupe VRAM entre ejecuciones. Ambos son listas argv que se ejecutan sin shell, `{model}` se sustituye, y la ausencia o falla de cualquiera degrada silenciosamente al comportamiento antiguo:

  ```toml
  [reasoning_training]
  distill_use_llm = true
  distill_llm_model = "<a chat model your server serves>"
  distill_llm_endpoint = "http://127.0.0.1:PORT/v1"   # loopback only
  distill_llm_load_cmd = ["<your-launcher>", "load", "{model}", "--n-gpu-layers", "99"]
  distill_llm_unload_cmd = ["<your-launcher>", "stop", "{model}"]
  ```

#### Ciclo de vida y almacenamiento
- **Consolidación de memoria** — los recuerdos episódicos maduran en conocimiento semántico mediante **recuperación espaciada**, no mediante un comando. Una fibra alcanza `semantic` después de 7 días en `episodic` *más* reforzamiento distribuido en 3+ días distintos (o 15+ ensayos en 5+ ventanas de tiempo) — recuperar un recuerdo es lo que lo avanza; `smem consolidate` no puede moverlo por sí solo. `smem health` muestra dónde se encuentra cada recuerdo (`stage_distribution`) y en qué sigue esperando (`semantic_gate_blockers`: tiempo de permanencia, espaciado de recuperación o ya elegible), por lo que una `consolidation_ratio` plana es diagnosticable en lugar de misteriosa. La recuperación ensaya los recuerdos que realmente mostró — aumenta `brain.reinforcement_neuron_limit` (15 por defecto) para ampliar ese alcance a costa de cierta latencia de recuperación.
- **Niveles de compresión** — completo → resumen → esencia → fantasma → metadatos
- **Control de versiones del cerebro** — instantánea, reversión, diff, trasplante de recuerdos entre cerebros

#### Ecosistema
- **Panel web** — UI React multipágina en `/ui` (vista general, radar de salud, grafo, línea temporal, evolución, almacenamiento, estadísticas de herramientas, visualización, configuración, incertidumbre) — cada página gratuita, sin puerta Pro
- **Extensión para VS Code** — árbol de memoria, explorador de grafo, CodeLens, sincronización por WebSocket
- **Adaptador para LangChain** — extra opcional (`pip install surreal-memory[langchain]`) que expone un `BaseRetriever` e historial de mensajes de chat respaldado por un cerebro — ver [API de Python](#python-api)
- **Seguridad** — cifrado Fernet, detección automática de contenido sensible, firewall de entrada
- **Sistema de plugins** — extensible con herramientas MCP propias y una función de compresión

---

## Embeddings

El núcleo de palabras clave + grafo funciona **sin ninguna API de embeddings**. Los embeddings son
opcionales: agregan recuperación semántica (búsqueda vectorial vía HNSW de SurrealDB) para que los
recuerdos aparezcan incluso cuando la redacción es diferente. Configura de forma interactiva con `smem setup embeddings`
o mediante variables de entorno `SURREAL_MEMORY_EMBEDDING_*`.

**Recomendado — Google Gemini** (`gemini-embedding-001`, 3072-dim, multilingüe, nivel gratuito):

```bash
pip install "surreal-memory[surrealdb,embeddings-gemini]"
export GEMINI_API_KEY=...        # free key: https://aistudio.google.com/apikey
```

**¿Sin clave API? Ejecútalo localmente** — la misma clase de modelo en dispositivo que usa ChromaDB/MemPalace,
vía `sentence-transformers` (offline, sin clave):

```bash
pip install "surreal-memory[surrealdb,embeddings]"
smem setup embeddings            # choose "Sentence Transformers"
```

| Proveedor | Modelo por defecto | Clave | Notas |
|----------|---------------|-----|-------|
| **Gemini** (recomendado) | `gemini-embedding-001` | `GEMINI_API_KEY` | 3072-dim, multilingüe, nivel gratuito |
| Local (sentence-transformers) | `all-MiniLM-L6-v2` · `paraphrase-multilingual-MiniLM-L12-v2` | — | offline, sin clave, ~440MB de descarga |
| Ollama | `nomic-embed-text` · `bge-m3` | — | servidor local (`ollama serve`) |
| OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` | pago |
| OpenRouter | `openai/text-embedding-3-small` | `OPENROUTER_API_KEY` | Compatible con OpenAI |

Establece el proveedor en `auto` para elegir la mejor opción disponible en tiempo de ejecución
(orden: Ollama → sentence-transformers local → Gemini → OpenAI → OpenRouter).

**La combinación se valida.** Cada proveedor asume una dimensión para un modelo que no
reconoce, por lo que apuntar a un nombre de modelo de otro proveedor genera vectores del ancho
incorrecto — que el índice vectorial rechaza en cada escritura. `smem doctor`, `smem_health`
y el inicio de MCP reportan dos casos imposibles en lugar de pasarlos: un modelo fuera del
catálogo de un proveedor alojado (el informe lista los modelos que sí sirve) y un modelo conocido
cuya dimensión contradice la configurada. La verificación se mantiene silenciosa donde no puede
saberlo — un servidor compatible con OpenAI local sirve los archivos que se le indiquen, por lo que un
nombre de modelo desconocido allí es normal, y solo una coincidencia exacta en el catálogo cuenta como conocer
una dimensión.

> Los embeddings usan **un solo** modelo por cerebro — cambiar de modelo altera las dimensiones vectoriales
> e invalida los vectores existentes. Elige un proveedor antes de ingerir a escala.

---

## Reclasificación

La activación propagada recupera en exceso candidatos; un reclasificador con codificador cruzado opcional luego
puntuada cada par `(query, memory)` por relevancia y mezcla esa puntuación con el
nivel de activación (`blend_weight`, por defecto `0.7`) para una pasada final de precisión. Desactivado por
defecto: la recuperación funciona igual sin él.

```toml
[reranker]
enabled = true
endpoint = "http://127.0.0.1:11435/v1"   # OpenAI-compatible /rerank (e.g. llamastash)
model_name = "BAAI/bge-reranker-v2-m3"
blend_weight = 0.7
```

- **Modo HTTP** (`endpoint` establecido) — se ejecuta en un servidor de inferencia compartido (ej. llama.cpp /
  llamastash en GPU), no requiere dependencia de `torch` localmente. Vuelve a la
  variable de entorno `SURREAL_MEMORY_RERANKER_ENDPOINT` cuando `endpoint` no está establecido. Para endpoints
  que requieren autenticación, establece `SURREAL_MEMORY_RERANKER_API_KEY` y las solicitudes incluyen un
  encabezado `Bearer` (una clave vacía no envía nada, por lo que llamastash no necesita configuración extra).
- **Modo en proceso** (sin endpoint) — carga un `CrossEncoder` local de `sentence-transformers`.
  Instala con `pip install "surreal-memory[reranker]"`.
- La reclasificación nunca rompe la recuperación, pero tampoco falla *silenciosamente*. Una reclasificación fallida
  se reintenta una vez; si aún no puede ejecutarse, la recuperación devuelve el orden de activación propagada
  **y lo indica** — `rerank_degraded` en la respuesta de MCP,
  `rerank_degraded_warning` en la CLI. Los resultados sin reclasificación se ven exactamente como
  los reclasificados, por lo que una recuperación silenciosa ocultaría una caída real en la precisión.

---

## Configuración por herramienta

<details>
<summary><b>Claude Code (Plugin)</b></summary>

```bash
/plugin marketplace add acidkill/surreal-memory
/plugin install surreal-memory@surreal-memory-marketplace
```

</details>

<details>
<summary><b>Cursor / Windsurf / Otros clientes MCP</b></summary>

```bash
pip install surreal-memory[surrealdb]
```

Agrega a la configuración MCP de tu editor:

```json
{
  "mcpServers": {
    "surreal-memory": { "command": "smem-mcp" }
  }
}
```

</details>

<details>
<summary><b>OpenClaw (Plugin)</b></summary>

```bash
pip install surreal-memory[surrealdb] && npm install -g surrealmemory
```

Establece el slot de memoria en `~/.openclaw/openclaw.json`:
```json
{ "plugins": { "slots": { "memory": "surrealmemory" } } }
```

</details>

<details>
<summary><b>TypeScript / JavaScript (REST SDK)</b></summary>

```bash
npm install @acidkill/surreal-memory-client
```

```ts
import { SurrealMemoryClient } from "@acidkill/surreal-memory-client"

const client = new SurrealMemoryClient({
  baseUrl: "http://localhost:8000",
  brain: "myproject",
})

await client.remember({ content: "Fixed auth bug", type: "fix", priority: 7 })
const { results } = await client.recall({ query: "auth bug" })
```

Referencia completa: [`integrations/surreal-memory-client/README.md`](integrations/surreal-memory-client/README.md).

</details>

<details>
<summary><b>Docker (autoalojado)</b></summary>

```bash
cp .env.example .env          # configure SurrealDB + embeddings
docker compose -f docker-compose.surrealdb.yml up -d
```

Panel de control: http://localhost:8000/ui

</details>

---

## API de Python

```python
import asyncio
from surreal_memory import Brain
from surreal_memory.storage import create_storage
from surreal_memory.core.brain_mode import BrainModeConfig, BrainMode
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline

async def main():
    config = BrainModeConfig(mode=BrainMode.LOCAL)
    storage = await create_storage(config, brain_id="my_brain")

    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Met Alice to discuss API design")
    await encoder.encode("Decided to use FastAPI for backend")

    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What did we decide about backend?")
    print(result.context)  # "Decided to use FastAPI for backend"

asyncio.run(main())
```

### Integración con LangChain

Instala el extra (`pip install surreal-memory[langchain]`) y envuelve un cerebro como un
retriever de LangChain + historial de mensajes de chat. Ambos son en proceso (no se necesita servidor REST);
todo es asíncrono por debajo con un puente síncrono para la API síncrona de LangChain.

```python
from surreal_memory.adapters.langchain import (
    SurrealMemoryChatMessageHistory,
    SurrealMemoryRetriever,
)

# Retriever — las fibras coincidentes se convierten en Documentos de LangChain (page_content = texto de memoria,
# metadata lleva fiber_id, tags, salience, confidence, source="surreal-memory").
retriever = SurrealMemoryRetriever(brain_name="my_brain", k=5)
docs = await retriever.ainvoke("what did we decide about the backend?")

# Historial de chat por sesión — los turnos se almacenan como recuerdos etiquetados lc-session:<id>.
history = SurrealMemoryChatMessageHistory("session-42", brain_name="my_brain")
history.add_user_message("Which database are we using?")
history.add_ai_message("SurrealDB.")
print(history.messages)  # reproducidos textualmente, en orden

# ¿Ya tienes un handle de storage (tests, cableado personalizado)? Inyéctalo:
retriever = SurrealMemoryRetriever.from_storage(storage, k=5)
```

Consulta [`examples/langchain_rag.py`](examples/langchain_rag.py) para una cadena RAG LCEL completa
con `RunnableWithMessageHistory`.

---

## Desarrollo

```bash
git clone https://github.com/acidkill/surreal-memory
cd surreal-memory && pip install -e ".[dev]"
smem doctor --dev        # Verificar configuración de colaborador
pytest tests/ -v          # 7200+ tests
ruff check src/ tests/    # Lint
make verify               # Full CI gate
```

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para directrices.

## Agradecimientos

Surreal-Memory se construye sobre [**NeuralMemory**](https://github.com/nhadaututtheky/neural-memory) por [nhadaututtheky](https://github.com/nhadaututtheky) — un excepcional sistema de memoria basado en grafos para agentes de IA. La arquitectura central (neuronas, sinapsis, fibras, activación propagada, consolidación, compresión y la interfaz MCP de 53 herramientas) es enteramente su trabajo.

Surreal-Memory lo extiende con un backend de almacenamiento SurrealDB y un plugin comunitario que hace que todas las funciones avanzadas estén disponibles gratis.

Igualmente importante: este proyecto no existiría sin [**SurrealDB**](https://surrealdb.com/). El modelo combinado de documento + grafo + vector en un solo motor es lo que hizo posible retirar la división SQLite + pago-Pro. El cambio depende específicamente de los cambios lanzados en **SurrealDB 3.x** — sin ellos, el backend de almacenamiento en este fork seguiría siendo un concepto teórico. A partir de v2.6.0, el grafo de sinapsis usa bordes RELATE nativos y ISO GQL interno, por lo que **se requiere SurrealDB ≥ 3.2.0**.

> Si encuentras útil Surreal-Memory, por favor da estrella tanto al [proyecto original NeuralMemory](https://github.com/nhadaututtheky/neural-memory) como a [SurrealDB](https://github.com/surrealdb/surrealdb).

## Hoja de ruta

Los elementos aquí están explícitamente **no** incluidos en la versión actual. Se aceptan PRs de la comunidad — consulta [CONTRIBUTING.md](CONTRIBUTING.md) y [AGENTS.md](AGENTS.md).

### Diferidos (se requiere acción externa)

- **Listado en ClawHub** — el plugin OpenClaw aún necesita que su entrada en el registro de ClawHub esté alineada con el slug `surreal-memory`. El flujo de publicación ya apunta a `--slug surreal-memory`; esperando por el lado del registro.
- **Editor de Marketplace de VS Code** — `vscode-extension/package.json` ahora usa el editor `ai-flow-nowak`. La cuenta de editor debe crearse / verificarse antes de que la próxima versión pueda publicar en Marketplace.
- **Nombre canónico en PyPI** — `surreal-memory` es el único paquete publicado. Si queda algún paquete con nombre anterior en PyPI bajo una cuenta que controlamos, marcalo como `Development Status :: 7 - Inactive` con un README que apunte a `surreal-memory`.

### Deseables (se aceptan contribuciones de la comunidad)

- **Puerta de regresión de calidad de recuperación** — `benchmarks/ground_truth.py` y `metrics.py` ya pueden puntuar la recuperación contra un conjunto etiquetado, pero nada los ejecuta. Conéctalos a CI con un piso en precisión/recuperación para que un cambio de recuperación no pueda silenciosamente empeorar el producto. Para un sistema de memoria, esta es la única prueba de mayor valor que aún no existe.
- **Prueba en un solo comando** — `docker compose up` que levante SurrealDB, la API, el panel y un cerebro de demostración precargado. Evaluar el proyecto actualmente significa aprovisionar una base de datos y elegir un proveedor de embeddings primero; un nuevo usuario debería poder ver un grafo poblado en un solo comando y juzgarlo a partir de ahí.
- **Inyección de patrones de razonamiento activada por defecto** — la minería aprende estrategias por modelo, pero `injection_enabled` predetermina `false`, por lo que los patrones se quedan sin usar a menos que encuentres la bandera. Actívala detrás de un presupuesto de tokens y un piso de calidad, y muestra en el panel qué patrón se activó en qué turno.
- **Enrutamiento de cerebro por proyecto** — resolver el cerebro activo desde la raíz del proyecto o el remoto de git en lugar de un `current_brain` global único, para que cambiar de repositorio cambie la memoria sin `smem brain use`. Elimina la fuente más común de "por qué está este recuerdo aquí".
- **Gancho de restauración de contexto PostCompact** — análogo a `session_start.py` pero disparado justo después de una compactación de Claude Code. Canaliza `smem context --limit 20` — más un equivalente CLI de la herramienta MCP `smem_recap`, que el gancho necesitaría primero — a stdout como un bloque `## Context restored after compaction`, para que el agente no pierda el hilo cuando se activa el búfer del 80%.
- **Plugin para IDE JetBrains** — plugin Kotlin / Java que use la misma API REST que el panel. Característica de paridad para IDEs de la familia IntelliJ.
- **Cloudflare Pages para documentación** — alternativa al flujo eliminado de GitHub Pages. `mkdocs build` estático desplegado en CF Pages, sin dependencia de GitHub.
- **Adaptador de retriever para LlamaIndex** — misma forma que el adaptador de LangChain enviado (`from surreal_memory.adapters.langchain import SurrealMemoryRetriever`; ver la sección API de Python), para el lado de LlamaIndex del ecosistema RAG.
- **Más proveedores de embeddings** — Voyage AI, Cohere, Mistral. Conjunto actual: Gemini, OpenAI, OpenRouter, Ollama, BGE-M3, sentence-transformers.
- **Bot de sincronización con upstream** — flujo programado que escanea `nhadaututtheky/neural-memory` en busca de nuevos commits, los clasifica como GREEN / YELLOW / RED contra nuestro fork y abre un PR de borrador para el lote verde.
- **Bot de Telegram bidireccional** — la integración de Telegram es unidireccional (notas de lanzamiento, copias de seguridad). Extiéndela con `smem remember` mediante comandos de bot.
- **Panel de benchmarks público** — `benchmarks/` ya tiene scripts de comparación contra mem0 y cognee; publica los resultados como un panel estático para que las afirmaciones sean verificables.
- **Plantillas de cerebro / packs iniciales** — cerebros precargados para flujos de trabajo comunes (desarrollo Python, administración K8s, notas de investigación).

## Licencia

MIT — consulta [LICENSE](LICENSE).
