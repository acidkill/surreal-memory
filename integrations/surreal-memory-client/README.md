# @acidkill/surreal-memory-client

TypeScript client for the [Surreal-Memory](https://github.com/acidkill/surreal-memory) REST API.

Version 3.12.2 aligns with the core maintenance release and keeps the client API
unchanged. Core consolidation and capture-policy changes are documented in the
[release changelog](https://github.com/acidkill/surreal-memory/blob/main/CHANGELOG.md).

```bash
npm install @acidkill/surreal-memory-client
```

## Quick start

```ts
import { SurrealMemoryClient } from "@acidkill/surreal-memory-client"

const client = new SurrealMemoryClient({
  baseUrl: "http://localhost:8000",
  brain: "myproject",
})

// Save a memory
const { fiber_id } = await client.remember({
  content: "Fixed auth bug with null check in login.py:42",
  type: "fix",
  priority: 7,
  tags: ["auth"],
})

// Recall related memories
const { results } = await client.recall({ query: "auth bug", limit: 5 })

for (const { fiber, score } of results) {
  console.log(score, fiber.content)
}

// Pull recent context
const { fibers } = await client.context({ limit: 20 })
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `remember(req)` | `POST /api/remember` | Save a memory. |
| `recall(req)` | `POST /api/recall` | Recall by query (spreading activation + optional vector). |
| `context(req?)` | `GET /api/context` | Recent fibers in the current brain. |
| `getFiber(id)` | `GET /api/fibers/:id` | Fetch one fiber. |
| `forget(id)` | `DELETE /api/fibers/:id` | Hard-delete a fiber. |
| `listBrains()` | `GET /api/brains` | List all brains. |
| `getBrainStats()` | `GET /api/stats` | Neuron / synapse / fiber counts. |
| `health()` | `GET /api/health` | Health probe. |

Every method accepts an optional `RequestOptions` argument (`brain` override, `timeoutMs`, `AbortSignal`, extra `headers`).

## Configuration

```ts
new SurrealMemoryClient({
  baseUrl: "https://memory.example.com",   // required, no trailing slash
  brain: "default",                         // optional default brain
  apiKey: process.env.SURREAL_MEMORY_API_KEY, // optional bearer token
  timeoutMs: 30_000,                        // default 30s
  fetch: customFetch,                       // optional fetch impl
  headers: { "X-Tenant": "acme" },          // optional default headers
})
```

## Error handling

```ts
import { ApiError, SurrealMemoryClient } from "@acidkill/surreal-memory-client"

try {
  await client.recall({ query: "..." })
} catch (err: unknown) {
  if (err instanceof ApiError) {
    console.error(err.status, err.message, err.payload)
  } else {
    throw err
  }
}
```

## Server compatibility

| Client | Server |
|---|---|
| `2.x` | Surreal-Memory `2.x` (REST API at `/api/*`) |

## License

MIT — see [LICENSE](https://github.com/acidkill/surreal-memory/blob/main/LICENSE).
