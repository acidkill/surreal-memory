const BASE_URL = ""

interface FetchOptions extends RequestInit {
  brain?: string
}

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

async function request<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { brain, ...fetchOptions } = options

  const headers = new Headers(fetchOptions.headers)
  if (brain) {
    headers.set("X-Brain-ID", brain)
  }
  if (fetchOptions.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...fetchOptions,
    headers,
  })

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error")
    throw new ApiError(response.status, text)
  }

  return response.json() as Promise<T>
}

export const api = {
  get: <T>(path: string, options?: FetchOptions) =>
    request<T>(path, { ...options, method: "GET" }),

  post: <T>(path: string, body?: unknown, options?: FetchOptions) =>
    request<T>(path, {
      ...options,
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    }),

  put: <T>(path: string, body?: unknown, options?: FetchOptions) =>
    request<T>(path, {
      ...options,
      method: "PUT",
      body: body ? JSON.stringify(body) : undefined,
    }),

  delete: <T>(path: string, options?: FetchOptions) =>
    request<T>(path, { ...options, method: "DELETE" }),
}

/**
 * Extract a human-readable message from an ApiError. FastAPI returns
 * `{"detail": "..."}` bodies, which `request()` stores verbatim as the error
 * message — parse it out so toasts show the message, not raw JSON.
 */
export function apiErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    try {
      const parsed = JSON.parse(err.message)
      if (parsed && typeof parsed.detail === "string") return parsed.detail
    } catch {
      /* body was not JSON — fall through to the raw text */
    }
    return err.message || fallback
  }
  return fallback
}

export { ApiError }
