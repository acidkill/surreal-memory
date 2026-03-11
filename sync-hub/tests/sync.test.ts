/**
 * Integration tests for sync hub endpoints.
 */

import { describe, it, expect } from "vitest";
import app from "../src/index.js";
import type { Env, SyncResponse, RegisterDeviceResponse } from "../src/types.js";

// --- Helpers ---

const mockEnv = { SYNC_DB: {} } as unknown as Env;
const ctx = {} as ExecutionContext;

function makeReq(
  method: string,
  path: string,
  body?: unknown,
  headers?: Record<string, string>,
): Request {
  const init: RequestInit = {
    method,
    headers: { "Content-Type": "application/json", ...headers },
  };
  if (body) init.body = JSON.stringify(body);
  return new Request(`http://localhost${path}`, init);
}

// --- Public Endpoints (no auth required) ---

describe("Health endpoint", () => {
  it("returns ok status", async () => {
    const res = await app.fetch(makeReq("GET", "/v1/health"), mockEnv, ctx);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { status: string; version: string };
    expect(body.status).toBe("ok");
    expect(body.version).toBe("1.0.0");
  });
});

describe("Root endpoint", () => {
  it("returns service info", async () => {
    const res = await app.fetch(makeReq("GET", "/"), mockEnv, ctx);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { name: string };
    expect(body.name).toBe("Neural Memory Sync Hub");
  });
});

describe("404 handler", () => {
  it("returns 404 for unknown routes", async () => {
    const res = await app.fetch(makeReq("GET", "/v1/nonexistent"), mockEnv, ctx);
    expect(res.status).toBe(404);
  });
});

// --- Auth Required (hub routes reject without API key) ---

describe("Auth middleware", () => {
  it("rejects hub requests without API key", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/hub/register", {
        device_id: "abc123",
        brain_id: "default",
      }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("API key");
  });

  it("rejects hub sync without API key", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/hub/sync", {
        device_id: "abc123",
        brain_id: "default",
        last_sequence: 0,
        changes: [],
        strategy: "prefer_recent",
      }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(401);
  });

  it("rejects with invalid bearer format", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/hub/register", { device_id: "abc123", brain_id: "default" }, {
        Authorization: "Bearer invalid_key_format",
      }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("API key");
  });
});

// --- Auth register endpoint (public) ---

describe("Auth register validation", () => {
  it("rejects missing email", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/auth/register", { name: "Test" }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(422);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("email");
  });

  it("rejects invalid email", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/auth/register", { email: "notanemail" }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(422);
  });
});

// --- Type Shape Tests ---

describe("SyncResponse shape", () => {
  it("matches Python protocol fields", () => {
    const response: SyncResponse = {
      hub_sequence: 42,
      changes: [
        {
          sequence: 1,
          entity_type: "neuron",
          entity_id: "abc",
          operation: "insert",
          device_id: "def",
          changed_at: "2026-01-01T00:00:00Z",
          payload: { content: "test" },
        },
      ],
      conflicts: [],
      status: "success",
      message: "",
    };

    expect(response.hub_sequence).toBe(42);
    expect(response.changes).toHaveLength(1);
    expect(response.changes[0]!.entity_type).toBe("neuron");
    expect(response.status).toBe("success");
  });
});

describe("RegisterDeviceResponse shape", () => {
  it("matches Python protocol fields", () => {
    const response: RegisterDeviceResponse = {
      device_id: "abc123",
      device_name: "My PC",
      brain_id: "default",
      registered_at: "2026-01-01T00:00:00Z",
      last_sync_sequence: 0,
    };

    expect(response.device_id).toBe("abc123");
    expect(response.brain_id).toBe("default");
    expect(response.last_sync_sequence).toBe(0);
  });
});
