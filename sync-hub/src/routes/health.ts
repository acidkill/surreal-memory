/**
 * GET /v1/health — Simple uptime check.
 */

import { Hono } from "hono";
import type { AppEnv } from "../types.js";

const health = new Hono<AppEnv>();

health.get("/", (c) => {
  return c.json({
    status: "ok",
    version: "1.0.0",
    service: "neural-memory-sync-hub",
  });
});

export default health;
