/**
 * Auth middleware — extracts API key from Authorization header,
 * verifies against D1, attaches user context.
 */

import type { Context, Next } from "hono";
import type { AppEnv, AuthContext } from "../types.js";
import { hashKey } from "../auth/keys.js";
import { HubError } from "../errors.js";

/**
 * Middleware that requires a valid API key.
 * Attaches userId and tier to c.set("auth", ...).
 */
export async function requireAuth(
  c: Context<AppEnv>,
  next: Next,
): Promise<Response | void> {
  const header = c.req.header("Authorization");
  if (!header || !header.startsWith("Bearer nmk_")) {
    throw new HubError(401, "Missing or invalid API key");
  }

  const rawKey = header.slice(7); // Remove "Bearer "
  const keyHash = await hashKey(rawKey);

  const db = c.env.SYNC_DB;

  const row = await db
    .prepare(
      `SELECT ak.id as key_id, ak.user_id, ak.revoked_at, u.tier
       FROM api_keys ak
       JOIN users u ON u.id = ak.user_id
       WHERE ak.key_hash = ?`,
    )
    .bind(keyHash)
    .first<{
      key_id: string;
      user_id: string;
      revoked_at: string | null;
      tier: string;
    }>();

  if (!row) {
    throw new HubError(401, "Invalid API key");
  }

  if (row.revoked_at) {
    throw new HubError(401, "API key has been revoked");
  }

  // Update last_used_at (fire-and-forget, don't block response)
  c.executionCtx.waitUntil(
    db
      .prepare("UPDATE api_keys SET last_used_at = ? WHERE id = ?")
      .bind(new Date().toISOString(), row.key_id)
      .run(),
  );

  c.set("auth", { userId: row.user_id, tier: row.tier } satisfies AuthContext);
  await next();
}
