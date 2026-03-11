/**
 * Auth routes — register, profile, key management.
 *
 * POST /v1/auth/register — Create account + first API key
 * GET  /v1/auth/me — Profile + usage (requires auth)
 * POST /v1/auth/keys — Create additional API key (requires auth)
 * GET  /v1/auth/keys — List keys with prefix only (requires auth)
 * DELETE /v1/auth/keys/:keyId — Revoke a key (requires auth)
 */

import { Hono } from "hono";
import type { AppEnv } from "../types.js";
import { handleError, HubError } from "../errors.js";
import { generateApiKey, generateId } from "../auth/keys.js";
import { requireAuth } from "../middleware/auth.js";

const auth = new Hono<AppEnv>();

// --- Public: Register ---
auth.post("/register", async (c) => {
  try {
    const body = await c.req.json<{ email?: string; name?: string }>();

    const email = (body.email ?? "").trim().toLowerCase();
    if (!email || !email.includes("@")) {
      throw new HubError(422, "Valid email is required");
    }

    const db = c.env.SYNC_DB;

    // Check if email already exists
    const existing = await db
      .prepare("SELECT id FROM users WHERE email = ?")
      .bind(email)
      .first<{ id: string }>();

    if (existing) {
      throw new HubError(409, "Email already registered");
    }

    const userId = generateId();
    const now = new Date().toISOString();
    const key = await generateApiKey();
    const keyId = generateId();

    // Create user + first API key in batch
    await db.batch([
      db
        .prepare(
          "INSERT INTO users (id, email, name, tier, created_at, updated_at) VALUES (?, ?, ?, 'free', ?, ?)",
        )
        .bind(userId, email, body.name ?? "", now, now),
      db
        .prepare(
          "INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, created_at) VALUES (?, ?, ?, ?, 'default', ?)",
        )
        .bind(keyId, userId, key.hash, key.prefix, now),
    ]);

    return c.json({
      user_id: userId,
      email,
      api_key: key.raw, // Shown ONCE — never retrievable again
      key_prefix: key.prefix,
      tier: "free",
      message:
        "Save your API key now — it cannot be retrieved later. Use it in your NM config.",
    });
  } catch (err) {
    return handleError(c, err);
  }
});

// --- Protected routes below ---

// GET /me — Profile
auth.get("/me", requireAuth, async (c) => {
  try {
    const { userId } = c.get("auth");
    const db = c.env.SYNC_DB;

    const user = await db
      .prepare("SELECT id, email, name, tier, created_at FROM users WHERE id = ?")
      .bind(userId)
      .first<{
        id: string;
        email: string;
        name: string;
        tier: string;
        created_at: string;
      }>();

    if (!user) {
      throw new HubError(404, "User not found");
    }

    // Count brains and devices
    const stats = await db
      .prepare(
        `SELECT
          (SELECT COUNT(*) FROM brains WHERE user_id = ?) as brain_count,
          (SELECT COUNT(DISTINCT d.device_id) FROM devices d JOIN brains b ON b.id = d.brain_id WHERE b.user_id = ?) as device_count`,
      )
      .bind(userId, userId)
      .first<{ brain_count: number; device_count: number }>();

    return c.json({
      user_id: user.id,
      email: user.email,
      name: user.name,
      tier: user.tier,
      created_at: user.created_at,
      usage: {
        brains: stats?.brain_count ?? 0,
        devices: stats?.device_count ?? 0,
      },
    });
  } catch (err) {
    return handleError(c, err);
  }
});

// POST /keys — Create new key
auth.post("/keys", requireAuth, async (c) => {
  try {
    const { userId } = c.get("auth");
    const body = await c.req.json<{ name?: string }>();

    const db = c.env.SYNC_DB;
    const key = await generateApiKey();
    const keyId = generateId();
    const now = new Date().toISOString();

    await db
      .prepare(
        "INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      )
      .bind(keyId, userId, key.hash, key.prefix, body.name ?? "default", now)
      .run();

    return c.json({
      key_id: keyId,
      api_key: key.raw,
      key_prefix: key.prefix,
      name: body.name ?? "default",
      message: "Save your API key now — it cannot be retrieved later.",
    });
  } catch (err) {
    return handleError(c, err);
  }
});

// GET /keys — List keys (prefix only, never raw)
auth.get("/keys", requireAuth, async (c) => {
  try {
    const { userId } = c.get("auth");
    const db = c.env.SYNC_DB;

    const result = await db
      .prepare(
        "SELECT id, key_prefix, name, last_used_at, created_at, revoked_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
      )
      .bind(userId)
      .all<{
        id: string;
        key_prefix: string;
        name: string;
        last_used_at: string | null;
        created_at: string;
        revoked_at: string | null;
      }>();

    return c.json({
      keys: result.results.map((k) => ({
        key_id: k.id,
        prefix: k.key_prefix,
        name: k.name,
        last_used_at: k.last_used_at,
        created_at: k.created_at,
        active: !k.revoked_at,
      })),
    });
  } catch (err) {
    return handleError(c, err);
  }
});

// DELETE /keys/:keyId — Revoke
auth.delete("/keys/:keyId", requireAuth, async (c) => {
  try {
    const { userId } = c.get("auth");
    const keyId = c.req.param("keyId");
    const db = c.env.SYNC_DB;

    const key = await db
      .prepare("SELECT id FROM api_keys WHERE id = ? AND user_id = ?")
      .bind(keyId, userId)
      .first<{ id: string }>();

    if (!key) {
      throw new HubError(404, "Key not found");
    }

    await db
      .prepare("UPDATE api_keys SET revoked_at = ? WHERE id = ?")
      .bind(new Date().toISOString(), keyId)
      .run();

    return c.json({ revoked: true, key_id: keyId });
  } catch (err) {
    return handleError(c, err);
  }
});

export default auth;
