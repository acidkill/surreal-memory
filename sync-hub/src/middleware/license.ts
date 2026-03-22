/**
 * License tier check middleware.
 *
 * Checks if the authenticated user has an active Pro/Team license.
 * Does NOT block free users — just sets tier context for downstream
 * routes to decide behavior (e.g. full sync vs merkle sync).
 */

import type { Context, Next } from "hono";
import type { AppEnv } from "../types.js";

export interface LicenseContext {
  tier: "free" | "pro" | "team";
  licenseId: string | null;
  expiresAt: string | null;
}

/**
 * Attach license info to context. Never blocks — just enriches.
 * Downstream routes check c.get("license").tier to gate features.
 */
export async function attachLicense(
  c: Context<AppEnv>,
  next: Next,
): Promise<Response | void> {
  const auth = c.get("auth");
  const db = c.env.SYNC_DB;

  // Default to free
  let license: LicenseContext = {
    tier: "free",
    licenseId: null,
    expiresAt: null,
  };

  // Check for active license
  const row = await db
    .prepare(
      `SELECT id, tier, expires_at
       FROM licenses
       WHERE user_id = ? AND status = 'active'
       ORDER BY created_at DESC
       LIMIT 1`,
    )
    .bind(auth.userId)
    .first<{
      id: string;
      tier: string;
      expires_at: string | null;
    }>();

  if (row) {
    // Check expiry
    const isExpired =
      row.expires_at && new Date(row.expires_at) < new Date();

    if (!isExpired) {
      license = {
        tier: row.tier as "pro" | "team",
        licenseId: row.id,
        expiresAt: row.expires_at,
      };
    } else {
      // Auto-expire the license (fire-and-forget)
      c.executionCtx.waitUntil(
        db
          .prepare("UPDATE licenses SET status = 'expired' WHERE id = ?")
          .bind(row.id)
          .run(),
      );
    }
  }

  c.set("license", license);
  await next();
}

/**
 * Check if user has Pro or Team tier.
 * Use in routes: `if (!isPro(c)) { return upsell response }`
 */
export function isPro(c: Context<AppEnv>): boolean {
  const license = c.get("license") as LicenseContext | undefined;
  return license?.tier === "pro" || license?.tier === "team";
}
