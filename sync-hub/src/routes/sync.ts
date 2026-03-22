/**
 * POST /v1/hub/sync — Push/pull incremental changes.
 *
 * Hub is a dumb ordered log: append incoming changes, return unseen changes.
 * NO conflict resolution on hub — client handles it.
 */

import { Hono } from "hono";
import type { AppEnv, SyncChange, SyncResponse, LicenseContext } from "../types.js";
import { SyncStatus } from "../types.js";
import { handleError, HubError } from "../errors.js";
import {
  validateBrainId,
  validateDeviceId,
  validateStrategy,
  capChanges,
  checkContentLength,
} from "../middleware/validate.js";
import {
  getOrCreateBrain,
  insertChanges,
  getChangesSince,
  getMaxSequence,
  upsertDevice,
  updateDeviceSync,
  getBrainOwner,
  setBrainOwner,
} from "../db/queries.js";

const sync = new Hono<AppEnv>();

sync.post("/", async (c) => {
  try {
    checkContentLength(c.req.header("content-length") ?? null);

    const body = await c.req.json<{
      device_id: string;
      brain_id: string;
      last_sequence: number;
      changes: SyncChange[];
      strategy: string;
    }>();

    validateBrainId(body.brain_id);
    validateDeviceId(body.device_id);
    validateStrategy(body.strategy ?? "prefer_recent");
    const changes = capChanges(body.changes ?? []);
    const lastSequence = Math.max(0, body.last_sequence ?? 0);

    const db = c.env.SYNC_DB;
    const { userId } = c.get("auth");

    // 1. Auto-create brain if needed
    await getOrCreateBrain(db, body.brain_id);

    // 2. Ownership check — first user to sync claims the brain
    const owner = await getBrainOwner(db, body.brain_id);
    if (!owner) {
      await setBrainOwner(db, body.brain_id, userId);
    } else if (owner !== userId) {
      throw new HubError(403, "You don't have access to this brain");
    }

    // 3. Ensure device is registered
    await upsertDevice(db, body.device_id, body.brain_id, "");

    // 4. Append incoming changes to change_log
    if (changes.length > 0) {
      const timestamped: SyncChange[] = changes.map((ch) => ({
        ...ch,
        device_id: body.device_id,
        changed_at: ch.changed_at || new Date().toISOString(),
      }));
      await insertChanges(db, body.brain_id, timestamped);
    }

    // 5. Get changes this device hasn't seen (excluding its own)
    const unseen = await getChangesSince(
      db,
      body.brain_id,
      lastSequence,
      body.device_id,
    );

    // 6. Get current max sequence
    const hubSequence = await getMaxSequence(db, body.brain_id);

    // 7. Update device's last sync position
    await updateDeviceSync(db, body.device_id, body.brain_id, hubSequence);

    // 8. Return — NO conflict resolution here
    const response: SyncResponse & { hints?: string[] } = {
      hub_sequence: hubSequence,
      changes: unseen,
      conflicts: [],
      status: SyncStatus.SUCCESS,
      message: `Pushed ${changes.length}, pulled ${unseen.length}`,
    };

    // 9. Upsell hint for free users with significant sync volume
    const license = c.get("license") as LicenseContext | undefined;
    const totalChanges = changes.length + unseen.length;
    if ((!license || license.tier === "free") && totalChanges > 50) {
      response.hints = [
        `Synced ${totalChanges} changes via full changelist. Pro users get Merkle delta sync (~95% faster).`,
      ];
    }

    return c.json(response);
  } catch (err) {
    return handleError(c, err);
  }
});

export default sync;
