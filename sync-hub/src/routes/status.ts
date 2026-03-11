/**
 * GET /v1/hub/status/:brainId — Sync stats for a brain.
 */

import { Hono } from "hono";
import type { AppEnv, HubStatusResponse } from "../types.js";
import { handleError, HubError } from "../errors.js";
import { validateBrainId } from "../middleware/validate.js";
import {
  brainExists,
  getChangeLogStats,
  listDevices,
  getBrainOwner,
} from "../db/queries.js";

const status = new Hono<AppEnv>();

status.get("/:brainId", async (c) => {
  try {
    const brainId = c.req.param("brainId");
    validateBrainId(brainId);

    const db = c.env.SYNC_DB;
    const { userId } = c.get("auth");

    if (!(await brainExists(db, brainId))) {
      throw new HubError(404, "Brain not found");
    }

    const owner = await getBrainOwner(db, brainId);
    if (owner && owner !== userId) {
      throw new HubError(403, "You don't have access to this brain");
    }

    const [stats, deviceList] = await Promise.all([
      getChangeLogStats(db, brainId),
      listDevices(db, brainId),
    ]);

    const response: HubStatusResponse = {
      brain_id: brainId,
      device_count: deviceList.length,
      change_log: {
        total_changes: stats.total_changes,
        synced_changes: stats.total_changes,
        unsynced_changes: 0,
        latest_sequence: stats.latest_sequence,
      },
    };

    return c.json(response);
  } catch (err) {
    return handleError(c, err);
  }
});

export default status;
