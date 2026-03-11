/**
 * GET /v1/hub/devices/:brainId — List devices for a brain.
 */

import { Hono } from "hono";
import type { AppEnv } from "../types.js";
import { handleError, HubError } from "../errors.js";
import { validateBrainId } from "../middleware/validate.js";
import { brainExists, listDevices, getBrainOwner } from "../db/queries.js";

const devices = new Hono<AppEnv>();

devices.get("/:brainId", async (c) => {
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

    const deviceList = await listDevices(db, brainId);

    return c.json({
      brain_id: brainId,
      devices: deviceList.map((d) => ({
        device_id: d.device_id,
        device_name: d.device_name,
        registered_at: d.registered_at,
        last_sync_sequence: d.last_sync_sequence,
      })),
    });
  } catch (err) {
    return handleError(c, err);
  }
});

export default devices;
