/**
 * POST /v1/hub/register — Register a device for a brain.
 */

import { Hono } from "hono";
import type { AppEnv, RegisterDeviceResponse } from "../types.js";
import { handleError, HubError } from "../errors.js";
import { validateBrainId, validateDeviceId } from "../middleware/validate.js";
import {
  getOrCreateBrain,
  upsertDevice,
  getBrainOwner,
  setBrainOwner,
} from "../db/queries.js";

const register = new Hono<AppEnv>();

register.post("/", async (c) => {
  try {
    const body = await c.req.json<{
      device_id: string;
      brain_id: string;
      device_name?: string;
    }>();

    validateBrainId(body.brain_id);
    validateDeviceId(body.device_id);

    const db = c.env.SYNC_DB;
    const { userId } = c.get("auth");

    await getOrCreateBrain(db, body.brain_id);

    // Ownership: first user claims brain, others rejected
    const owner = await getBrainOwner(db, body.brain_id);
    if (!owner) {
      await setBrainOwner(db, body.brain_id, userId);
    } else if (owner !== userId) {
      throw new HubError(403, "You don't have access to this brain");
    }

    const device = await upsertDevice(
      db,
      body.device_id,
      body.brain_id,
      body.device_name ?? "",
    );

    const response: RegisterDeviceResponse = {
      device_id: device.device_id,
      device_name: device.device_name,
      brain_id: body.brain_id,
      registered_at: device.registered_at,
      last_sync_sequence: device.last_sync_sequence,
    };

    return c.json(response);
  } catch (err) {
    return handleError(c, err);
  }
});

export default register;
