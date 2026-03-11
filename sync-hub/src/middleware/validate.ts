/**
 * Input validation for brain_id, device_id, and request bodies.
 */

import { HubError } from "../errors.js";
import type { ConflictStrategy } from "../types.js";
import { ConflictStrategy as CS } from "../types.js";

const BRAIN_ID_RE = /^[a-zA-Z0-9_\-.]+$/;
const DEVICE_ID_RE = /^[a-fA-F0-9]+$/;
const MAX_CHANGES = 500;
const MAX_BODY_BYTES = 1_048_576; // 1MB

export function validateBrainId(brainId: string): void {
  if (!brainId || brainId.length > 128 || !BRAIN_ID_RE.test(brainId)) {
    throw new HubError(422, "Invalid brain_id format");
  }
}

export function validateDeviceId(deviceId: string): void {
  if (!deviceId || deviceId.length > 32 || !DEVICE_ID_RE.test(deviceId)) {
    throw new HubError(
      422,
      "Invalid device_id format: must be hex characters only",
    );
  }
}

export function validateStrategy(strategy: string): ConflictStrategy {
  const valid = Object.values(CS) as string[];
  if (!valid.includes(strategy)) {
    throw new HubError(422, "Invalid conflict strategy");
  }
  return strategy as ConflictStrategy;
}

export function capChanges<T>(changes: T[]): T[] {
  return changes.slice(0, MAX_CHANGES);
}

export function checkContentLength(contentLength: string | null): void {
  if (contentLength && parseInt(contentLength, 10) > MAX_BODY_BYTES) {
    throw new HubError(413, "Request body too large (max 1MB)");
  }
}
