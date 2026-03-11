/**
 * D1 prepared statement wrappers for sync hub.
 */

import type { SyncChange, DeviceRecord } from "../types.js";

const MAX_CHANGES_RETURN = 1000;

// --- Brain ---

export async function getOrCreateBrain(
  db: D1Database,
  brainId: string,
): Promise<void> {
  const existing = await db
    .prepare("SELECT id FROM brains WHERE id = ?")
    .bind(brainId)
    .first<{ id: string }>();

  if (!existing) {
    const now = new Date().toISOString();
    await db
      .prepare(
        "INSERT INTO brains (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
      )
      .bind(brainId, brainId, now, now)
      .run();
  }
}

export async function brainExists(
  db: D1Database,
  brainId: string,
): Promise<boolean> {
  const row = await db
    .prepare("SELECT id FROM brains WHERE id = ?")
    .bind(brainId)
    .first<{ id: string }>();
  return row !== null;
}

// --- Device ---

export async function upsertDevice(
  db: D1Database,
  deviceId: string,
  brainId: string,
  deviceName: string,
): Promise<DeviceRecord> {
  const now = new Date().toISOString();

  await db
    .prepare(
      `INSERT INTO devices (device_id, brain_id, device_name, registered_at, last_sync_sequence)
       VALUES (?, ?, ?, ?, 0)
       ON CONFLICT (device_id, brain_id) DO UPDATE SET device_name = excluded.device_name`,
    )
    .bind(deviceId, brainId, deviceName, now)
    .run();

  return getDevice(db, deviceId, brainId);
}

export async function getDevice(
  db: D1Database,
  deviceId: string,
  brainId: string,
): Promise<DeviceRecord> {
  const row = await db
    .prepare(
      "SELECT device_id, brain_id, device_name, registered_at, last_sync_at, last_sync_sequence FROM devices WHERE device_id = ? AND brain_id = ?",
    )
    .bind(deviceId, brainId)
    .first<DeviceRecord>();

  if (!row) {
    throw new Error(`Device ${deviceId} not found for brain ${brainId}`);
  }
  return row;
}

export async function listDevices(
  db: D1Database,
  brainId: string,
): Promise<DeviceRecord[]> {
  const result = await db
    .prepare(
      "SELECT device_id, brain_id, device_name, registered_at, last_sync_at, last_sync_sequence FROM devices WHERE brain_id = ? ORDER BY registered_at ASC",
    )
    .bind(brainId)
    .all<DeviceRecord>();

  return result.results;
}

export async function updateDeviceSync(
  db: D1Database,
  deviceId: string,
  brainId: string,
  sequence: number,
): Promise<void> {
  const now = new Date().toISOString();
  await db
    .prepare(
      "UPDATE devices SET last_sync_at = ?, last_sync_sequence = ? WHERE device_id = ? AND brain_id = ?",
    )
    .bind(now, sequence, deviceId, brainId)
    .run();
}

// --- Change Log ---

export async function insertChanges(
  db: D1Database,
  brainId: string,
  changes: SyncChange[],
): Promise<void> {
  // D1 batch: up to 500 statements per batch
  const stmts = changes.map((c) =>
    db
      .prepare(
        "INSERT INTO change_log (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
      )
      .bind(
        brainId,
        c.entity_type,
        c.entity_id,
        c.operation,
        c.device_id,
        c.changed_at,
        JSON.stringify(c.payload),
      ),
  );

  if (stmts.length > 0) {
    await db.batch(stmts);
  }
}

export async function getChangesSince(
  db: D1Database,
  brainId: string,
  afterSequence: number,
  excludeDeviceId: string,
): Promise<SyncChange[]> {
  const result = await db
    .prepare(
      "SELECT id, entity_type, entity_id, operation, device_id, changed_at, payload FROM change_log WHERE brain_id = ? AND id > ? AND device_id != ? ORDER BY id ASC LIMIT ?",
    )
    .bind(brainId, afterSequence, excludeDeviceId, MAX_CHANGES_RETURN)
    .all<{
      id: number;
      entity_type: string;
      entity_id: string;
      operation: string;
      device_id: string;
      changed_at: string;
      payload: string;
    }>();

  return result.results.map((row) => ({
    sequence: row.id,
    entity_type: row.entity_type,
    entity_id: row.entity_id,
    operation: row.operation,
    device_id: row.device_id,
    changed_at: row.changed_at,
    payload: safeJsonParse(row.payload),
  }));
}

export async function getMaxSequence(
  db: D1Database,
  brainId: string,
): Promise<number> {
  const row = await db
    .prepare("SELECT MAX(id) as max_id FROM change_log WHERE brain_id = ?")
    .bind(brainId)
    .first<{ max_id: number | null }>();

  return row?.max_id ?? 0;
}

export async function getChangeLogStats(
  db: D1Database,
  brainId: string,
): Promise<{
  total_changes: number;
  latest_sequence: number;
}> {
  const row = await db
    .prepare(
      "SELECT COUNT(*) as total, MAX(id) as latest FROM change_log WHERE brain_id = ?",
    )
    .bind(brainId)
    .first<{ total: number; latest: number | null }>();

  return {
    total_changes: row?.total ?? 0,
    latest_sequence: row?.latest ?? 0,
  };
}

// --- Brain Ownership ---

export async function getBrainOwner(
  db: D1Database,
  brainId: string,
): Promise<string | null> {
  const row = await db
    .prepare("SELECT user_id FROM brains WHERE id = ?")
    .bind(brainId)
    .first<{ user_id: string }>();

  return row?.user_id || null;
}

export async function setBrainOwner(
  db: D1Database,
  brainId: string,
  userId: string,
): Promise<void> {
  await db
    .prepare("UPDATE brains SET user_id = ?, updated_at = ? WHERE id = ?")
    .bind(userId, new Date().toISOString(), brainId)
    .run();
}

// --- Helpers ---

function safeJsonParse(str: string): Record<string, unknown> {
  try {
    return JSON.parse(str) as Record<string, unknown>;
  } catch {
    return {};
  }
}
