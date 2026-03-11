-- Neural Memory Sync Hub — Initial Schema

CREATE TABLE IF NOT EXISTS brains (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT '',
  name TEXT NOT NULL,
  config_json TEXT DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT NOT NULL,
  brain_id TEXT NOT NULL,
  device_name TEXT DEFAULT '',
  registered_at TEXT NOT NULL,
  last_sync_at TEXT,
  last_sync_sequence INTEGER DEFAULT 0,
  PRIMARY KEY (device_id, brain_id),
  FOREIGN KEY (brain_id) REFERENCES brains(id)
);

CREATE TABLE IF NOT EXISTS change_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  brain_id TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  operation TEXT NOT NULL,
  device_id TEXT NOT NULL,
  changed_at TEXT NOT NULL,
  payload TEXT DEFAULT '{}',
  FOREIGN KEY (brain_id) REFERENCES brains(id)
);

CREATE INDEX IF NOT EXISTS idx_change_log_brain_seq ON change_log(brain_id, id);
CREATE INDEX IF NOT EXISTS idx_change_log_brain_device ON change_log(brain_id, device_id);
