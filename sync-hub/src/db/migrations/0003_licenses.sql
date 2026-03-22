-- License management for Pro/Team tiers

CREATE TABLE IF NOT EXISTS licenses (
  id TEXT PRIMARY KEY,                    -- nm_pro_XXXX_XXXX_XXXX
  user_id TEXT NOT NULL,
  tier TEXT NOT NULL DEFAULT 'pro',       -- pro, team
  status TEXT NOT NULL DEFAULT 'active',  -- active, expired, cancelled
  payment_provider TEXT DEFAULT '',       -- lemonsqueezy, stripe
  payment_id TEXT DEFAULT '',             -- external order ID
  created_at TEXT NOT NULL,
  expires_at TEXT,                        -- NULL = lifetime
  cancelled_at TEXT,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_licenses_user ON licenses(user_id);
CREATE INDEX IF NOT EXISTS idx_licenses_status ON licenses(status);
