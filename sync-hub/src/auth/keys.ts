/**
 * API key generation and hashing.
 *
 * Key format: nmk_ + 32 hex chars (128 bits entropy)
 * Stored as SHA-256 hash — raw key never persisted.
 */

const KEY_PREFIX = "nmk_";
const KEY_BYTES = 16; // 128 bits = 32 hex chars

export interface GeneratedKey {
  raw: string; // nmk_a1b2c3d4... (shown to user ONCE)
  hash: string; // SHA-256 hex digest (stored in D1)
  prefix: string; // nmk_a1b2c3d4 (first 12 chars, for display)
}

export async function generateApiKey(): Promise<GeneratedKey> {
  const bytes = new Uint8Array(KEY_BYTES);
  crypto.getRandomValues(bytes);

  const hex = Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  const raw = `${KEY_PREFIX}${hex}`;
  const hash = await hashKey(raw);
  const prefix = raw.slice(0, 12);

  return { raw, hash, prefix };
}

export async function hashKey(raw: string): Promise<string> {
  const encoded = new TextEncoder().encode(raw);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export function generateId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  // UUID v4 format
  bytes[6] = (bytes[6]! & 0x0f) | 0x40;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join("-");
}
