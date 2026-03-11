/**
 * Standardized error responses — never leak internal details.
 */

import { Context } from "hono";

export class HubError extends Error {
  constructor(
    public readonly statusCode: number,
    message: string,
  ) {
    super(message);
  }
}

export function errorResponse(c: Context, status: number, message: string) {
  return c.json({ error: message }, status as 400);
}

export function handleError(c: Context, err: unknown) {
  if (err instanceof HubError) {
    return errorResponse(c, err.statusCode, err.message);
  }
  console.error("Unexpected error:", err);
  return errorResponse(c, 500, "Internal server error");
}
