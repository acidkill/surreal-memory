import { Hono } from "hono";
import { cors } from "hono/cors";
import type { AppEnv } from "./types.js";
import checkout from "./routes/checkout.js";
import order from "./routes/order.js";
import webhook from "./routes/webhook.js";

const app = new Hono<AppEnv>();

// CORS for landing page
app.use(
  "*",
  cors({
    origin: [
      "https://neuralmemory.theio.vn",
      "https://nhadaututtheky.github.io",
      "http://localhost:3000",
    ],
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
  }),
);

// Health
app.get("/", (c) =>
  c.json({
    name: "Neural Memory Pay Hub",
    version: "1.0.0",
    status: "healthy",
  }),
);

app.get("/health", (c) => c.json({ status: "ok" }));

// Routes
app.route("/checkout", checkout);
app.route("/order", order);
app.route("/webhook", webhook);

// Pro features granted on all verified licenses
const PRO_FEATURES = [
  "merkle_sync",
  "cone_queries",
  "directional_compression",
  "cross_encoder",
  "smart_merge",
  "infinity_db",
];

// Verify endpoint — D1 primary, XLabs API fallback
app.post("/verify", async (c) => {
  const body = await c.req.json<{ key: string }>();
  const key = (body.key || "").trim();

  if (!key) {
    return c.json({ valid: false, error: "Missing key" });
  }

  // Normalize: nm_pro_xxxx → NM-PRO-XXXX
  const normalizedKey = key.startsWith("nm_")
    ? key.replaceAll("_", "-").toUpperCase()
    : key.toUpperCase();

  // 1. Check D1 (primary — fulfilled orders have license_key)
  const db = c.env.PAY_DB;
  const d1Order = await db
    .prepare(
      "SELECT product, license_key, fulfilled_at FROM orders WHERE UPPER(license_key) = ? AND status = 'fulfilled'",
    )
    .bind(normalizedKey)
    .first<{ product: string; license_key: string; fulfilled_at: string }>();

  if (d1Order) {
    const tier = d1Order.product.includes("TEAM") ? "team" : "pro";
    return c.json({
      valid: true,
      tier,
      expires_at: null,
      features: PRO_FEATURES,
    });
  }

  // 2. Fallback: check XLabs API
  try {
    const res = await fetch("https://admin.theio.vn/api/licenses", {
      headers: { Authorization: `Bearer ${c.env.XLABS_API_KEY}` },
    });

    if (res.ok) {
      const data = await res.json<{
        data: Array<{
          license_key: string;
          project_slug: string;
          tier: string;
          status: string;
          features_json: string;
          expires_at: string | null;
        }>;
      }>();

      const match = data.data?.find(
        (l) =>
          l.license_key.toUpperCase() === normalizedKey &&
          l.project_slug === "neural-memory" &&
          l.status === "active",
      );

      if (match) {
        let features: string[] = [];
        try {
          features = JSON.parse(match.features_json || "[]");
        } catch {
          features = PRO_FEATURES;
        }
        return c.json({
          valid: true,
          tier: match.tier,
          expires_at: match.expires_at,
          features,
        });
      }
    }
  } catch {
    // XLabs API unreachable — D1 was already checked, report not found
  }

  return c.json({ valid: false, error: "Invalid or expired license key" });
});

export default app;
