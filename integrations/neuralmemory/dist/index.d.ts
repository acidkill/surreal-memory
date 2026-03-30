/**
 * NeuralMemory — OpenClaw Memory Plugin
 *
 * Brain-inspired persistent memory for AI agents.
 * Occupies the exclusive "memory" plugin slot.
 *
 * Architecture:
 *   OpenClaw ←→ Plugin (TypeScript) ←→ MCP stdio ←→ NeuralMemory (Python)
 *
 * v1.7.0: Dynamic tool proxy — fetches all tools from MCP `tools/list`
 * instead of hardcoding 6 tools. Automatically exposes every tool the
 * MCP server provides (39+ tools in NM v2.28.0).
 *
 * v1.8.0: Compatible with NM v2.29.0 — RRF score fusion, graph-based
 * query expansion, and Personalized PageRank activation.
 *
 * v1.8.1: Fix async register() — OpenClaw requires synchronous registration.
 * Fallback tools registered sync; MCP connection deferred to service.start().
 *
 * v1.9.0: Backward-compat shim tools (memory_search, memory_get) to prevent
 * "allowList contains unknown entries" warnings when NM replaces memory-core.
 *
 * v1.10.0: Singleton MCP client — multiple workspaces (multi-agent) share
 * the same connected client instance, keyed by (pythonPath, brain). Fixes
 * "NeuralMemory service not running" when OpenClaw registers the plugin
 * for a second workspace after gateway startup.
 *
 * Registers:
 *   N tools    — dynamically from MCP server (fallback: 5 core + 2 compat)
 *   1 service  — MCP process lifecycle (start/stop)
 *   5 hooks    — before_prompt_build (auto-context), agent_end (auto-capture),
 *                before_compaction (flush), before_reset (flush),
 *                gateway_start (consolidation)
 */
import type { OpenClawPluginDefinition } from "./types.js";
/**
 * Strip metadata preamble from raw prompts before recall.
 *
 * OpenClaw + Telegram injects JSON metadata, NeuralMemory context blocks,
 * env vars, and system boilerplate into ev.prompt. Passing these raw to
 * nmem_recall creates junk neurons like "[concept] json message id".
 *
 * Stripping order matters — later passes clean up residue from earlier ones.
 */
export declare function stripPromptMetadata(raw: string): string;
/**
 * Strip NeuralMemory context noise and metadata from auto-capture text.
 *
 * When agent_end forwards assistant messages to nmem_auto, those messages
 * may contain NM context wrappers that were injected by before_prompt_build.
 * Re-ingesting these creates junk neurons like "[concept] json message id".
 *
 * This is defense-in-depth — the Python input_firewall also strips these,
 * but catching them here avoids wasting network round-trips.
 */
export declare function sanitizeAutoCapture(raw: string): string;
type PluginConfig = {
    pythonPath: string;
    brain: string;
    autoContext: boolean;
    autoCapture: boolean;
    autoFlush: boolean;
    autoConsolidate: boolean;
    contextDepth: number;
    maxContextTokens: number;
    timeout: number;
    initTimeout: number;
};
export declare const BRAIN_NAME_RE: RegExp;
export declare const MAX_AUTO_CAPTURE_CHARS = 50000;
export declare function resolveConfig(raw?: Record<string, unknown>): PluginConfig;
declare const plugin: OpenClawPluginDefinition;
export default plugin;
