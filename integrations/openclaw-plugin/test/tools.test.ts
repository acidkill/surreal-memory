import { describe, it, expect, vi } from "vitest";
import { createTools, type ToolDefinition } from "../src/tools.js";
import type { NeuralMemoryMcpClient } from "../src/mcp-client.js";

function makeMockMcp(
  overrides: Partial<NeuralMemoryMcpClient> = {},
): NeuralMemoryMcpClient {
  return {
    connected: true,
    callTool: vi.fn().mockResolvedValue("{}"),
    connect: vi.fn(),
    ensureConnected: vi.fn().mockResolvedValue(undefined),
    close: vi.fn(),
    ...overrides,
  } as unknown as NeuralMemoryMcpClient;
}

// ── Schema helpers ────────────────────────────────────────

type SchemaObj = {
  type: string;
  properties?: Record<string, Record<string, unknown>>;
  required?: readonly string[];
  [key: string]: unknown;
};

function getSchema(tool: ToolDefinition): SchemaObj {
  return tool.parameters as unknown as SchemaObj;
}

function getPropNames(tool: ToolDefinition): string[] {
  const schema = getSchema(tool);
  return Object.keys(schema.properties ?? {});
}

// ── createTools ───────────────────────────────────────────

describe("createTools", () => {
  it("returns 6 tools", () => {
    const tools = createTools(makeMockMcp());
    expect(tools).toHaveLength(6);
  });

  it("returns tools with expected names", () => {
    const tools = createTools(makeMockMcp());
    const names = tools.map((t) => t.name);
    expect(names).toEqual([
      "nmem_remember",
      "nmem_recall",
      "nmem_context",
      "nmem_todo",
      "nmem_stats",
      "nmem_health",
    ]);
  });

  it("each tool has name, description, parameters, and execute", () => {
    const tools = createTools(makeMockMcp());
    for (const tool of tools) {
      expect(typeof tool.name).toBe("string");
      expect(tool.name.length).toBeGreaterThan(0);
      expect(typeof tool.description).toBe("string");
      expect(tool.description.length).toBeGreaterThan(0);
      expect(tool.parameters).toBeDefined();
      expect(typeof tool.execute).toBe("function");
    }
  });
});

// ── Schema compliance (Anthropic API requirements) ────────

describe("schema compliance", () => {
  let tools: ToolDefinition[];

  function findTool(name: string): ToolDefinition {
    const tool = tools.find((t) => t.name === name);
    if (!tool) throw new Error(`Tool ${name} not found`);
    return tool;
  }

  tools = createTools(makeMockMcp());

  it("every schema has type=object at root", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      expect(schema.type).toBe("object");
    }
  });

  it("every schema has properties key at root (even if empty)", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      expect(schema).toHaveProperty("properties");
      expect(typeof schema.properties).toBe("object");
    }
  });

  it("no schema uses $ref at root", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      expect(schema).not.toHaveProperty("$ref");
    }
  });

  it("no schema uses oneOf/allOf/anyOf at root", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      expect(schema).not.toHaveProperty("oneOf");
      expect(schema).not.toHaveProperty("allOf");
      expect(schema).not.toHaveProperty("anyOf");
    }
  });

  it("every property has a type field", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      for (const [propName, propSchema] of Object.entries(
        schema.properties ?? {},
      )) {
        expect(propSchema).toHaveProperty(
          "type",
          expect.any(String),
        );
      }
    }
  });

  it("every property has a description", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      for (const [, propSchema] of Object.entries(
        schema.properties ?? {},
      )) {
        expect(propSchema).toHaveProperty("description");
        expect(typeof propSchema.description).toBe("string");
        expect((propSchema.description as string).length).toBeGreaterThan(0);
      }
    }
  });

  it("required fields reference existing properties", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      const propNames = Object.keys(schema.properties ?? {});
      for (const req of schema.required ?? []) {
        expect(propNames).toContain(req);
      }
    }
  });

  it("every schema has additionalProperties=false (OpenAI strict mode)", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      expect(schema.additionalProperties).toBe(false);
    }
  });

  it("no schema uses integer type (use number for Gemini compat)", () => {
    for (const tool of tools) {
      const schema = getSchema(tool);
      for (const [, propSchema] of Object.entries(
        schema.properties ?? {},
      )) {
        expect(propSchema.type).not.toBe("integer");
      }
    }
  });

  it("schemas are plain JSON-serializable objects (no Zod, no class instances)", () => {
    for (const tool of tools) {
      const json = JSON.stringify(tool.parameters);
      const parsed = JSON.parse(json);
      expect(parsed).toEqual(tool.parameters);
    }
  });
});

// ── Individual tool schemas ───────────────────────────────

describe("tool schemas", () => {
  let tools: ToolDefinition[];

  function findTool(name: string): ToolDefinition {
    const tool = tools.find((t) => t.name === name);
    if (!tool) throw new Error(`Tool ${name} not found`);
    return tool;
  }

  tools = createTools(makeMockMcp());

  describe("nmem_remember", () => {
    it("has content as required", () => {
      const schema = getSchema(findTool("nmem_remember"));
      expect(schema.required).toContain("content");
    });

    it("has all expected properties", () => {
      const props = getPropNames(findTool("nmem_remember"));
      expect(props).toEqual(
        expect.arrayContaining([
          "content",
          "type",
          "priority",
          "tags",
          "expires_days",
        ]),
      );
    });

    it("content is string type", () => {
      const schema = getSchema(findTool("nmem_remember"));
      expect(schema.properties!.content).toHaveProperty("type", "string");
    });

    it("type has valid enum values", () => {
      const schema = getSchema(findTool("nmem_remember"));
      const typeSchema = schema.properties!.type as Record<string, unknown>;
      expect(typeSchema.enum).toEqual(
        expect.arrayContaining([
          "fact",
          "decision",
          "preference",
          "error",
          "instruction",
        ]),
      );
    });

    it("priority has number type", () => {
      const schema = getSchema(findTool("nmem_remember"));
      const prio = schema.properties!.priority as Record<string, unknown>;
      expect(prio.type).toBe("number");
    });

    it("tags is array of strings", () => {
      const schema = getSchema(findTool("nmem_remember"));
      const tags = schema.properties!.tags as Record<string, unknown>;
      expect(tags.type).toBe("array");
      expect(tags.items).toHaveProperty("type", "string");
    });
  });

  describe("nmem_recall", () => {
    it("has query as required", () => {
      const schema = getSchema(findTool("nmem_recall"));
      expect(schema.required).toContain("query");
    });

    it("has all expected properties", () => {
      const props = getPropNames(findTool("nmem_recall"));
      expect(props).toEqual(
        expect.arrayContaining([
          "query",
          "depth",
          "max_tokens",
          "min_confidence",
        ]),
      );
    });

    it("depth has number type", () => {
      const schema = getSchema(findTool("nmem_recall"));
      const depth = schema.properties!.depth as Record<string, unknown>;
      expect(depth.type).toBe("number");
    });

    it("min_confidence is number type", () => {
      const schema = getSchema(findTool("nmem_recall"));
      const conf = schema.properties!.min_confidence as Record<string, unknown>;
      expect(conf.type).toBe("number");
    });
  });

  describe("nmem_context", () => {
    it("has no required fields", () => {
      const schema = getSchema(findTool("nmem_context"));
      expect(schema.required ?? []).toHaveLength(0);
    });

    it("has limit and fresh_only properties", () => {
      const props = getPropNames(findTool("nmem_context"));
      expect(props).toContain("limit");
      expect(props).toContain("fresh_only");
    });
  });

  describe("nmem_todo", () => {
    it("has task as required", () => {
      const schema = getSchema(findTool("nmem_todo"));
      expect(schema.required).toContain("task");
    });

    it("has task and priority properties", () => {
      const props = getPropNames(findTool("nmem_todo"));
      expect(props).toContain("task");
      expect(props).toContain("priority");
    });
  });

  describe("nmem_stats / nmem_health", () => {
    it("have empty properties (no params)", () => {
      const statsSchema = getSchema(findTool("nmem_stats"));
      const healthSchema = getSchema(findTool("nmem_health"));
      expect(Object.keys(statsSchema.properties ?? {})).toHaveLength(0);
      expect(Object.keys(healthSchema.properties ?? {})).toHaveLength(0);
    });

    it("still have properties key present (Anthropic API requirement)", () => {
      const statsSchema = getSchema(findTool("nmem_stats"));
      const healthSchema = getSchema(findTool("nmem_health"));
      expect(statsSchema).toHaveProperty("properties");
      expect(healthSchema).toHaveProperty("properties");
    });
  });
});

// ── Tool execution ────────────────────────────────────────

describe("tool execution", () => {
  it("auto-connects when service not connected", async () => {
    const ensureConnected = vi.fn().mockResolvedValue(undefined);
    const callTool = vi.fn().mockResolvedValue("{}");
    const mcp = makeMockMcp({
      connected: false,
      ensureConnected,
      callTool,
    });
    const tools = createTools(mcp);
    await tools[0].execute("call-1", { content: "test" });
    expect(ensureConnected).toHaveBeenCalledOnce();
    expect(callTool).toHaveBeenCalledWith("nmem_remember", { content: "test" });
  });

  it("returns error when auto-connect fails", async () => {
    const ensureConnected = vi
      .fn()
      .mockRejectedValue(new Error("python not found"));
    const mcp = makeMockMcp({ connected: false, ensureConnected });
    const tools = createTools(mcp);
    const result = await tools[0].execute("call-1", { content: "test" });
    expect(result).toEqual({
      error: true,
      message: "NeuralMemory auto-connect failed: python not found",
    });
  });

  it("skips auto-connect when already connected", async () => {
    const ensureConnected = vi.fn();
    const mcp = makeMockMcp({ connected: true, ensureConnected });
    const tools = createTools(mcp);
    await tools[0].execute("call-1", { content: "test" });
    expect(ensureConnected).not.toHaveBeenCalled();
  });

  it("catches callTool exceptions and returns structured error", async () => {
    const mcp = makeMockMcp({
      callTool: vi.fn().mockRejectedValue(new Error("connection lost")),
    });
    const tools = createTools(mcp);
    const result = await tools[0].execute({ content: "test" });
    expect(result).toEqual({
      error: true,
      message: "Tool nmem_remember failed: connection lost",
    });
  });

  it("parses JSON response correctly", async () => {
    const mcp = makeMockMcp({
      callTool: vi
        .fn()
        .mockResolvedValue('{"answer": "hello", "confidence": 0.9}'),
    });
    const tools = createTools(mcp);
    const result = await tools[1].execute({ query: "test" });
    expect(result).toEqual({ answer: "hello", confidence: 0.9 });
  });

  it("handles non-JSON response as {text: raw}", async () => {
    const mcp = makeMockMcp({
      callTool: vi.fn().mockResolvedValue("plain text response"),
    });
    const tools = createTools(mcp);
    const result = await tools[0].execute({ content: "test" });
    expect(result).toEqual({ text: "plain text response" });
  });

  it("passes correct tool name and args to callTool", async () => {
    const callTool = vi.fn().mockResolvedValue("{}");
    const mcp = makeMockMcp({ callTool });
    const tools = createTools(mcp);

    await tools[0].execute("call-1", { content: "remember this", priority: 5 });
    expect(callTool).toHaveBeenCalledWith("nmem_remember", {
      content: "remember this",
      priority: 5,
    });
  });

  it("each tool routes to correct MCP tool name", async () => {
    const callTool = vi.fn().mockResolvedValue("{}");
    const mcp = makeMockMcp({ callTool });
    const tools = createTools(mcp);

    const expectedNames = [
      "nmem_remember",
      "nmem_recall",
      "nmem_context",
      "nmem_todo",
      "nmem_stats",
      "nmem_health",
    ];

    for (let i = 0; i < tools.length; i++) {
      callTool.mockClear();
      await tools[i].execute(`call-${i}`, {});
      expect(callTool).toHaveBeenCalledWith(expectedNames[i], {});
    }
  });

  it("handles empty string response as {text: ''}", async () => {
    const mcp = makeMockMcp({
      callTool: vi.fn().mockResolvedValue(""),
    });
    const tools = createTools(mcp);
    const result = await tools[0].execute({ content: "test" });
    expect(result).toEqual({ text: "" });
  });
});
