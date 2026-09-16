/**
 * Provider resolution for the scheduled trade decision.
 *
 * trade_tick originally hardcoded api.openai.com, which meant an exhausted
 * OpenAI balance stopped the agent trading entirely even though the registry
 * already knew about free, OpenAI-compatible providers. These pin the order
 * and the request-shape differences between them.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { resolveInferenceTarget } from "../heartbeat/trade-task.js";

const ENV_KEYS = [
  "AUTOMATON_TRADE_BASE_URL",
  "AUTOMATON_TRADE_API_KEY",
  "AUTOMATON_TRADE_MODEL",
  "GROQ_API_KEY",
  "OPENAI_API_KEY",
];

describe("resolveInferenceTarget", () => {
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = {};
    for (const key of ENV_KEYS) {
      saved[key] = process.env[key];
      delete process.env[key];
    }
  });

  afterEach(() => {
    for (const key of ENV_KEYS) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key] as string;
    }
  });

  it("returns null when nothing is configured, so the caller holds", () => {
    expect(resolveInferenceTarget({})).toBeNull();
  });

  it("prefers Groq over OpenAI -- its free tier covers this task many times over", () => {
    const target = resolveInferenceTarget({
      groqApiKey: "gsk_test",
      openaiApiKey: "sk-test",
    });
    expect(target?.baseUrl).toBe("https://api.groq.com/openai/v1");
    expect(target?.apiKey).toBe("gsk_test");
    expect(target?.model).toBe("llama-3.3-70b-versatile");
  });

  it("falls back to OpenAI when no Groq key exists", () => {
    const target = resolveInferenceTarget({ openaiApiKey: "sk-test" });
    expect(target?.baseUrl).toBe("https://api.openai.com/v1");
    expect(target?.apiKey).toBe("sk-test");
  });

  it("reads GROQ_API_KEY from the environment", () => {
    process.env.GROQ_API_KEY = "gsk_env";
    expect(resolveInferenceTarget({})?.apiKey).toBe("gsk_env");
  });

  it("explicit env override wins, so any compatible endpoint works uncoded", () => {
    // e.g. NVIDIA NIM, Together, or a local Ollama on :11434/v1
    process.env.AUTOMATON_TRADE_BASE_URL = "http://localhost:11434/v1";
    process.env.AUTOMATON_TRADE_API_KEY = "ollama";
    process.env.AUTOMATON_TRADE_MODEL = "llama3.2";
    process.env.GROQ_API_KEY = "gsk_ignored";
    const target = resolveInferenceTarget({ openaiApiKey: "sk-ignored" });
    expect(target?.baseUrl).toBe("http://localhost:11434/v1");
    expect(target?.model).toBe("llama3.2");
  });

  it("strips a trailing slash so the path is never doubled", () => {
    process.env.AUTOMATON_TRADE_BASE_URL = "https://example.com/v1/";
    process.env.AUTOMATON_TRADE_API_KEY = "k";
    expect(resolveInferenceTarget({})?.baseUrl).toBe("https://example.com/v1");
  });

  it("an override needs BOTH url and key, or it is ignored", () => {
    // A half-configured override must not silently produce an unauthenticated
    // request; fall through to a provider that actually has credentials.
    process.env.AUTOMATON_TRADE_BASE_URL = "https://example.com/v1";
    const target = resolveInferenceTarget({ openaiApiKey: "sk-test" });
    expect(target?.baseUrl).toBe("https://api.openai.com/v1");
  });
});
