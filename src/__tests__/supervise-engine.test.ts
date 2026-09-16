/**
 * Supervision of the trading engine.
 *
 * The engine once sat HALTED for six days while its log filled with routine
 * lines and nothing noticed. These pin the states that must raise a hand --
 * and, just as importantly, that a healthy engine does not cry wolf.
 */

import { describe, it, expect } from "vitest";
import { assessEngine } from "../heartbeat/supervise-task.js";

const NOW = Date.parse("2026-09-16T12:00:00Z");
const fresh = (over: Record<string, unknown> = {}) => ({
  ticks: 12, trades: 3, data_failures: 0, consecutive_failures: 0,
  halted: false, halt_reason: null,
  heartbeat_at: "2026-09-16T11:59:30Z",
  ...over,
});

describe("assessEngine", () => {
  it("reports healthy when ticking recently and not halted", () => {
    const h = assessEngine(fresh(), NOW);
    expect(h.verdict).toBe("healthy");
    expect(h.needsAttention).toBe(false);
  });

  it("raises a hand when the engine is halted, and carries the reason", () => {
    const h = assessEngine(
      fresh({ halted: true, halt_reason: "reconciliation break -- 1 divergence" }),
      NOW,
    );
    expect(h.verdict).toBe("halted");
    expect(h.needsAttention).toBe(true);
    expect(h.detail).toContain("reconciliation break");
  });

  it("catches the six-day failure: stale ticks with halted=false", () => {
    // This is the exact shape that hid. Not halted, no errors -- simply not
    // ticking. Only an outside observer can see it.
    const h = assessEngine(fresh({ heartbeat_at: "2026-09-10T12:00:00Z" }), NOW);
    expect(h.verdict).toBe("stale");
    expect(h.needsAttention).toBe(true);
    expect(h.detail).toContain("minutes");
  });

  it("treats a missing heartbeat file as needing attention", () => {
    const h = assessEngine(null, NOW);
    expect(h.verdict).toBe("not-running");
    expect(h.needsAttention).toBe(true);
  });

  it("notes data failures without escalating -- the engine halts itself", () => {
    // Worth saying out loud, but the engine owns its own threshold. Escalating
    // here would second-guess machinery that already has a rule.
    const h = assessEngine(fresh({ consecutive_failures: 2 }), NOW);
    expect(h.verdict).toBe("degraded");
    expect(h.needsAttention).toBe(false);
  });

  it("does not mistake a just-started engine for a stale one", () => {
    const h = assessEngine(fresh({ ticks: 0, trades: 0 }), NOW);
    expect(h.verdict).toBe("healthy");
  });

  it("flags an unparseable timestamp rather than silently passing", () => {
    const h = assessEngine(fresh({ heartbeat_at: "not-a-date" }), NOW);
    expect(h.verdict).toBe("unknown");
    expect(h.needsAttention).toBe(true);
  });

  it("falls back to last_tick_at when heartbeat_at is absent", () => {
    const h = assessEngine(
      { halted: false, last_tick_at: "2026-09-16T11:59:00Z", ticks: 1 },
      NOW,
    );
    expect(h.verdict).toBe("healthy");
  });

  it("prefers halted over stale -- the reason is what a human needs", () => {
    const h = assessEngine(
      fresh({ halted: true, halt_reason: "kill switch", heartbeat_at: "2026-09-01T00:00:00Z" }),
      NOW,
    );
    expect(h.verdict).toBe("halted");
    expect(h.detail).toContain("kill switch");
  });
});
