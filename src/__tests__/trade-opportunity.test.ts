/**
 * The opportunity gate.
 *
 * Running the model on a timer asks the same question of unchanged data and
 * pays for every answer. The gate is free arithmetic, so it can run often; the
 * model is woken only when the position actually diverges from the trend rule.
 */

import { describe, it, expect } from "vitest";
import { findOpportunity } from "../heartbeat/trade-task.js";

describe("findOpportunity", () => {
  it("acts when the trend is up but the account is in cash", () => {
    const o = findOpportunity({ cash: 480, positionValue: 0, trendPct: 20 });
    expect(o.act).toBe(true);
    expect(o.targetPct).toBeCloseTo(0.95);
  });

  it("acts when the trend turns down while still holding", () => {
    const o = findOpportunity({ cash: 20, positionValue: 460, trendPct: -14.9 });
    expect(o.act).toBe(true);
    expect(o.targetPct).toBe(0);   // binary: out means out
  });

  it("does nothing when already positioned correctly", () => {
    // Exactly at target: the gap is $0, so the venue-minimum check fires first.
    const o = findOpportunity({ cash: 24, positionValue: 456, trendPct: 20 });
    expect(o.act).toBe(false);
  });

  it("does nothing for drift inside the band, even above the venue minimum", () => {
    // 92% vs a 95% target on $480 is a $14.40 gap -- tradeable, but 3% drift is
    // noise on a 100-day trend, and rebalancing on it is the churn FINDINGS.md
    // measures as loss-making.
    const o = findOpportunity({ cash: 38.4, positionValue: 441.6, trendPct: 20 });
    expect(o.act).toBe(false);
    expect(o.reason).toContain("band");
  });

  it("does nothing when the gap is below the venue minimum", () => {
    // A $4 gap cannot be traded at all -- Alpaca rejects crypto orders under $10.
    const o = findOpportunity({ cash: 28, positionValue: 472, trendPct: 20 });
    expect(o.act).toBe(false);
    expect(o.reason).toContain("minimum");
  });

  it("never acts on an unknown trend", () => {
    // Missing data must not provoke a trade, in either direction.
    const o = findOpportunity({ cash: 480, positionValue: 0, trendPct: null });
    expect(o.act).toBe(false);
    expect(o.reason).toContain("unknown");
  });

  it("treats a flat-but-positive trend as invested", () => {
    const o = findOpportunity({ cash: 480, positionValue: 0, trendPct: 0.01 });
    expect(o.act).toBe(true);
    expect(o.targetPct).toBeCloseTo(0.95);
  });

  it("treats exactly zero trend as out, not in", () => {
    // `> 0` is the rule; zero is not an uptrend. Pinned because an off-by-one
    // here silently inverts the strategy at the crossover.
    const o = findOpportunity({ cash: 20, positionValue: 460, trendPct: 0 });
    expect(o.targetPct).toBe(0);
    expect(o.act).toBe(true);
  });

  it("handles an empty account without dividing by zero", () => {
    const o = findOpportunity({ cash: 0, positionValue: 0, trendPct: 20 });
    expect(o.act).toBe(false);
    expect(Number.isFinite(o.currentPct)).toBe(true);
  });
});
