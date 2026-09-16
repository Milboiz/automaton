/**
 * Supervise the trading engine.
 *
 * The two systems are one: a validated execution engine, and an agent watching
 * it. This task is the watching half, and it deliberately does NOT trade.
 *
 * The engine owns strategy, sizing, the risk engine, reconciliation, the fill
 * journal and the kill switch -- all of it tested. Re-deciding any of that here
 * would mean a language model second-guessing machinery that a backtest
 * validated, which is strictly worse. So this reads the engine's heartbeat and
 * reports, and nothing else.
 *
 * It exists because of a real failure: the engine sat HALTED for six days while
 * its log filled with routine lines. Nothing was watching, so nothing noticed.
 * A process cannot reliably report that it is wedged -- that is exactly the
 * state it cannot observe about itself -- so the observer has to be outside it.
 *
 * What it will not do:
 *   * place orders -- the engine does that
 *   * restart through a halt -- a halt is a human decision, and a supervisor
 *     that restarts through a breaker has disabled the breaker
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import type { TickContext, HeartbeatLegacyContext } from "../types.js";
import { createLogger } from "../observability/logger.js";

const logger = createLogger("heartbeat.supervise");

/** A tick older than this means the engine is wedged or gone, not merely idle. */
const STALE_TICK_MS = 15 * 60 * 1000;

export function enginePath(): string {
  return process.env.AUTOMATON_ENGINE_DIR ||
    path.join(os.homedir(), "crypto-trading-agent");
}

export type EngineVerdict =
  | "healthy"
  | "halted"
  | "stale"
  | "degraded"
  | "not-running"
  | "unknown";

export interface EngineHealth {
  verdict: EngineVerdict;
  detail: string;
  /** True when a human needs to look. Wakes the agent. */
  needsAttention: boolean;
}

/**
 * Judge the engine from its heartbeat file.
 *
 * `now` is injectable so staleness is testable without waiting.
 */
export function assessEngine(
  heartbeat: Record<string, unknown> | null,
  now: number = Date.now(),
): EngineHealth {
  if (!heartbeat) {
    return {
      verdict: "not-running",
      detail: "no heartbeat file -- the engine has never run, or its state was cleared",
      needsAttention: true,
    };
  }

  if (heartbeat.halted === true) {
    const reason = String(heartbeat.halt_reason ?? "no reason recorded");
    return {
      verdict: "halted",
      detail: `engine HALTED: ${reason}. Resume is a human action.`,
      needsAttention: true,
    };
  }

  const lastTick = heartbeat.heartbeat_at ?? heartbeat.last_tick_at;
  if (typeof lastTick !== "string") {
    return { verdict: "unknown", detail: "heartbeat has no timestamp", needsAttention: true };
  }

  const age = now - new Date(lastTick).getTime();
  if (!Number.isFinite(age)) {
    return { verdict: "unknown", detail: `unparseable timestamp: ${lastTick}`, needsAttention: true };
  }
  if (age > STALE_TICK_MS) {
    return {
      verdict: "stale",
      detail: `no tick for ${Math.round(age / 60000)} minutes -- wedged, or the process is gone`,
      needsAttention: true,
    };
  }

  // Running, but repeatedly failing to read the venue. Not yet fatal: the
  // engine halts itself at its own threshold. Worth saying out loud though,
  // because it is the shape of an outage starting.
  const consecutive = Number(heartbeat.consecutive_failures ?? 0);
  if (consecutive > 0) {
    return {
      verdict: "degraded",
      detail: `${consecutive} consecutive data failure(s); engine still running`,
      needsAttention: false,
    };
  }

  const ticks = Number(heartbeat.ticks ?? 0);
  const trades = Number(heartbeat.trades ?? 0);
  return {
    verdict: "healthy",
    detail: `${ticks} tick(s), ${trades} trade(s), last tick ${Math.round(age / 1000)}s ago`,
    needsAttention: false,
  };
}

export const superviseTick = async (
  _ctx: TickContext,
  taskCtx: HeartbeatLegacyContext,
): Promise<{ shouldWake: boolean; message?: string }> => {
  const hbPath = path.join(enginePath(), "reports", "heartbeat.json");

  let heartbeat: Record<string, unknown> | null = null;
  try {
    heartbeat = JSON.parse(fs.readFileSync(hbPath, "utf-8"));
  } catch {
    heartbeat = null;
  }

  const health = assessEngine(heartbeat);
  taskCtx.db.setKV("engine_health", JSON.stringify({
    at: new Date().toISOString(), verdict: health.verdict, detail: health.detail,
  }));

  if (health.needsAttention) {
    logger.warn(`engine ${health.verdict}: ${health.detail}`);
    return { shouldWake: true, message: `Trading engine ${health.verdict}: ${health.detail}` };
  }

  // Say the healthy case out loud too. Silence is how the six-day halt hid:
  // "fine" and "dead" look identical when neither prints anything.
  logger.info(`engine ${health.verdict}: ${health.detail}`);
  return { shouldWake: false };
};
