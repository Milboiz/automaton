/**
 * Scheduled Trading Task
 *
 * Trading happens on the heartbeat, not inside the ReAct loop. The loop is
 * free-form and empirically will not commit to an order — it orients
 * indefinitely. Here the mechanism is fixed and only the *direction* is asked
 * of the model:
 *
 *   1. Read account, position and price (deterministic, via the wrapper script)
 *   2. Ask the model for exactly one of buy / sell / hold  <-- the only judgement
 *   3. Execute through the same wrapper, with the size clamped here
 *   4. Journal the decision
 *
 * Every failure path degrades to HOLD. A model that returns nonsense, a network
 * error, an unparseable reply, a missing script — none of them can produce a
 * trade. The only way an order is placed is an explicit, validated "buy" or
 * "sell" that also passes the balance checks below.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import type { TickContext, HeartbeatLegacyContext } from "../types.js";
import { createLogger } from "../observability/logger.js";

const execFileAsync = promisify(execFile);
const logger = createLogger("heartbeat.trade");

/** Hard ceiling on a single order, enforced here regardless of what the model asks for. */
const MAX_NOTIONAL = 100;
/** Alpaca rejects crypto orders below $10 notional. */
const MIN_NOTIONAL = 10;
/** Trend lookback. L100 is the pre-committed prior that survived out-of-sample testing. */
const TREND_LOOKBACK_DAYS = 100;

export function tradeScriptPath(): string {
  return (
    process.env.AUTOMATON_TRADE_SCRIPT ||
    path.join(os.homedir(), "automaton-trader", "trade.sh")
  );
}

function journalPath(): string {
  return path.join(path.dirname(tradeScriptPath()), "journal.md");
}

async function runScript(...args: string[]): Promise<string> {
  const { stdout } = await execFileAsync(tradeScriptPath(), args, {
    timeout: 30_000,
    maxBuffer: 1024 * 1024,
  });
  return stdout.trim();
}

export interface TradeDecision {
  action: "buy" | "sell" | "hold";
  notional: number;
  reason: string;
}

/**
 * Ask the model for a direction. Returns a hold on any failure -- an
 * unavailable or malformed answer must never become a trade.
 */
export async function decideDirection(params: {
  apiKey: string;
  model: string;
  cash: number;
  positionValue: number;
  price: number | null;
  trendPct: number | null;
  trendLookback: number;
}): Promise<TradeDecision> {
  const hold = (reason: string): TradeDecision => ({ action: "hold", notional: 0, reason });

  const prompt = [
    "You are deciding a single crypto trading action for a PAPER account trading BTC only.",
    "",
    `Cash available: $${params.cash.toFixed(2)}`,
    `Current BTC position value: $${params.positionValue.toFixed(2)}`,
    params.price !== null ? `Latest BTC price: $${params.price.toFixed(2)}` : "Latest BTC price: unknown",
    params.trendPct !== null
      ? `${params.trendLookback}-day trend: ${params.trendPct >= 0 ? "+" : ""}${params.trendPct.toFixed(2)}%`
      : `${params.trendLookback}-day trend: unknown`,
    "",
    "The strategy is a pre-committed long-only trend rule, chosen because it was the only",
    "hypothesis that survived out-of-sample testing on this data:",
    `  - Trend POSITIVE  -> be IN BTC. If mostly in cash, BUY.`,
    `  - Trend NEGATIVE  -> be IN CASH. If holding BTC, SELL to reduce exposure.`,
    "  - Already positioned correctly -> HOLD.",
    "",
    "Also respect this: rotating between crypto assets loses money even at zero cost, and",
    "over-trading destroys value. Move in modest steps and do not churn. Never trade merely",
    "because nothing has happened.",
    "",
    "Reply with ONLY a JSON object, no prose:",
    '{"action":"buy"|"sell"|"hold","notional":<number between 10 and 100>,"reason":"<one sentence>"}',
    'Use notional 0 when action is "hold".',
  ].join("\n");

  let raw: string;
  try {
    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${params.apiKey}`,
      },
      body: JSON.stringify({
        model: params.model,
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
        // gpt-5-class models spend this budget on reasoning tokens FIRST. Too
        // small a value returns an empty content string and no answer at all.
        max_completion_tokens: 3000,
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!resp.ok) {
      return hold(`inference failed: HTTP ${resp.status}`);
    }
    const data = (await resp.json()) as any;
    raw = data?.choices?.[0]?.message?.content ?? "";
    if (!raw.trim()) {
      const fr = data?.choices?.[0]?.finish_reason ?? "unknown";
      return hold(`model returned empty content (finish_reason: ${fr})`);
    }
  } catch (err: any) {
    return hold(`inference error: ${err?.message || String(err)}`);
  }

  let parsed: any;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return hold(`model reply was not valid JSON: ${raw.slice(0, 120)}`);
  }

  const action = parsed?.action;
  if (action !== "buy" && action !== "sell" && action !== "hold") {
    return hold(`model returned unrecognised action: ${String(action)}`);
  }
  if (action === "hold") {
    return hold(typeof parsed.reason === "string" ? parsed.reason : "model chose to hold");
  }

  const n = Number(parsed?.notional);
  if (!Number.isFinite(n) || n <= 0) {
    return hold("model gave no usable notional");
  }

  return {
    action,
    notional: Math.min(MAX_NOTIONAL, Math.max(MIN_NOTIONAL, n)),
    reason: typeof parsed.reason === "string" ? parsed.reason : "(no reason given)",
  };
}

export const tradeTick = async (
  _ctx: TickContext,
  taskCtx: HeartbeatLegacyContext,
): Promise<{ shouldWake: boolean; message?: string }> => {
  const script = tradeScriptPath();
  if (!fs.existsSync(script)) {
    logger.warn(`trade script not found at ${script}; skipping`);
    return { shouldWake: false };
  }

  const apiKey = taskCtx.config.openaiApiKey || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    logger.warn("no inference key available for trade decision; holding");
    return { shouldWake: false };
  }

  // 1. Deterministic state read.
  let cash = 0;
  let positionValue = 0;
  let price: number | null = null;
  let trendPct: number | null = null;
  try {
    const account = JSON.parse(await runScript("account"));
    cash = Number(account?.cash ?? 0);

    const positions = JSON.parse(await runScript("positions"));
    if (Array.isArray(positions)) {
      const btc = positions.find((p: any) => String(p?.symbol || "").startsWith("BTC"));
      positionValue = btc ? Number(btc.market_value ?? 0) : 0;
    }

    try {
      const bars = JSON.parse(await runScript("price"));
      const bar = bars?.bars?.["BTC/USD"];
      if (bar && Number.isFinite(Number(bar.c))) price = Number(bar.c);
    } catch {
      // Price is advisory only; its absence must not block or force a trade.
    }

    // Trend signal. Without it the model has nothing to reason from and can
    // only ever answer "hold", which is not a strategy.
    try {
      const hist = JSON.parse(await runScript("bars", String(TREND_LOOKBACK_DAYS + 20)));
      const series = hist?.bars?.["BTC/USD"];
      if (Array.isArray(series) && series.length > 1) {
        const closes = series
          .map((b: any) => Number(b?.c))
          .filter((c: number) => Number.isFinite(c));
        if (closes.length > 1) {
          const last = closes[closes.length - 1];
          // Use the oldest close within the lookback window we actually have.
          const idx = Math.max(0, closes.length - 1 - TREND_LOOKBACK_DAYS);
          const past = closes[idx];
          if (past > 0) trendPct = (last / past - 1) * 100;
        }
      }
    } catch {
      // Unknown trend -> the model is told so, and will hold.
    }
  } catch (err: any) {
    logger.error(`state read failed, holding: ${err?.message || String(err)}`);
    return { shouldWake: false };
  }

  // 2. The single judgement call.
  const model =
    taskCtx.config.modelStrategy?.lowComputeModel ||
    taskCtx.config.inferenceModel ||
    "gpt-5-mini";
  const decision = await decideDirection({
    apiKey, model, cash, positionValue, price,
    trendPct, trendLookback: TREND_LOOKBACK_DAYS,
  });

  // 3. Deterministic execution, with the balance checks the model cannot override.
  let outcome: string;
  if (decision.action === "hold") {
    outcome = "HOLD";
  } else if (decision.action === "buy" && cash < decision.notional) {
    outcome = `BLOCKED: wanted to buy $${decision.notional.toFixed(2)} but only $${cash.toFixed(2)} cash`;
  } else if (decision.action === "sell" && positionValue < decision.notional) {
    outcome = `BLOCKED: wanted to sell $${decision.notional.toFixed(2)} but position is only $${positionValue.toFixed(2)}`;
  } else {
    try {
      const result = await runScript(decision.action, decision.notional.toFixed(2));
      const order = JSON.parse(result);
      outcome = order?.id
        ? `${decision.action.toUpperCase()} $${decision.notional.toFixed(2)} -> order ${order.id} (${order.status})`
        : `${decision.action.toUpperCase()} $${decision.notional.toFixed(2)} -> unexpected reply: ${result.slice(0, 200)}`;
    } catch (err: any) {
      outcome = `ORDER FAILED: ${err?.message || String(err)}`;
    }
  }

  // 4. Journal every tick, including holds -- a decision not written down did not happen.
  const entry = [
    `## ${new Date().toISOString()}`,
    `- cash: $${cash.toFixed(2)} | BTC position: $${positionValue.toFixed(2)}` +
      (price !== null ? ` | price: $${price.toFixed(2)}` : "") +
      (trendPct !== null ? ` | ${TREND_LOOKBACK_DAYS}d trend: ${trendPct >= 0 ? "+" : ""}${trendPct.toFixed(2)}%` : ""),
    `- model (${model}) said: ${decision.action}${decision.notional ? ` $${decision.notional.toFixed(2)}` : ""}`,
    `- reason: ${decision.reason}`,
    `- outcome: ${outcome}`,
    "",
  ].join("\n");
  try {
    fs.appendFileSync(journalPath(), entry + "\n");
  } catch (err: any) {
    logger.warn(`could not write journal: ${err?.message || String(err)}`);
  }

  logger.info(`trade tick: ${outcome} (${decision.reason})`);
  taskCtx.db.setKV("last_trade_tick", JSON.stringify({ at: new Date().toISOString(), outcome }));

  return { shouldWake: false, message: outcome.startsWith("HOLD") ? undefined : `Trade: ${outcome}` };
};
