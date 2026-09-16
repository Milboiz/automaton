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
/** Target BTC weight when the trend is positive. The remainder is a deliberate
 *  fee buffer: fees are charged on top, so an order sized to the whole cash
 *  balance is rejected. */
const TARGET_INVESTED_PCT = 0.95;

/** The trend comes from DAILY bars, so recomputing it every poll fetches 120
 *  days of history to produce an identical number. Cache it; the underlying
 *  data cannot change faster than once a day. */
const TREND_CACHE_MS = 15 * 60 * 1000;
let _trendCache: { at: number; value: number | null } | null = null;

/** After acting, wait before acting again. Without this a fast poll can fire
 *  several orders into the same gap before the first fill settles, turning one
 *  intended trade into a burst. */
const ACTION_COOLDOWN_MS = 60 * 1000;
let _lastActionAt = 0;

/** Exposed for tests. */
export function _resetTradeTickState(): void {
  _trendCache = null;
  _lastActionAt = 0;
}

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

/** Minimum gap between actual and target weight before it is worth acting.
 *  Below this the trade would be noise: the strategy is a slow trend rule, and
 *  rebalancing on small drift is the churn FINDINGS.md measures as loss-making. */
const REBALANCE_BAND = 0.05;

export interface Opportunity {
  act: boolean;
  reason: string;
  currentPct: number;
  targetPct: number;
}

/**
 * Cheap, deterministic check for whether there is anything to do.
 *
 * This is the whole point of running often: the arithmetic is free, so it can
 * run every few minutes, and the model is only woken when the position
 * genuinely diverges from what the trend rule wants. Polling a language model
 * on a timer asks the same question of unchanged data and pays for the answer
 * every time.
 *
 * No trend reading is NOT an opportunity -- unknown data must never provoke a
 * trade.
 */
export function findOpportunity(params: {
  cash: number;
  positionValue: number;
  trendPct: number | null;
  targetInvested?: number;
}): Opportunity {
  const equity = params.cash + params.positionValue;
  const targetInvested = params.targetInvested ?? TARGET_INVESTED_PCT;
  const currentPct = equity > 0 ? params.positionValue / equity : 0;

  if (params.trendPct === null) {
    return { act: false, reason: "trend unknown -- holding", currentPct, targetPct: currentPct };
  }

  // The rule is binary: in when the trend is up, out when it is down.
  const targetPct = params.trendPct > 0 ? targetInvested : 0;
  const drift = Math.abs(currentPct - targetPct);
  const driftValue = drift * equity;

  if (driftValue < MIN_NOTIONAL) {
    return {
      act: false,
      reason: `gap $${driftValue.toFixed(2)} is below the $${MIN_NOTIONAL} venue minimum`,
      currentPct, targetPct,
    };
  }
  if (drift < REBALANCE_BAND) {
    return {
      act: false,
      reason: `within ${(REBALANCE_BAND * 100).toFixed(0)}% band (${(currentPct * 100).toFixed(1)}% vs ${(targetPct * 100).toFixed(0)}% target)`,
      currentPct, targetPct,
    };
  }
  return {
    act: true,
    reason: `${(currentPct * 100).toFixed(1)}% invested vs ${(targetPct * 100).toFixed(0)}% target, trend ${params.trendPct >= 0 ? "+" : ""}${params.trendPct.toFixed(2)}%`,
    currentPct, targetPct,
  };
}

export interface InferenceTarget {
  baseUrl: string;
  apiKey: string;
  model: string;
}

/**
 * Pick an inference provider for the trade decision.
 *
 * Groq is preferred when a key is present: its free tier covers this task many
 * times over (one small call an hour) and it is OpenAI-compatible, so only the
 * base URL and model differ. OpenAI remains the fallback. Explicit env vars
 * override both, which is how any other compatible endpoint (NVIDIA NIM,
 * Together, a local Ollama at :11434/v1) gets used without a code change.
 *
 * Returns null when nothing is configured -- the caller then holds rather than
 * trading on no decision.
 */
export function resolveInferenceTarget(config: {
  openaiApiKey?: string;
  groqApiKey?: string;
  modelStrategy?: { lowComputeModel?: string };
  inferenceModel?: string;
}): InferenceTarget | null {
  const envUrl = process.env.AUTOMATON_TRADE_BASE_URL;
  const envKey = process.env.AUTOMATON_TRADE_API_KEY;
  if (envUrl && envKey) {
    return {
      baseUrl: envUrl.replace(/\/$/, ""),
      apiKey: envKey,
      model: process.env.AUTOMATON_TRADE_MODEL || "llama-3.3-70b-versatile",
    };
  }

  const groqKey = config.groqApiKey || process.env.GROQ_API_KEY;
  if (groqKey) {
    return {
      baseUrl: "https://api.groq.com/openai/v1",
      apiKey: groqKey,
      model: process.env.AUTOMATON_TRADE_MODEL || "llama-3.3-70b-versatile",
    };
  }

  const openaiKey = config.openaiApiKey || process.env.OPENAI_API_KEY;
  if (openaiKey) {
    return {
      baseUrl: "https://api.openai.com/v1",
      apiKey: openaiKey,
      model:
        process.env.AUTOMATON_TRADE_MODEL ||
        config.modelStrategy?.lowComputeModel ||
        config.inferenceModel ||
        "gpt-5-mini",
    };
  }

  return null;
}

/** gpt-5-class models reject `max_tokens` and spend the budget on reasoning
 *  tokens first; llama models on Groq want `max_tokens`. Send the right one. */
function tokenLimitField(model: string): string {
  return /^(gpt-5|o[1-9])/i.test(model) ? "max_completion_tokens" : "max_tokens";
}

/**
 * Ask the model for a direction. Returns a hold on any failure -- an
 * unavailable or malformed answer must never become a trade.
 */
export async function decideDirection(params: {
  baseUrl?: string;
  apiKey: string;
  model: string;
  cash: number;
  positionValue: number;
  price: number | null;
  trendPct: number | null;
  trendLookback: number;
  targetPct?: number;
}): Promise<TradeDecision> {
  const hold = (reason: string): TradeDecision => ({ action: "hold", notional: 0, reason });

  const prompt = [
    "You are deciding a single crypto trading action for a PAPER account trading BTC only.",
    "",
    `Cash available: $${params.cash.toFixed(2)}`,
    `Current BTC position value: $${params.positionValue.toFixed(2)}`,
    (() => {
      const eq = params.cash + params.positionValue;
      if (eq <= 0) return "Current allocation: unknown";
      return `Current allocation: ${((params.positionValue / eq) * 100).toFixed(1)}% BTC / ${((params.cash / eq) * 100).toFixed(1)}% cash (equity $${eq.toFixed(2)})`;
    })(),
    params.targetPct !== undefined
      ? `TARGET allocation: ${(params.targetPct * 100).toFixed(0)}% BTC`
      : "",
    params.price !== null ? `Latest BTC price: $${params.price.toFixed(2)}` : "Latest BTC price: unknown",
    params.trendPct !== null
      ? `${params.trendLookback}-day trend: ${params.trendPct >= 0 ? "+" : ""}${params.trendPct.toFixed(2)}%`
      : `${params.trendLookback}-day trend: unknown`,
    "",
    "The strategy is a pre-committed long-only trend rule, chosen because it was the only",
    "hypothesis that survived out-of-sample testing on this data. It is BINARY -- fully in",
    "or fully out. Idle cash in an uptrend is the strategy failing to be applied.",
    "",
    "Move the allocation toward the TARGET above:",
    "  - Below target -> BUY the gap (one order, capped at 100).",
    "  - Above target -> SELL toward it.",
    "  - Already there -> HOLD.",
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
    const baseUrl = (params.baseUrl || "https://api.openai.com/v1").replace(/\/$/, "");
    const body: Record<string, unknown> = {
      model: params.model,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
    };
    // gpt-5-class models spend this budget on reasoning tokens FIRST, so too
    // small a value returns empty content and no answer at all. Groq's llama
    // models do not reason and take the classic field name.
    body[tokenLimitField(params.model)] = 3000;

    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${params.apiKey}`,
      },
      body: JSON.stringify(body),
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

  const target = resolveInferenceTarget(taskCtx.config as any);
  if (!target) {
    logger.warn("no inference provider configured for trade decision; holding");
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
    // only ever answer "hold", which is not a strategy. Cached: daily bars.
    if (_trendCache && Date.now() - _trendCache.at < TREND_CACHE_MS) {
      trendPct = _trendCache.value;
    } else try {
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
      _trendCache = { at: Date.now(), value: trendPct };
    } catch {
      // Unknown trend -> the model is told so, and will hold.
    }
  } catch (err: any) {
    logger.error(`state read failed, holding: ${err?.message || String(err)}`);
    return { shouldWake: false };
  }

  // 2. Is there anything to do? Free arithmetic, so this can run often. The
  //    model is only woken when the position actually diverges from the rule.
  const opportunity = findOpportunity({ cash, positionValue, trendPct });
  if (!opportunity.act) {
    // Logged at info, not debug, on purpose. A quiet log is how this system's
    // six-day halt hid in plain sight: "working and correctly idle" and
    // "wedged" look identical when neither prints anything. Say so out loud.
    logger.info(`no action: ${opportunity.reason}`);
    taskCtx.db.setKV("last_trade_tick", JSON.stringify({
      at: new Date().toISOString(), outcome: "NO-OP", reason: opportunity.reason,
    }));
    return { shouldWake: false };
  }
  const sinceAction = Date.now() - _lastActionAt;
  if (_lastActionAt && sinceAction < ACTION_COOLDOWN_MS) {
    logger.info(
      `opportunity held: ${opportunity.reason} -- ${Math.ceil((ACTION_COOLDOWN_MS - sinceAction) / 1000)}s cooldown`,
    );
    return { shouldWake: false };
  }
  logger.info(`opportunity: ${opportunity.reason}`);

  // 3. The single judgement call -- reached only when it is worth making.
  const decision = await decideDirection({
    baseUrl: target.baseUrl, apiKey: target.apiKey, model: target.model,
    cash, positionValue, price,
    trendPct, trendLookback: TREND_LOOKBACK_DAYS,
    targetPct: opportunity.targetPct,
  });

  // 4. Deterministic execution, with the balance checks the model cannot override.
  let outcome: string;
  if (decision.action === "hold") {
    outcome = "HOLD";
  } else if (decision.action === "buy" && cash < decision.notional) {
    outcome = `BLOCKED: wanted to buy $${decision.notional.toFixed(2)} but only $${cash.toFixed(2)} cash`;
  } else if (decision.action === "sell" && positionValue < decision.notional) {
    outcome = `BLOCKED: wanted to sell $${decision.notional.toFixed(2)} but position is only $${positionValue.toFixed(2)}`;
  } else {
    try {
      _lastActionAt = Date.now();   // reset the clock on a real action
      const result = await runScript(decision.action, decision.notional.toFixed(2));
      const order = JSON.parse(result);
      outcome = order?.id
        ? `${decision.action.toUpperCase()} $${decision.notional.toFixed(2)} -> order ${order.id} (${order.status})`
        : `${decision.action.toUpperCase()} $${decision.notional.toFixed(2)} -> unexpected reply: ${result.slice(0, 200)}`;
    } catch (err: any) {
      outcome = `ORDER FAILED: ${err?.message || String(err)}`;
    }
  }

  // 5. Journal every acted-on tick -- a decision not written down did not happen.
  const entry = [
    `## ${new Date().toISOString()}`,
    `- cash: $${cash.toFixed(2)} | BTC position: $${positionValue.toFixed(2)}` +
      (price !== null ? ` | price: $${price.toFixed(2)}` : "") +
      (trendPct !== null ? ` | ${TREND_LOOKBACK_DAYS}d trend: ${trendPct >= 0 ? "+" : ""}${trendPct.toFixed(2)}%` : ""),
    `- model (${target.model} @ ${new URL(target.baseUrl).host}) said: ${decision.action}${decision.notional ? ` $${decision.notional.toFixed(2)}` : ""}`,
    `- reason: ${decision.reason}`,
    `- outcome: ${outcome}`,
    "",
  ].join("\n");
  try {
    fs.appendFileSync(journalPath(), entry + "\n");
  } catch (err: any) {
    logger.warn(`could not write journal: ${err?.message || String(err)}`);
  }

  logger.info(`trade tick [${target.model}]: ${outcome} (${decision.reason})`);
  taskCtx.db.setKV("last_trade_tick", JSON.stringify({ at: new Date().toISOString(), outcome }));

  return { shouldWake: false, message: outcome.startsWith("HOLD") ? undefined : `Trade: ${outcome}` };
};
