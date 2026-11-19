// gta6-trophy-rarity-forecast-01 · hermansto · v2
// Node 22+ standard library only. Offline, deterministic, no tuning on case data.
//
// Why rates move: Exophase earn rate = trophy earners / tracked players. After launch the
// tracked base balloons, so every rate is diluted by denominator growth — but not equally.
// A trophy's earners scale with the base raised to an elasticity beta in [0.2, 1]:
//
//     beta   = 0.2 + 0.8 * (last observed rate / 100)
//     median = last rate * growth^(beta - 1)          (clamped to 0.01 .. 99.99)
//
// Story trophies nearly everyone earns (rate ~ 100, beta ~ 1) barely dilute; obscure
// collectibles (rate ~ 0, beta ~ 0.2) dilute almost one-for-one with the base.
//
// growth comes from a log-linear extrapolation of players_tracked: least squares of
// log(players) on log(1 + hours since the first read), evaluated at the target time,
// divided by the last read's count, capped to [1, 50]. A flat/shrinking base means 1.
//
// Uncertainty: random-walk diffusion on the logit scale, sigma = 0.6 per sqrt(day)
// (the house's 0.5 proved too tight across launch-night refreshes):
//
//     q_tau = sigmoid(logit(median) + z_tau * 0.6 * sqrt(horizon_minutes / 1440))
//
// with z = (-1.645, -0.674, 0, 0.674, 1.645) for tau = (.05, .25, .5, .75, .95),
// each quantile rounded to 4 decimals.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const EXPECTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const NORMAL_Z = [-1.645, -0.674, 0, 0.674, 1.645];
const SIGMA_PER_SQRT_DAY = 0.6;
const MAX_DENOMINATOR_GROWTH = 50;
const FLOOR = 0.01;
const CEIL = 99.99;
const HOUR_MS = 3_600_000;

// ---- percent <-> logit plumbing (clamp exactly like the scorer) ----

function clampPct(x: number): number {
  return Math.min(CEIL, Math.max(FLOOR, x));
}

function logitOf(pct: number): number {
  const c = clampPct(pct);
  return Math.log(c / (100 - c));
}

function pctOf(logit: number): number {
  return 100 / (1 + Math.exp(-logit));
}

function to4(x: number): number {
  return Math.round(x * 10_000) / 10_000;
}

// ---- least squares: y = a + b·x, with flat-line fallbacks for degenerate samples ----

function leastSquares(x: number[], y: number[]): [number, number] {
  const n = x.length;
  if (n < 2) return [y[0], 0];
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  const det = n * sxx - sx * sx;
  if (Math.abs(det) < 1e-12) return [sy / n, 0];
  const b = (n * sxy - sx * sy) / det;
  const a = (sy - b * sx) / n;
  return [a, b];
}

// ---- case schema (only the fields the model touches) ----

type Read = { at: string; players_tracked: number; rates: Array<number | null> };
type Case = {
  target: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Array<{ id: number }>;
  history: Read[];
};

/**
 * Denominator growth factor between the newest read and the target.
 * Uses every read with a positive player count; time is measured in hours from the
 * very first read (even one with zero players anchors the clock). Only an actually
 * growing base triggers extrapolation — otherwise the factor is exactly 1.
 */
function growthFactor(history: Read[], target: string, newestCount: number): number {
  const clockStart = Date.parse(history[0].at) / HOUR_MS;
  const samples: Array<{ t: number; p: number }> = [];
  for (const read of history) {
    if (read.players_tracked > 0) {
      samples.push({ t: Date.parse(read.at) / HOUR_MS - clockStart, p: read.players_tracked });
    }
  }
  if (samples.length < 2) return 1;
  if (samples[samples.length - 1].p <= samples[0].p) return 1;

  const logT: number[] = [];
  const logP: number[] = [];
  for (const s of samples) {
    logT.push(Math.log(s.t + 1));
    logP.push(Math.log(s.p));
  }
  const [a, b] = leastSquares(logT, logP);
  const tTarget = Date.parse(target) / HOUR_MS - clockStart;
  const projected = Math.exp(a + b * Math.log(tTarget + 1));
  return Math.min(MAX_DENOMINATOR_GROWTH, Math.max(1, projected / newestCount));
}

/** Diluted median rate for one trophy under denominator growth. */
function medianRate(observed: number | null, growth: number): number {
  // Guard: a null (never-observed) rate is treated as the 0.01 floor; so is any rate <= 0.
  const rate = observed == null ? FLOOR : observed;
  if (rate <= 0) return FLOOR;
  const r = clampPct(rate);
  const elasticity = 0.2 + 0.8 * (r / 100);
  return clampPct(r * Math.pow(growth, elasticity - 1));
}

// ---- per-case pipeline ----

function processCase(fileName: string): void {
  const caseId = fileName.slice(0, -3); // strip ".in"
  const c: Case = JSON.parse(readFileSync(join(IN_DIR, fileName), 'utf8'));

  if (c.levels.length !== EXPECTED_LEVELS.length || c.levels.some((v, i) => v !== EXPECTED_LEVELS[i])) {
    throw new Error(`${fileName}: levels ${JSON.stringify(c.levels)} are not the contest levels`);
  }
  // Guard: no history means nothing to forecast from — fail loudly rather than guess.
  if (c.history.length === 0) throw new Error(`${fileName}: history is empty`);
  const newest = c.history[c.history.length - 1];

  const growth = growthFactor(c.history, c.target, newest.players_tracked);
  const days = c.horizon_minutes / 1440;
  const spread = SIGMA_PER_SQRT_DAY * Math.sqrt(days);

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const med = medianRate(newest.rates[i], growth);
    const anchor = logitOf(med);
    const q: number[] = [];
    for (const z of NORMAL_Z) {
      q.push(to4(pctOf(anchor + z * spread)));
    }
    rows.push({ id: c.trophies[i].id, q });
  }

  writeFileSync(join(OUT_DIR, `${caseId}.out`), JSON.stringify({ trophies: rows }) + '\n');
}

// ---- entry point: every case file, name order, fixed formulas => reproducible bytes ----

mkdirSync(OUT_DIR, { recursive: true });
const files = readdirSync(IN_DIR).filter((f) => f.endsWith('.in')).sort();
for (const f of files) processCase(f);