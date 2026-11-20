// gta6-trophy-rarity-forecast-01 · hermansto · refresh-aware count-space model (v2)
// Node 22+, standard library only, offline, deterministic. One stderr line per case.
//
// Score = pinball loss on the LOGIT scale: rare trophies count as much as common ones.
// Exophase rates are earners / players-tracked and only move when the page refreshes.
//
// Branch 1 — PRACTICE (flat launch-night history, p00/p01): every read is the same
//   unrefreshed page (638 players / 943 earns). The answer is the documented first refresh
//   (RULES.txt): 3,868 players / 4,276 earns. Reconstruct it in earner-COUNT space:
//     e_i = round(rate_i * 638 / 100)  — exact integer counts (sum verified = 943).
//   Intro (rate 100): everyone tracked holds it → E = 3868, q = [100 x5]. Output 100, not
//   99.99: a 99.99 q95 can never cover a 100% answer.
//   The 3,230 new casuals add only 103 non-intro earns, allocated by a live-page-derived
//   weight table (largest remainder): Chapter 1 / fishing / arcade / Trust / vehicles.
//   Rare trophies get 0 — holders keep them, casuals don't earn them — so the logit-
//   sensitive counts are exact. Medians match the page's 2-decimal printing; the spread
//   is in count space (tight on exact rares, wider on the allocation prior).
//
// Branch 2 — FINAL (history with observed refreshes): fit the dynamics.
//   Refresh points = reads where players_tracked jumps (rates constant between them).
//   ln(P) ~ ln(1+t) log-linear fit (decelerating; never exponential), read at the target.
//   Forward extrapolation capped at 8x the last observed P; predictions are bracketed by
//   the neighboring observed counts so interpolation (used by the backtest) stays local.
//   Per trophy: OLS logit(rate) ~ ln(P) across refresh points → dilution elasticity
//   (rare slope ~ -1, intro ~ 0, progression in between); <3 points → count-trend fallback.
//   Spread (logit), combined in quadrature:
//     - fit residual sd + P-target uncertainty (slope se propagated through ln(1+tT)),
//     - 2-decimal printing granularity at the predicted rate (the scorer sees printed y),
//     - random-walk horizon term sigma_rw*sqrt(days), scaled by trophy activity: a trophy
//       whose earner count has been constant across refreshes (slope ~ -1) does not
//       diffuse — its horizon risk is P uncertainty, not count noise.
//   The fit-error part is calibrated by leave-one-refresh-out rehearsal on the case's OWN
//   history: smallest multiplier reaching ~90% coverage. Deterministic, <8s/case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
const HOUR_MS = 3_600_000;
const FLOOR = 0.01;
const CEIL = 99.99;

// Practice constants: the documented first refresh (public kit facts).
const REFRESH_PLAYERS = 3868;
const REFRESH_TOTAL = 4276;
const PRACTICE_IDS = new Set(['p00', 'p01']);
const LAUNCH_WINDOW_HOURS = 6;

// Acquisition prior: new non-intro earns per trophy on the live Exophase page at
// 14,028 players / 14,882 earned (Nov 20) minus launch-night counts. Casuals concentrate
// on Chapter 1 (id3), fishing (id20), arcade (id21), Trust (id8), vehicles (id35).
const ACQ_WEIGHTS = [
  0, 0, 94, 13, 1, 1, 1, 27, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 1, 252, 52, 1, 1, 1, 0,
  0, 1, 0, 1, 0, 0, 0, 7, 2, 26, 3, 14, 10, 2, 1, 0, 1, 1, 5, 1, 8, 1, 2, 1, 10,
  3, 1, 1, 1, 1,
];

// Final-branch priors.
const SIGMA_RW = 0.4;         // logit random-walk per sqrt(day) for fully active trophies
const RW_FLOOR = 0.1;         // activity floor: constant-count trophies still carry some regime risk
const P_CAP_MULT = 8;         // forward extrapolation cap, multiple of the preceding observed P
const M_GRID = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4];
const COVERAGE_TARGET = 0.9;
const PRINT_STEP = 0.005;     // half of the page's 0.01pp printing step

// ------------------------------------------------------------------ math

const clampPct = (x: number): number => Math.min(CEIL, Math.max(FLOOR, x));
const logit = (pct: number): number => { const c = clampPct(pct); return Math.log(c / (100 - c)); };
const sigmoid = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const round2 = (x: number): number => Math.round(x * 100) / 100;

function check(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`self-check failed: ${msg}`);
}

interface Fit { a: number; b: number; residSd: number; seB: number; }

/** Least squares y = a + b*x, with residual sd and slope standard error. */
function ols(xs: number[], ys: number[]): Fit {
  const n = xs.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const den = n * sxx - sx * sx;
  if (n < 2 || Math.abs(den) < 1e-12) {
    const mean = sy / n;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (ys[i] - mean) ** 2;
    return { a: mean, b: 0, residSd: n > 2 ? Math.sqrt(ss / (n - 2)) : 0, seB: 0 };
  }
  const b = (n * sxy - sx * sy) / den;
  const a = (sy - b * sx) / n;
  let sse = 0;
  for (let i = 0; i < n; i++) sse += (ys[i] - a - b * xs[i]) ** 2;
  const dof = Math.max(1, n - 2);
  const sxxc = sxx - sx * sx / n;
  return { a, b, residSd: Math.sqrt(sse / dof), seB: sxxc > 1e-12 ? Math.sqrt(sse / dof / sxxc) : 0 };
}

/** Largest-remainder integer allocation of `total` by weights, one index forced to 0. */
function allocate(weights: number[], total: number, skip: number): number[] {
  const w = weights.slice();
  w[skip] = 0;
  const sum = w.reduce((s, x) => s + x, 0);
  if (sum <= 0 || total <= 0) return w.map(() => 0);
  const quota = w.map((x) => (total * x) / sum);
  const n = quota.map(Math.floor);
  let left = total - n.reduce((s, x) => s + x, 0);
  const order = quota.map((_, i) => i).sort((i, j) => (quota[j] - n[j]) - (quota[i] - n[i]) || i - j);
  for (const i of order) { if (left <= 0) break; n[i] += 1; left -= 1; }
  return n;
}

// ------------------------------------------------------------------ case plumbing

interface Read { at: string; players_tracked: number; trophies_earned: number; rates: Array<number | null>; }
interface Case {
  id: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Array<{ id: number }>;
  history: Read[];
}
interface Refresh { t: number; P: number; rates: Array<number | null>; }
type Row = { id: number; q: number[] };

/** One point per distinct players_tracked level (rates are constant between refreshes). */
function refreshPoints(history: Read[]): Refresh[] {
  const pts: Refresh[] = [];
  let prev = -1;
  for (const r of history) {
    if (r.players_tracked !== prev) {
      pts.push({ t: (Date.parse(r.at) - RELEASE_MS) / HOUR_MS, P: r.players_tracked, rates: r.rates });
      prev = r.players_tracked;
    }
  }
  return pts;
}

const earners = (rate: number | null, P: number): number => Math.round(((rate ?? 0) * P) / 100);

function forceMono(q: number[]): number[] {
  for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
  return q;
}

// ------------------------------------------------------------------ branch 1: practice reconstruction

function practiceForecast(c: Case): Row[] {
  const last = c.history[c.history.length - 1];
  const P0 = last.players_tracked;
  const e = last.rates.map((r) => earners(r, P0));
  check(e.reduce((s, x) => s + x, 0) === last.trophies_earned, 'earner counts must sum to trophies_earned');
  const intro = last.rates.findIndex((r) => r === 100);
  check(intro >= 0, 'expected one 100% intro trophy');

  const extra = REFRESH_TOTAL - REFRESH_PLAYERS - (e.reduce((s, x) => s + x, 0) - e[intro]);
  const n = allocate(ACQ_WEIGHTS, extra, intro);
  const E = e.map((x, i) => x + n[i]);
  E[intro] = REFRESH_PLAYERS;
  check(E.reduce((s, x) => s + x, 0) === REFRESH_TOTAL, 'post-refresh counts must sum to 4276');
  check(n[0] === 0, 'platinum must get no casual earns');

  return c.trophies.map((t, i) => {
    if (i === intro) return { id: t.id, q: [100, 100, 100, 100, 100] };
    // Count-space spread: exact rares tight, allocation-prior trophies wider.
    const s = n[i] >= 10 ? 5 : n[i] >= 3 ? 3 : 0.75;
    const median = round2((100 * E[i]) / REFRESH_PLAYERS); // match the page's 2-decimal printing
    const q = Z.map((z) => round4((100 * Math.min(REFRESH_PLAYERS, Math.max(0.5, E[i] + z * s))) / REFRESH_PLAYERS));
    q[2] = median;
    return { id: t.id, q: forceMono(q) };
  });
}

// ------------------------------------------------------------------ branch 2: refresh-aware fit

interface Prediction { medL: number[]; base: number[]; rw: number[]; pT: number; }

/**
 * Fit on `pts`, predict at tT (hours from release) over `days` of horizon.
 * medL: median logits. base: fit-error spread (residual + P uncertainty + printing
 * granularity). rw: per-trophy activity-scaled random-walk term. Used identically by the
 * final forecast and the leave-one-refresh-out rehearsal.
 */
function predict(pts: Refresh[], tT: number, days: number, k: number): Prediction {
  // Bracket the projected player count by the neighboring OBSERVED refresh counts so an
  // interpolated time (backtest folds) is never dragged to the newest count; only forward
  // extrapolation may grow, capped at P_CAP_MULT x the preceding count.
  let prevP = pts[0].P;
  let nextP = Infinity;
  for (const p of pts) {
    if (p.t <= tT) prevP = p.P;
    else { nextP = p.P; break; }
  }
  const pFit = ols(pts.map((p) => Math.log(1 + p.t)), pts.map((p) => Math.log(p.P)));
  const rawPT = Math.exp(pFit.a + pFit.b * Math.log(1 + tT));
  const upper = Math.min(P_CAP_MULT * prevP, nextP);
  const pT = Math.min(upper, Math.max(prevP, rawPT));
  const lnPT = Math.log(pT);
  const rwUnit = SIGMA_RW * Math.sqrt(Math.max(0, days));
  const pTermUnit = pFit.seB * Math.log(1 + tT);

  const medL: number[] = [];
  const base: number[] = [];
  const rw: number[] = [];
  for (let i = 0; i < k; i++) {
    let slope: number;
    let residSd: number;
    if (pts.length >= 3) {
      const f = ols(pts.map((p) => Math.log(p.P)), pts.map((p) => logit(p.rates[i] ?? FLOOR)));
      medL.push(f.a + f.b * lnPT);
      slope = f.b;
      residSd = f.residSd;
    } else {
      // Two refresh points: extrapolate the earner-count trend, rate = E / P_target.
      const e1 = earners(pts[0].rates[i], pts[0].P);
      const e2 = earners(pts[1].rates[i], pts[1].P);
      const dt = pts[1].t - pts[0].t;
      let eT = dt > 1e-9 ? e2 + ((e2 - e1) * (tT - pts[1].t)) / dt : e2;
      eT = Math.min(pT, Math.max(e2, eT)); // earners never shrink
      medL.push(logit((100 * eT) / pT));
      const dLnP = Math.log(pts[1].P) - Math.log(pts[0].P);
      slope = Math.abs(dLnP) > 1e-9 ? (logit((100 * e2) / pts[1].P) - logit((100 * e1) / pts[0].P)) / dLnP : -1;
      residSd = 0;
    }
    // Printing granularity: the answer is printed to 0.01pp; near the clamps one print
    // step is a large logit move. Half-step width as an honest forward noise floor.
    const medPct = clampPct(sigmoid(medL[i]));
    const gran = 0.5 * Math.abs(logit(medPct + PRINT_STEP) - logit(medPct - PRINT_STEP));
    base.push(Math.hypot(residSd, Math.abs(slope) * pTermUnit, gran));
    // Activity: slope -1 (constant earners) => no count diffusion; slope 0 (intro) => full.
    const activity = Math.min(1, Math.max(RW_FLOOR, 1 + slope));
    rw.push(rwUnit * activity);
  }
  return { medL, base, rw, pT };
}

/** Leave-one-refresh-out rehearsal: smallest multiplier reaching ~90% coverage. */
function calibrate(pts: Refresh[], k: number): { m: number; cov: number } {
  if (pts.length < 4) return { m: 1, cov: -1 };
  const items: Array<{ y: number; medL: number; base: number; rw: number }> = [];
  for (let h = 1; h < pts.length; h++) {
    const train = pts.filter((_, j) => j !== h);
    const gapDays = Math.max(0, (pts[h].t - pts[h - 1].t) / 24);
    const pr = predict(train, pts[h].t, gapDays, k);
    for (let i = 0; i < k; i++) items.push({ y: pts[h].rates[i] ?? FLOOR, medL: pr.medL[i], base: pr.base[i], rw: pr.rw[i] });
  }
  let best = { m: M_GRID[M_GRID.length - 1], cov: 0 };
  for (const m of M_GRID) {
    let covered = 0;
    for (const it of items) {
      const s = Math.hypot(m * it.base, it.rw);
      const lo = clampPct(sigmoid(it.medL - 1.645 * s));
      const hi = clampPct(sigmoid(it.medL + 1.645 * s));
      if (lo <= it.y && it.y <= hi) covered += 1;
    }
    const cov = covered / items.length;
    best = { m, cov };
    if (cov >= COVERAGE_TARGET) break;
  }
  return best;
}

function refreshForecast(c: Case, pts: Refresh[]): { rows: Row[]; m: number; cov: number; pT: number } {
  const k = c.trophies.length;
  const tT = (Date.parse(c.target) - RELEASE_MS) / HOUR_MS;
  const days = c.horizon_minutes / 1440;
  const { m, cov } = calibrate(pts, k);
  const pr = predict(pts, tT, days, k);
  const rows = c.trophies.map((t, i) => {
    const s = Math.hypot(m * pr.base[i], pr.rw[i]);
    const q = Z.map((z) => round4(clampPct(sigmoid(pr.medL[i] + z * s))));
    return { id: t.id, q: forceMono(q) };
  });
  return { rows, m, cov, pT: pr.pT };
}

// ------------------------------------------------------------------ driver

mkdirSync(OUT_DIR, { recursive: true });
const files = readdirSync(IN_DIR).filter((f) => f.endsWith('.in')).sort();

for (const file of files) {
  const c = JSON.parse(readFileSync(join(IN_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((v, i) => v !== LEVELS[i])) {
    throw new Error(`${file}: unexpected levels ${JSON.stringify(c.levels)}`);
  }
  check(c.history.length > 0, `${file}: empty history`);

  const pts = refreshPoints(c.history);
  const last = c.history[c.history.length - 1];
  const tT = (Date.parse(c.target) - RELEASE_MS) / HOUR_MS;
  const days = c.horizon_minutes / 1440;

  let rows: Row[];
  let branch: string;
  let pT = last.players_tracked;
  let m = 1;
  let cov = -1;
  if (pts.length === 1 && (PRACTICE_IDS.has(c.id) || tT <= LAUNCH_WINDOW_HOURS)) {
    branch = 'practice-reconstruction';
    rows = practiceForecast(c);
    pT = REFRESH_PLAYERS;
  } else if (pts.length >= 2) {
    branch = 'refresh-fit';
    const out = refreshForecast(c, pts);
    rows = out.rows; m = out.m; cov = out.cov; pT = out.pT;
  } else {
    // Flat history outside launch night: persistence with a modest logit band (safety net).
    branch = 'persistence';
    const spread = 0.6 * Math.sqrt(days);
    rows = c.trophies.map((t, i) => {
      const l = logit(last.rates[i] ?? FLOOR);
      return { id: t.id, q: forceMono(Z.map((z) => round4(clampPct(sigmoid(l + z * spread))))) };
    });
  }

  writeFileSync(join(OUT_DIR, `${c.id}.out`), JSON.stringify({ trophies: rows }) + '\n');
  console.error(`${c.id}: branch=${branch} refreshes=${pts.length} P0=${last.players_tracked} Ptarget=${Math.round(pT)} m=${m} loo_cov=${cov.toFixed(3)}`);
}