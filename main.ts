// gta6-trophy-rarity-forecast-01 — v1 forecaster (Node 22, standard library only).
//
// For every GTA VI trophy it forecasts the Exophase earn rate (percent of
// tracked players holding it) at the case's target, as five quantiles.
//
// Model (per DESIGN-v1.md). A rate is earners over tracked players,
// rate = 100 * E / P, so on the log scale ln(rate) = ln(E) - ln(P).  For
// each trophy we fit ln(E) against x = ln(1 + hours since
// 2026-11-19T00:00:00Z) by least squares over the reads of the last 24h,
// and fit ln(P) the same way once for the whole case.  Forecasting E and
// P separately (instead of the rate directly) makes the dilution effect
// explicit: when tracked players grow faster than earners, the forecast
// rate falls, even while the earner count rises.  For small rates this is
// the same as the logit-linear trend fit on the rate itself.
//
//   - Earner-growth slopes are shrunk hierarchically toward the mean slope
//     of their group (tier x current-rate band): rare trophies have noisy
//     slopes and must not be trusted in full.  Weight n / (n + 2).
//   - Fewer than 3 usable reads for a trophy -> persistence (last rate).
//   - Extrapolation is capped: |logit change| <= 3.5, earners never exceed
//     tracked players, platinum never above 10%.
//   - Spread on the logit scale: sigma = sqrt(0.25 * days + resid^2 +
//     flat^2), i.e. the house random walk (0.5 per sqrt-day) plus the fit
//     residual, plus a 1.0-logit floor for trophies whose history is
//     perfectly flat: a recorder refresh can shift every rate at once
//     (launch night: tracked players 638 -> 3868 at one refresh, most
//     rates down ~6x), so a flat history is no evidence of a narrow
//     future.  Quantiles use the standard normal at the case's levels.
//
// Deterministic: inputs are read in name order, rows follow the case's
// trophy order, output is one compact JSON line per case.  No network,
// no clock, no randomness.  Reads $MAKEFUN_INPUT_DIR else /input, writes
// $MAKEFUN_OUTPUT_DIR else /output; one <id>.out per <id>.in.  Never
// deletes or touches the output directory itself.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Launch-night origin for the log-time axis (fixed by the design, not
// by the case: the first reads are just before it).
const T0_MS = Date.parse('2026-11-19T00:00:00Z');
const WINDOW_MS = 24 * 60 * 60 * 1000;
const HALF_EARNER = 0.5; // pseudo-count so a 0% rate still has a log
const MAX_LOGIT_MOVE = 3.5; // cap on |logit(forecast) - logit(last rate)|
const PLATINUM_CAP = 10; // platinum stays near 0 even a week after launch
const FLAT_SPREAD = 1.0; // logit sigma floor for a perfectly flat history
const BASE_SIGMA_PER_SQRT_DAY = 0.5; // house random-walk spread

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at?: string;
  slot?: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  target?: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const readMs = (r: Read): number => Date.parse(r.at ?? r.slot ?? '');

// Inverse standard normal CDF (Acklam's rational approximation,
// accurate to ~1e-9; pure arithmetic, so deterministic).
function normQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= 1 - pLow) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

interface Fit {
  n: number;
  slope: number;
  intercept: number;
  resid: number; // residual std dev of the fit
}

// Ordinary least squares of y on x.  Returns null when too few points
// or x has no spread (then the caller falls back to persistence).
function ols(xs: number[], ys: number[]): Fit | null {
  const n = xs.length;
  if (n < 3) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx < 1e-12) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    ss += e * e;
  }
  const resid = Math.sqrt(ss / Math.max(1, n - 2));
  return { n, slope, intercept, resid };
}

function rateBand(lastRate: number): string {
  if (lastRate < 1) return 'rare';
  if (lastRate < 5) return 'low';
  if (lastRate < 20) return 'mid';
  return 'common';
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  // Checkpoint / target timestamps, with fallbacks so one unparsable
  // timestamp can never poison the whole case with NaN.
  const readTimes = c.history.map((r) => readMs(r)).filter((t) => Number.isFinite(t));
  const lastReadMs = readTimes.length > 0 ? readTimes[readTimes.length - 1] : T0_MS;
  const parsedAt = Date.parse(c.at);
  const atMs = Number.isFinite(parsedAt) ? parsedAt : lastReadMs;
  const parsedTarget = c.target !== undefined ? Date.parse(c.target) : NaN;
  const targetMs = Number.isFinite(parsedTarget)
    ? parsedTarget
    : atMs + c.horizon_minutes * 60_000;
  const xTarget = Math.log(1 + Math.max(0, (targetMs - T0_MS) / 3_600_000));

  // Window: reads from the last 24h before the checkpoint; the whole
  // history when that leaves too few reads.
  let window = c.history.filter((r) => {
    const t = readMs(r);
    return Number.isFinite(t) && t >= atMs - WINDOW_MS && t <= atMs;
  });
  if (window.length < 3) window = c.history.slice();
  // Log-time axis; reads without a parsable timestamp fall back to
  // their position in the window (half an hour apart, the recorder's
  // usual cadence) so x is always finite.
  const xs = window.map((r, j) => {
    const t = readMs(r);
    if (Number.isFinite(t)) return Math.log(1 + Math.max(0, (t - T0_MS) / 3_600_000));
    return Math.log(1 + j * 0.5);
  });

  // One shared fit for tracked players: ln(P) on x.
  const pFit = ols(
    xs,
    window.map((r) => Math.log(Math.max(1, r.players_tracked))),
  );
  const pLast = window.length > 0
    ? window[window.length - 1].players_tracked
    : 1;
  let pStar = pLast;
  if (pFit !== null) {
    pStar = Math.exp(pFit.intercept + pFit.slope * xTarget);
    // Tracked players do not collapse or explode without evidence.
    pStar = Math.min(Math.max(pStar, Math.max(1, pLast * 0.5)), pLast * 50);
  }

  // First pass: per-trophy earner fits and group slope means.
  interface TrophyFit {
    lastRate: number;
    fit: Fit | null;
    flat: boolean;
  }
  const fits: TrophyFit[] = c.trophies.map((t, i) => {
    const ex: number[] = [];
    const ey: number[] = [];
    let lastRate: number | null = null;
    let flat = true;
    let prev: number | null = null;
    for (let j = 0; j < window.length; j++) {
      const rate = window[j].rates[i];
      if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
      lastRate = rate;
      if (prev !== null && Math.abs(rate - prev) > 1e-9) flat = false;
      prev = rate;
      const earners = Math.max(HALF_EARNER, (rate / 100) * window[j].players_tracked);
      ex.push(xs[j]);
      ey.push(Math.log(earners));
    }
    if (lastRate === null) lastRate = 1; // no read ever listed it
    return { lastRate, fit: ols(ex, ey), flat };
  });

  const groupSum = new Map<string, { sum: number; n: number }>();
  fits.forEach((f, i) => {
    if (f.fit === null) return;
    const key = `${c.trophies[i].tier}|${rateBand(f.lastRate)}`;
    const g = groupSum.get(key) ?? { sum: 0, n: 0 };
    g.sum += f.fit.slope;
    g.n += 1;
    groupSum.set(key, g);
  });

  const days = c.horizon_minutes / 1440;
  const z = c.levels.map((tau) => normQuantile(Math.min(0.999, Math.max(0.001, tau))));

  const trophies = c.trophies.map((t, i) => {
    const f = fits[i];
    const lastRate = f.lastRate;
    let median = lastRate;
    let resid = 0.35; // persistence fallback spread component
    if (f.fit !== null) {
      const key = `${t.tier}|${rateBand(lastRate)}`;
      const g = groupSum.get(key);
      const groupSlope = g !== undefined && g.n > 0 ? g.sum / g.n : 0;
      const w = f.fit.n / (f.fit.n + 2); // shrink toward the group mean
      const slope = w * f.fit.slope + (1 - w) * groupSlope;
      let eStar = Math.exp(f.fit.intercept + slope * xTarget);
      eStar = Math.min(Math.max(eStar, HALF_EARNER), pStar); // earners <= tracked
      median = (100 * eStar) / pStar;
      resid = Math.min(1.0, Math.max(0.1, f.fit.resid));
    }
    // Reasonable-extrapolation caps.
    if (!Number.isFinite(median)) median = lastRate; // never emit NaN
    median = clampPct(median);
    if (t.tier === 'platinum') median = Math.min(median, PLATINUM_CAP);
    const lLast = logit(lastRate);
    const lMed = Math.min(lLast + MAX_LOGIT_MOVE, Math.max(lLast - MAX_LOGIT_MOVE, logit(median)));
    median = sigmoidPct(lMed);
    let sigma = Math.sqrt(
      BASE_SIGMA_PER_SQRT_DAY * BASE_SIGMA_PER_SQRT_DAY * days +
      resid * resid +
      (f.flat ? FLAT_SPREAD * FLAT_SPREAD : 0),
    );
    if (!Number.isFinite(sigma) || sigma <= 0) {
      sigma = BASE_SIGMA_PER_SQRT_DAY * Math.sqrt(days) + 0.35;
    }
    const base = logit(median);
    const q = z.map((zv) => round4(clampPct(sigmoidPct(base + zv * sigma))));
    // Rounding can only tie, never invert, but enforce it explicitly.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });

  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.levels) || c.levels.length !== 5) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, expected 5 levels`);
  }
  if (!Array.isArray(c.history) || c.history.length === 0) {
    throw new Error(`${file} has no history`);
  }
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
