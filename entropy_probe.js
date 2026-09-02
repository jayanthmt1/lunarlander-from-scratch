'use strict';
/* =============================================================================
 * entropy_probe.js — does the entropy bonus pin the agent in the hover plateau?
 *
 * HYPOTHESIS: beta = max(entropyMin, entropy0 * 0.5^(ep/entropyHalfLife)) keeps
 * the policy too stochastic to execute a touchdown.  Escape should therefore
 * happen shortly after beta reaches its floor, and should MOVE when the
 * schedule moves.
 *
 * DESIGN: vary entropyHalfLife / entropyMin, everything else at core defaults.
 * ESCAPE EPISODE = first episode where the landing rate over the trailing 500
 * episodes exceeds 50%.  40000 episodes, 3 seeds per config.
 *
 * Results are appended to RESULT_FILE as JSONL the moment each run finishes
 * (and the moment an escape is detected), so a killed / timed-out sweep loses
 * nothing and re-running resumes where it left off.
 *
 *   node entropy_probe.js            run/resume the sweep, then print tables
 *   node entropy_probe.js --report   just print the tables from the JSONL
 * ========================================================================== */
const path = require('path');
const fs = require('fs');
const { fork } = require('child_process');

const EPISODES = 40000;
const WINDOW = 500;          // trailing window for landing rate
const THRESH = 0.50;         // >50% landed = escaped
const SEEDS = [1, 2, 3];
const ENTROPY0 = 0.02;       // held fixed (core default)
const MAX_PARALLEL = 12;

const RESULT_FILE = process.env.PROBE_OUT ||
  path.join(__dirname, 'entropy_probe_results.jsonl');

// episode at which the exponential term hits the entropyMin floor
function betaFloorEp(hl, min, e0) {
  if (e0 === undefined) e0 = ENTROPY0;
  if (e0 <= 0) return 0;                 // beta identically 0 from episode 0
  if (min <= 0) return Infinity;
  if (min >= e0) return 0;
  return hl * Math.log2(e0 / min);
}

const CONFIGS = [
  { name: 'HL=500  min=0.001',  entropyHalfLife: 500,  entropyMin: 0.001 },
  { name: 'HL=2000 min=0.001*', entropyHalfLife: 2000, entropyMin: 0.001 },  // default
  { name: 'HL=8000 min=0.001',  entropyHalfLife: 8000, entropyMin: 0.001 },
  { name: 'HL=2000 min=0.02',   entropyHalfLife: 2000, entropyMin: 0.02 },   // beta never decays
  // --- second batch: 2 seeds each -----------------------------------------
  { name: 'beta=0 (no bonus)',  entropyHalfLife: 2000, entropyMin: 0, entropy0: 0, seeds: [1, 2] },
  { name: 'HL=2000 min=0',      entropyHalfLife: 2000, entropyMin: 0, seeds: [1, 2] },
  { name: 'HL=2000 min=0.005',  entropyHalfLife: 2000, entropyMin: 0.005, seeds: [1, 2] },
  { name: 'HL=1000 min=0.001',  entropyHalfLife: 1000, entropyMin: 0.001, seeds: [1, 2] },
  { name: 'HL=4000 min=0.001',  entropyHalfLife: 4000, entropyMin: 0.001, seeds: [1, 2] },
];
// how many of CONFIGS to run this invocation (front of the list first)
const N_CONFIGS = Number(process.env.PROBE_CONFIGS || 4);

/* ---------------------------------------------------------------- worker --- */
function append(rec) { fs.appendFileSync(RESULT_FILE, JSON.stringify(rec) + '\n'); }

function runOne(job) {
  const C = require(path.join(__dirname, 'core.js'));
  const cfg = {
    seed: job.seed,
    entropyHalfLife: job.entropyHalfLife,
    entropyMin: job.entropyMin,
  };
  if (job.entropy0 !== undefined) cfg.entropy0 = job.entropy0;
  const t = new C.Trainer(cfg);
  const ring = new Uint8Array(WINDOW);
  let filled = 0, landed = 0, ptr = 0, escape = -1;
  const trace = [];                       // trailing-500 land% every 2000 eps
  let nextMark = 2000;
  const t0 = Date.now();

  t.onEpisode = () => {
    const ep = t.episode;                 // already incremented
    const bit = t.ghost.outcome[(ep - 1) % C.CURVE_CAP] === 1 ? 1 : 0;
    if (filled === WINDOW) landed -= ring[ptr]; else filled++;
    ring[ptr] = bit; landed += bit; ptr = (ptr + 1) % WINDOW;
    if (escape < 0 && filled === WINDOW && landed / WINDOW > THRESH) {
      escape = ep;
      append({ type: 'escape', name: job.name, seed: job.seed, escape });
    }
    if (ep >= nextMark) { trace.push(Math.round(100 * landed / filled)); nextMark += 2000; }
  };

  while (t.episode < EPISODES) t.tick(1000, 200000);

  let L = 0, M = 0, n = 0;
  for (let i = Math.max(0, t.ghost.n - 2000); i < t.ghost.n; i++) {
    if (t.ghost.outcome[i % C.CURVE_CAP] === 1) L++;
    M += t.ghost.ret[i % C.CURVE_CAP]; n++;
  }
  const rec = {
    type: 'final', name: job.name, seed: job.seed,
    entropyHalfLife: job.entropyHalfLife, entropyMin: job.entropyMin,
    entropy0: job.entropy0 === undefined ? ENTROPY0 : job.entropy0,
    escape, finalLand: +(100 * L / n).toFixed(1), finalMean: +(M / n).toFixed(1),
    trace, secs: +((Date.now() - t0) / 1000).toFixed(0),
  };
  append(rec);
  return rec;
}

if (process.argv[2] === '--worker') {
  runOne(JSON.parse(process.argv[3]));
  process.exit(0);
}

/* ---------------------------------------------------------------- master --- */
function loadRecords() {
  if (!fs.existsSync(RESULT_FILE)) return [];
  return fs.readFileSync(RESULT_FILE, 'utf8').split('\n')
    .filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } })
    .filter(Boolean);
}

function report() {
  const recs = loadRecords();
  const finals = recs.filter(r => r.type === 'final');
  const escapes = recs.filter(r => r.type === 'escape');
  const pad = (s, n) => String(s).padEnd(n);
  const lp = (s, n) => String(s).padStart(n);

  console.log('\n=== ENTROPY SCHEDULE PROBE ===');
  console.log(`episodes=${EPISODES}  escape = first ep with trailing-${WINDOW} landing rate > ` +
    `${THRESH * 100}%  seeds=${SEEDS.join(',')}  entropy0=${ENTROPY0} (all else default)\n`);
  console.log(pad('config', 20) + lp('betaFloorEp', 12) +
    SEEDS.map(s => lp('esc.s' + s, 9)).join('') +
    lp('median', 9) + lp('finalLand%', 12) + lp('finalMean', 11) + lp('n', 4));
  console.log('-'.repeat(20 + 12 + 9 * SEEDS.length + 9 + 12 + 11 + 4));

  for (const c of CONFIGS) {
    const rs = SEEDS.map(s => finals.find(r => r.name === c.name && r.seed === s));
    if (!rs.some(Boolean)) continue;
    const escs = SEEDS.map((s, i) => {
      const r = rs[i];
      if (r) return r.escape >= 0 ? r.escape : null;
      const e = escapes.find(x => x.name === c.name && x.seed === s);
      return e ? e.escape : undefined;              // undefined = run not finished
    });
    const ok = escs.filter(e => typeof e === 'number').sort((a, b) => a - b);
    const med = ok.length ? (ok.length % 2 ? ok[(ok.length - 1) / 2]
      : Math.round((ok[ok.length / 2 - 1] + ok[ok.length / 2]) / 2)) : null;
    const got = rs.filter(Boolean);
    const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
    const bf = betaFloorEp(c.entropyHalfLife, c.entropyMin, c.entropy0);
    console.log(pad(c.name, 20) + lp(bf === Infinity ? 'never' : Math.round(bf), 12) +
      escs.map(e => lp(e === undefined ? '?' : e === null ? 'NONE' : e, 9)).join('') +
      lp(med === null ? '-' : med, 9) +
      lp(got.length ? avg(got.map(r => r.finalLand)).toFixed(0) : '-', 12) +
      lp(got.length ? avg(got.map(r => r.finalMean)).toFixed(0) : '-', 11) +
      lp(got.length, 4));
  }

  const marks = [];
  for (let k = 2000; k <= EPISODES; k += 4000) marks.push(k);
  console.log('\n--- landing % (trailing 500) sampled every 4000 episodes ---');
  console.log(pad('config/seed', 21) + marks.map(k => lp(k / 1000 + 'k', 6)).join(''));
  for (const c of CONFIGS) for (const s of SEEDS) {
    const r = finals.find(x => x.name === c.name && x.seed === s);
    if (!r) continue;
    const row = r.trace.filter((_, i) => i % 2 === 0);
    console.log(pad(c.name + ' s' + s, 21) + row.map(v => lp(v + '%', 6)).join(''));
  }
}

if (process.argv[2] === '--report') { report(); process.exit(0); }

const doneKeys = new Set(loadRecords().filter(r => r.type === 'final')
  .map(r => r.name + '|' + r.seed));
const jobs = [];
for (const c of CONFIGS.slice(0, N_CONFIGS))
  for (const seed of (c.seeds || SEEDS))
    if (!doneKeys.has(c.name + '|' + seed)) jobs.push({ ...c, seed });

console.error(`${doneKeys.size} runs already complete; launching ${jobs.length} ` +
  `(${EPISODES} episodes each), up to ${MAX_PARALLEL} at a time`);

let next = 0, done = 0;
const T0 = Date.now();
function launch() {
  if (next >= jobs.length) return;
  const job = jobs[next++];
  const ch = fork(__filename, ['--worker', JSON.stringify(job)]);
  ch.on('exit', () => {
    done++;
    const r = loadRecords().find(x => x.type === 'final' && x.name === job.name && x.seed === job.seed);
    console.error(`[${done}/${jobs.length}] ${job.name} s${job.seed} ` +
      (r ? `escape=${r.escape < 0 ? 'NONE' : r.escape} final=${r.finalLand}%/${r.finalMean} (${r.secs}s)`
         : 'NO RESULT') + `  [${((Date.now() - T0) / 1000).toFixed(0)}s elapsed]`);
    if (next < jobs.length) launch();
    else if (done === jobs.length) report();
  });
}
if (!jobs.length) report();
for (let i = 0; i < Math.min(MAX_PARALLEL, jobs.length); i++) launch();
