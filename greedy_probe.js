'use strict';
/* =============================================================================
 * greedy_probe.js — mechanism check for the entropy hypothesis.
 *
 * The hypothesis says: "a stochastic policy physically cannot perform the
 * precise final touchdown, so while beta is meaningful the agent is held in
 * hovering".  That makes a sharp, testable prediction:
 *
 *   during the plateau, the GREEDY (argmax) policy should already be able to
 *   land, and only the sampling noise should be preventing it.
 *
 * If greedy evaluation of the plateau policy ALSO fails to land, the policy
 * simply has not learned touchdown yet and entropy is not the gate.
 *
 * Also logs the policy's own mean entropy (nats) so we can see whether the
 * distribution actually sharpens at escape, and the value net's RMSE against
 * the realised discounted return.
 * ========================================================================== */
const path = require('path');
const { fork } = require('child_process');
const os = require('os');
const C = require(path.join(__dirname, 'core.js'));

const EPISODES = 26000;
const CHECKS = [];
for (let e = 1000; e <= EPISODES; e += 1000) CHECKS.push(e);
const N_EVAL = 40;                 // eval episodes per mode per checkpoint
const SEEDS = [1, 2, 3];

// Evaluate the current policy on fresh episodes, without touching trainer state.
function evaluate(t, greedy, evalSeed) {
  const rng = new C.RNG(evalSeed);
  const env = new C.LunarEnv(rng, t.cfg);
  let landed = 0, crashed = 0, hovered = 0, sumR = 0, sumEnt = 0, nEnt = 0;
  for (let i = 0; i < N_EVAL; i++) {
    let s = env.reset(), R = 0, steps = 0, done = false;
    while (steps < 1000) {
      const p = t.pol.forward(s);
      sumEnt += t.pol.entropy; nEnt++;
      let a;
      if (greedy) { a = 0; for (let j = 1; j < 4; j++) if (p[j] > p[a]) a = j; }
      else a = t.pol.sample(rng);
      const r = env.step(a);
      R += r.rawReward; s = r.obs; steps++;
      if (r.terminated) { done = true; if (r.outcome === 1) landed++; else crashed++; break; }
    }
    if (!done) hovered++;
    sumR += R;
  }
  return { land: 100 * landed / N_EVAL, crash: 100 * crashed / N_EVAL,
           hover: 100 * hovered / N_EVAL, ret: sumR / N_EVAL, ent: sumEnt / nEnt };
}

function runOne(seed) {
  const t = new C.Trainer({ seed });
  const rows = [];
  const ring = new Uint8Array(500); let filled = 0, landed = 0, ptr = 0;
  t.onEpisode = (R, outcome) => {
    const b = outcome === 1 ? 1 : 0;
    if (filled === 500) landed -= ring[ptr]; else filled++;
    ring[ptr] = b; landed += b; ptr = (ptr + 1) % 500;
  };
  for (const ck of CHECKS) {
    while (t.episode < ck) t.tick(1000, 200000);
    const g = evaluate(t, true, 987654);
    const st = evaluate(t, false, 987654);
    rows.push({
      ep: t.episode, beta: t.beta,
      trainLand: 100 * landed / Math.max(1, filled),
      greedyLand: g.land, greedyCrash: g.crash, greedyHover: g.hover, greedyRet: g.ret,
      stochLand: st.land, stochCrash: st.crash, stochHover: st.hover, stochRet: st.ret,
      polEnt: st.ent,
    });
  }
  return { seed, rows };
}

if (process.argv[2] === '--worker') {
  process.send(runOne(Number(process.argv[3])));
  process.exit(0);
}

const results = [];
let done = 0;
for (const s of SEEDS) {
  const ch = fork(__filename, ['--worker', String(s)]);
  ch.on('message', m => results.push(m));
  ch.on('exit', () => {
    if (++done === SEEDS.length) {
      results.sort((a, b) => a.seed - b.seed);
      const lp = (x, n) => String(x).padStart(n);
      for (const r of results) {
        console.log(`\n=== seed ${r.seed} ===`);
        console.log(lp('ep', 7) + lp('beta', 8) + lp('polEnt', 8) + lp('trainL%', 8) +
          lp('grdL%', 7) + lp('grdC%', 7) + lp('grdH%', 7) + lp('grdRet', 8) +
          lp('stoL%', 7) + lp('stoH%', 7) + lp('stoRet', 8));
        for (const w of r.rows) {
          console.log(lp(w.ep, 7) + lp(w.beta.toFixed(4), 8) + lp(w.polEnt.toFixed(3), 8) +
            lp(w.trainLand.toFixed(0), 8) + lp(w.greedyLand.toFixed(0), 7) +
            lp(w.greedyCrash.toFixed(0), 7) + lp(w.greedyHover.toFixed(0), 7) +
            lp(w.greedyRet.toFixed(0), 8) + lp(w.stochLand.toFixed(0), 7) +
            lp(w.stochHover.toFixed(0), 7) + lp(w.stochRet.toFixed(0), 8));
        }
      }
    }
  });
}
