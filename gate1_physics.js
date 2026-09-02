const C = require('./core.js');
const cfg = { fLeg: 65.0, timePenalty: 0 };

// ---- assertion A: state[1] at reset -----------------------------------------
{
  const rng = new C.RNG(7);
  const env = new C.LunarEnv(rng, cfg);
  const s = env.observe();
  console.log(`A. state[1] at reset      = ${s[1].toFixed(4)}   (expect ~1.41)`);
  console.log(`   state[2] after reset   = ${s[2].toFixed(4)}   (expect within +-0.81)`);
}

// ---- assertion B: state[1] at first pad contact ------------------------------
{
  const env = new C.LunarEnv(new C.RNG(3), cfg);
  env.cx = (env.helipadX1 + env.helipadX2) / 2;
  env.cy = 3.3 + 0.53864 + C.COM_OFF + 0.004;   // start just above contact
  env.vx = 0; env.vy = -0.02; env.theta = 0; env.omega = 0;
  env.phi = [C.PHI_MIN, C.PHI_MIN]; env.contact = [0,0]; env.prevShaping = null;
  let seen = null;
  for (let i = 0; i < 400; i++) {
    const r = env.step(0);
    if (r.obs[6] || r.obs[7]) { seen = r.obs[1]; break; }
  }
  console.log(`B. state[1] at 1st contact = ${seen === null ? 'NEVER' : seen.toFixed(4)}   (expect ~-0.0142)`);
}

// ---- assertion C: which action rotates which way -----------------------------
{
  const rng = new C.RNG(5);
  for (const a of [1, 3]) {
    const env = new C.LunarEnv(rng, cfg);
    env.cx = 10; env.cy = 8; env.vx = 0; env.vy = 0; env.theta = 0; env.omega = 0;
    const before = env.vx;
    env.step(a);
    console.log(`C. action ${a}: dvx=${(env.vx-before).toFixed(4)}  domega=${env.omega.toFixed(4)}`);
  }
}

// ---- assertion D: fly gymnasium's own heuristic() -----------------------------
function heuristic(s) {
  let angle_targ = s[0] * 0.5 + s[2] * 1.0;
  if (angle_targ > 0.4) angle_targ = 0.4;
  if (angle_targ < -0.4) angle_targ = -0.4;
  const hover_targ = 0.55 * Math.abs(s[0]);
  let angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0;
  let hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5;
  if (s[6] || s[7]) { angle_todo = 0; hover_todo = -s[3] * 0.5; }
  let a = 0;
  if (hover_todo > Math.abs(angle_todo) && hover_todo > 0.05) a = 2;
  else if (angle_todo < -0.05) a = 3;
  else if (angle_todo > +0.05) a = 1;
  return a;
}

let land = 0, crash = 0, trunc = 0, tot = 0;
const rets = [];
for (let ep = 0; ep < 100; ep++) {
  const rng = new C.RNG(1000 + ep);
  const env = new C.LunarEnv(rng, cfg);
  let s = env.observe(), R = 0, done = false;
  for (let t = 0; t < 1000 && !done; t++) {
    const r = env.step(heuristic(s));
    s = r.obs; R += r.reward;
    if (r.terminated) { done = true; if (r.outcome === 1) land++; else crash++; }
  }
  if (!done) trunc++;
  rets.push(R); tot += R;
}
rets.sort((a,b)=>a-b);
console.log(`\nD. gymnasium heuristic over 100 episodes:`);
console.log(`   landed ${land}   crashed ${crash}   truncated ${trunc}`);
console.log(`   mean return ${(tot/100).toFixed(1)}   median ${rets[50].toFixed(1)}`);
console.log(`   (real gymnasium: ~85-95 landed, mean return ~+230)`);
