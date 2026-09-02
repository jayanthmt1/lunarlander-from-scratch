const C = require('./core.js');
const seeds = process.argv[2] ? process.argv[2].split(',').map(Number) : [1];
const N = +(process.argv[3] || 3000);
for (const seed of seeds) {
  const t = new C.Trainer({ seed });
  const t0 = Date.now();
  let mark = 0; const row = [];
  while (t.episode < N) {
    t.tick(50);
    if (t.episode >= mark + 250) {
      mark = Math.floor(t.episode / 250) * 250;
      row.push({ ep: mark, m: t.meanRecent(250) });
    }
  }
  const el = (Date.now() - t0) / 1000;
  // outcome mix over last 500
  let land = 0, crash = 0, trunc = 0;
  for (let i = Math.max(0, t.ghost.n - 500); i < t.ghost.n; i++) {
    const o = t.ghost.outcome[i % C.CURVE_CAP];
    if (o === 1) land++; else if (o === 0) crash++; else trunc++;
  }
  console.log(`seed ${seed}: ${N} eps in ${el.toFixed(1)}s`);
  console.log('  ' + row.map(r => `${r.ep}:${r.m.toFixed(0)}`).join('  '));
  console.log(`  last 500 -> land ${land}  crash ${crash}  hover/trunc ${trunc}   meanLen ${(t.meanRecentSteps ? 0 : 0)}`);
  let sl = 0; for (let i = Math.max(0, t.ghost.n - 500); i < t.ghost.n; i++) sl += t.ghost.steps[i % C.CURVE_CAP];
  console.log(`  mean episode length (last 500): ${(sl / 500).toFixed(0)} / cap ${C.maxStepsFor(t.episode)}`);
}
