const C = require('./core.js');
const N = 12000;
const configs = [
  ['faithful (default)',      {}],
  ['+ time penalty 0.05',     { timePenalty: 0.05 }],
  ['+ gamma .999, batch 16',  { gamma: 0.999, batch: 16 }],
  ['+ hidden 64',             { hidden: 64 }],
  ['+ time pen & gamma .999', { timePenalty: 0.05, gamma: 0.999, batch: 16 }],
];
console.log('config                       meanRet(last500)  land  crash  hover   meanLen');
for (const [label, over] of configs) {
  let L=0,Cr=0,T=0,M=0,len=0;
  for (const seed of [1,2]) {
    const t = new C.Trainer(Object.assign({ seed }, over));
    while (t.episode < N) t.tick(50);
    for (let i = t.ghost.n-500; i < t.ghost.n; i++) {
      const o = t.ghost.outcome[i % C.CURVE_CAP];
      if (o===1) L++; else if (o===0) Cr++; else T++;
      M += t.ghost.ret[i % C.CURVE_CAP]; len += t.ghost.steps[i % C.CURVE_CAP];
    }
  }
  const n = 1000;
  console.log(`${label.padEnd(28)} ${(M/n).toFixed(0).padStart(8)}      ${String(L/10+'%').padStart(6)} ${String((Cr/10).toFixed(0)+'%').padStart(6)} ${String((T/10).toFixed(0)+'%').padStart(6)}  ${(len/n).toFixed(0).padStart(6)}`);
}
