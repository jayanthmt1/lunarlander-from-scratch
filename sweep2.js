const C = require('./core.js');
const N = 12000;
function evalCfg(over, seeds){
  let L=0,Cr=0,T=0,M=0,len=0,n=0;
  for(const seed of seeds){
    const t=new C.Trainer(Object.assign({seed},over));
    while(t.episode<N) t.tick(50);
    for(let i=t.ghost.n-500;i<t.ghost.n;i++){
      const o=t.ghost.outcome[i%C.CURVE_CAP];
      if(o===1)L++;else if(o===0)Cr++;else T++;
      M+=t.ghost.ret[i%C.CURVE_CAP]; len+=t.ghost.steps[i%C.CURVE_CAP]; n++;
    }
  }
  return {land:100*L/n, crash:100*Cr/n, hov:100*T/n, ret:M/n, len:len/n};
}
console.log('--- hidden width (params) ---');
for(const h of [16,24,32,48,64,96]){
  const p = h*8+h + 4*h+4;
  const r=evalCfg({hidden:h},[1,2,3]);
  console.log(`hidden ${String(h).padStart(3)} (${String(p).padStart(4)} w): ret ${r.ret.toFixed(0).padStart(4)}  land ${r.land.toFixed(0).padStart(3)}%  crash ${r.crash.toFixed(0).padStart(3)}%  hover ${r.hov.toFixed(0).padStart(3)}%  len ${r.len.toFixed(0)}`);
}
console.log('--- time penalty, at hidden 16 ---');
for(const tp of [0.05,0.1,0.2,0.4]){
  const r=evalCfg({timePenalty:tp},[1,2]);
  console.log(`timePen ${String(tp).padEnd(5)}      : ret ${r.ret.toFixed(0).padStart(4)}  land ${r.land.toFixed(0).padStart(3)}%  crash ${r.crash.toFixed(0).padStart(3)}%  hover ${r.hov.toFixed(0).padStart(3)}%  len ${r.len.toFixed(0)}`);
}
