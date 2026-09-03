const C=require('./core.js');
const TRAIN=30000, EVAL=1000;

function greedyAct(pol,s){ pol.forward(s); let b=0; for(let j=1;j<4;j++) if(pol.p[j]>pol.p[b]) b=j; return b; }

function evaluate(pol,cfg,seed0,greedy,n){
  let land=0,crash=0,trunc=0,sum=0; const fails=[];
  for(let e=0;e<n;e++){
    const rng=new C.RNG(900000+seed0*7919+e);
    const env=new C.LunarEnv(rng,cfg);
    let s=env.observe(),R=0,done=false;
    const s0={vx:s[2],vy:s[3],x:s[0]};
    for(let t=0;t<1000&&!done;t++){
      const a = greedy ? greedyAct(pol,s) : (pol.forward(s), pol.sample(rng));
      const r=env.step(a); s=r.obs; R+=r.rawReward;
      if(r.terminated){done=true; if(r.outcome===1) land++; else {crash++; fails.push({...s0,R,t});}}
    }
    if(!done){trunc++; fails.push({...s0,R,t:1000,to:1});}
    sum+=R;
  }
  return {land:100*land/n, crash:100*crash/n, trunc:100*trunc/n, mean:sum/n, fails};
}

console.log(`train ${TRAIN} eps (beta=0), then evaluate ${EVAL} fresh episodes\n`);
console.log('HIDDEN  SEED   SAMPLED-LAND%   GREEDY-LAND%   GREEDY-MEAN   crash%  timeout%');
const all=[];
for(const hidden of [16,32,64]){
  for(const seed of [1,2]){
    const t=new C.Trainer({seed,hidden,entropy0:0,entropyMin:0});
    while(t.episode<TRAIN) t.tick(60);
    const st=evaluate(t.pol,t.cfg,seed,false,300);
    const gr=evaluate(t.pol,t.cfg,seed,true,EVAL);
    all.push({hidden,seed,gr});
    console.log(`${String(hidden).padStart(6)}  ${String(seed).padStart(4)}   ${st.land.toFixed(1).padStart(11)}%   ${gr.land.toFixed(1).padStart(10)}%   ${gr.mean.toFixed(0).padStart(11)}   ${gr.crash.toFixed(1).padStart(5)}%  ${gr.trunc.toFixed(1).padStart(7)}%`);
  }
}
// what do the greedy failures look like?
const best=all.sort((a,b)=>b.gr.land-a.gr.land)[0];
console.log(`\nbest: hidden ${best.hidden} seed ${best.seed} -> ${best.gr.land.toFixed(1)}% greedy`);
const f=best.gr.fails;
console.log(`failures: ${f.length}/${EVAL}`);
if(f.length){
  const to=f.filter(x=>x.to).length;
  console.log(`  timeouts (hovered out): ${to}   crashes: ${f.length-to}`);
  const q=(a,p)=>{const b=[...a].sort((x,y)=>x-y);return b[Math.floor(p*(b.length-1))]};
  const avx=f.map(x=>Math.abs(x.vx)), avy=f.map(x=>Math.abs(x.vy));
  console.log(`  |vx0| at failure  median ${q(avx,.5).toFixed(3)}  p90 ${q(avx,.9).toFixed(3)}`);
  console.log(`  |vy0| at failure  median ${q(avy,.5).toFixed(3)}  p90 ${q(avy,.9).toFixed(3)}`);
}
