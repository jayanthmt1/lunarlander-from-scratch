const C=require('./core.js'), fs=require('fs');
const CKPTS=[5000,10000,20000,25000,30000,35000,40000,50000], EVAL=1500;
function measure(pol,cfg,n,seed){
  let onpad=0,crash=0,trunc=0,sum=0,fuel=0;
  for(let e=0;e<n;e++){
    const rng=new C.RNG(seed+e*7919); const env=new C.LunarEnv(rng,cfg);
    let s=env.observe(),R=0,done=false,mf=0;
    for(let t=0;t<1000&&!done;t++){
      pol.forward(s); const a=pol.greedy(); const r=env.step(a);
      if(a===2)mf++; s=r.obs; R+=r.rawReward;
      if(r.terminated){done=true;
        if(r.outcome===1){const x=env.origin()[0]; if(x>=env.helipadX1&&x<=env.helipadX2) onpad++;}
        else crash++;}
    }
    if(!done)trunc++; sum+=R; fuel+=mf;
  }
  return {land:100*onpad/n, crash:100*crash/n, trunc:100*trunc/n, mean:sum/n, fuel:fuel/n};
}
const t=new C.Trainer(Object.assign({seed:2},C.PRESETS.solve));
const rows=[];
console.log(`checkpoint      LAND%   crash%  clock%    MEAN   main-burns`);
for(const ck of CKPTS){
  while(t.episode<ck) t.tick(60);
  const m=measure(t.pol,t.cfg,EVAL,5000);
  rows.push({ck,m,w:t.pol.tensors().map(a=>Float64Array.from(a)),cfg:Object.assign({},t.cfg)});
  console.log(`${(ck/1000+'k').padStart(10)}  ${m.land.toFixed(2).padStart(7)}%  ${m.crash.toFixed(2).padStart(5)}%  ${m.trunc.toFixed(2).padStart(5)}%  ${m.mean.toFixed(1).padStart(6)}   ${m.fuel.toFixed(0).padStart(6)}`);
}
// require >=99% landing, then maximise return
const ok=rows.filter(r=>r.m.land>=99.0);
const win=(ok.length?ok:rows).sort((a,b)=>b.m.mean-a.m.mean)[0];
console.log(`\nselected: ${win.ck/1000}k  (>=99% landing, best mean return)`);
const pol=new C.PolicyNet(win.cfg.hidden,new C.RNG(1));
pol.tensors().forEach((d,i)=>d.set(win.w[i]));
const conf=measure(pol,win.cfg,4000,987654);
console.log(`confirmation, 4000 independent episodes:`);
console.log(`  landing on pad ${conf.land.toFixed(2)}%   crash ${conf.crash.toFixed(2)}%   clock ${conf.trunc.toFixed(2)}%   mean ${conf.mean.toFixed(1)}`);
const round=a=>Array.from(a,v=>+v.toFixed(6));
fs.writeFileSync('weights-solve.json', JSON.stringify({
  architecture:`8 -> ${win.cfg.hidden} tanh -> 4 softmax`, parameters:pol.nParams,
  trainedEpisodes:win.ck, actionSelection:'greedy (argmax)',
  evaluation:{episodes:4000, landingOnPadPct:+conf.land.toFixed(2), crashPct:+conf.crash.toFixed(2),
              ranOutClockPct:+conf.trunc.toFixed(2), meanReturn:+conf.mean.toFixed(1)},
  inputs:['x','y','vx','vy','theta','omega','leg1','leg2'],
  outputs:['idle','fire_right_thruster','fire_main','fire_left_thruster'],
  shapes:{W1:[win.cfg.hidden,8],b1:[win.cfg.hidden],W2:[4,win.cfg.hidden],b2:[4]},
  W1:round(pol.W1),b1:round(pol.b1),W2:round(pol.W2),b2:round(pol.b2),
},null,1));
console.log('saved weights-solve.json');
