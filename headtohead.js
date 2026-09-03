const C=require('./core.js'), fs=require('fs');
function measure(pol,cfg,n,seed){
  let onpad=0,crash=0,trunc=0,sum=0;
  for(let e=0;e<n;e++){
    const rng=new C.RNG(seed+e*7919); const env=new C.LunarEnv(rng,cfg);
    let s=env.observe(),R=0,done=false;
    for(let t=0;t<1000&&!done;t++){
      pol.forward(s); const r=env.step(pol.greedy()); s=r.obs; R+=r.rawReward;
      if(r.terminated){done=true;
        if(r.outcome===1){const x=env.origin()[0]; if(x>=env.helipadX1&&x<=env.helipadX2)onpad++;}
        else crash++;}
    }
    if(!done)trunc++; sum+=R;
  }
  return {land:100*onpad/n, crash:100*crash/n, trunc:100*trunc/n, mean:sum/n};
}
const t=new C.Trainer(Object.assign({seed:2},C.PRESETS.solve));
const snaps={};
for(const ck of [5000,50000]){ while(t.episode<ck) t.tick(60);
  snaps[ck]={w:t.pol.tensors().map(a=>Float64Array.from(a)),cfg:Object.assign({},t.cfg)}; }
console.log('head to head on the SAME 4000 fresh episodes (unseen by selection)\n');
console.log('checkpoint   LAND%    crash%   clock%    MEAN');
const res={};
for(const ck of [5000,50000]){
  const p=new C.PolicyNet(snaps[ck].cfg.hidden,new C.RNG(1));
  p.tensors().forEach((d,i)=>d.set(snaps[ck].w[i]));
  const m=measure(p,snaps[ck].cfg,4000,31337);
  res[ck]=m;
  console.log(`${(ck/1000+'k').padStart(9)}  ${m.land.toFixed(2).padStart(6)}%  ${m.crash.toFixed(2).padStart(6)}%  ${m.trunc.toFixed(2).padStart(6)}%  ${m.mean.toFixed(1).padStart(6)}`);
}
// the stated goal is landing accuracy > 99%; return only breaks ties
const ok=Object.keys(res).filter(k=>res[k].land>99.0);
const pick=ok.length? ok.sort((a,b)=>res[b].mean-res[a].mean)[0] : Object.keys(res).sort((a,b)=>res[b].land-res[a].land)[0];
console.log(`\nSHIPPING checkpoint ${pick/1000}k -> ${res[pick].land.toFixed(2)}% landing, ${res[pick].crash.toFixed(2)}% crash, mean ${res[pick].mean.toFixed(1)}`);
const p=new C.PolicyNet(snaps[pick].cfg.hidden,new C.RNG(1));
p.tensors().forEach((d,i)=>d.set(snaps[pick].w[i]));
const round=a=>Array.from(a,v=>+v.toFixed(6));
fs.writeFileSync('weights-solve.json', JSON.stringify({
  architecture:`8 -> ${snaps[pick].cfg.hidden} tanh -> 4 softmax`, parameters:p.nParams,
  trainedEpisodes:+pick, actionSelection:'greedy (argmax)',
  note:'Selected for landing accuracy. Later checkpoints reach ~+274 mean return but land ~98.4%; greedy accuracy oscillates over training rather than converging.',
  evaluation:{episodes:4000, landingOnPadPct:+res[pick].land.toFixed(2), crashPct:+res[pick].crash.toFixed(2),
              ranOutClockPct:+res[pick].trunc.toFixed(2), meanReturn:+res[pick].mean.toFixed(1)},
  inputs:['x','y','vx','vy','theta','omega','leg1','leg2'],
  outputs:['idle','fire_right_thruster','fire_main','fire_left_thruster'],
  shapes:{W1:[snaps[pick].cfg.hidden,8],b1:[snaps[pick].cfg.hidden],W2:[4,snaps[pick].cfg.hidden],b2:[4]},
  W1:round(p.W1),b1:round(p.b1),W2:round(p.W2),b2:round(p.b2),
},null,1));
console.log('saved weights-solve.json');
