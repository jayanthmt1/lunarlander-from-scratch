const C=require('./core.js'), fs=require('fs');
const MAX=50000, EVERY=5000, EVAL=500;
function run(label, seed, decay){
  const t=new C.Trainer(Object.assign({seed},C.PRESETS.solve));
  const lr0=t.cfg.lrPolicy;
  let best={land:-1};
  const row=[];
  for(let ck=EVERY; ck<=MAX; ck+=EVERY){
    while(t.episode<ck){
      t.tick(60);
      if(decay){ // linear decay from 20k onward, down to 25% of lr0
        const f = t.episode<20000 ? 1 : Math.max(0.25, 1-0.75*(t.episode-20000)/(MAX-20000));
        t.cfg.lrPolicy = lr0*f;
      }
    }
    const g=C.evaluatePolicy(t.pol,t.cfg,EVAL,seed*31+ck,true);
    row.push(`${ck/1000}k:${g.land.toFixed(1)}%`);
    if(g.land>best.land) best={land:g.land, mean:g.mean, ep:ck, w:t.pol.tensors().map(a=>Float64Array.from(a)), cfg:Object.assign({},t.cfg)};
  }
  console.log(`${label.padEnd(22)} ${row.join('  ')}`);
  console.log(`${''.padEnd(22)} best ${best.land.toFixed(1)}% @ ${best.ep/1000}k`);
  return best;
}
console.log(`greedy landing accuracy vs training episodes (${EVAL} fresh episodes per checkpoint)\n`);
const cands=[];
cands.push(run('solve seed 2', 2, false));
cands.push(run('solve+decay seed 2', 2, true));
cands.push(run('solve seed 1', 1, false));
cands.push(run('solve+decay seed 1', 1, true));
const win=cands.sort((a,b)=>b.land-a.land)[0];
// confirm the winner properly on a big independent sample
const pol=new C.PolicyNet(win.cfg.hidden,new C.RNG(1));
pol.tensors().forEach((d,i)=>d.set(win.w[i]));
const conf=C.evaluatePolicy(pol,win.cfg,3000,777,true);
console.log(`\nWINNER: ${win.ep/1000}k episodes`);
console.log(`confirmation on 3000 independent episodes: ${conf.land.toFixed(2)}% landed, mean ${conf.mean.toFixed(1)}, crash ${conf.crash.toFixed(2)}%, trunc ${conf.trunc.toFixed(2)}%`);
const round=a=>Array.from(a,v=>+v.toFixed(6));
fs.writeFileSync('weights-solve.json', JSON.stringify({
  architecture:`8 -> ${win.cfg.hidden} tanh -> 4 softmax`, parameters:pol.nParams,
  trainedEpisodes:win.ep, evaluation:{episodes:3000,greedy:true,landingRate:+conf.land.toFixed(2),meanReturn:+conf.mean.toFixed(1)},
  inputs:['x','y','vx','vy','theta','omega','leg1','leg2'],
  outputs:['idle','fire_right_thruster','fire_main','fire_left_thruster'],
  shapes:{W1:[win.cfg.hidden,8],b1:[win.cfg.hidden],W2:[4,win.cfg.hidden],b2:[4]},
  W1:round(pol.W1),b1:round(pol.b1),W2:round(pol.W2),b2:round(pol.b2),
},null,1));
console.log('saved weights-solve.json');
