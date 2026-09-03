const C=require('./core.js'), fs=require('fs');
const SEED=2, TRAIN=40000;
const t=new C.Trainer(Object.assign({seed:SEED},C.PRESETS.solve));
while(t.episode<TRAIN) t.tick(60);
const g=C.evaluatePolicy(t.pol,t.cfg,2000,SEED,true);
console.log(`trained ${TRAIN} eps, seed ${SEED}`);
console.log(`greedy over 2000 fresh episodes: ${g.land.toFixed(2)}% landed, mean ${g.mean.toFixed(1)}, crash ${g.crash.toFixed(2)}%, trunc ${g.trunc.toFixed(2)}%`);
const p=t.pol, hn=p.hn;
const round=a=>Array.from(a,v=>+v.toFixed(6));
const out={
  architecture:`8 -> ${hn} tanh -> 4 softmax`,
  parameters:p.nParams,
  trainedEpisodes:TRAIN, seed:SEED,
  evaluation:{episodes:2000, greedy:true, landingRate:+g.land.toFixed(2), meanReturn:+g.mean.toFixed(1)},
  config:{hidden:hn,gamma:t.cfg.gamma,lrPolicy:t.cfg.lrPolicy,batch:t.cfg.batch,entropy0:t.cfg.entropy0},
  inputs:['x','y','vx','vy','theta','omega','leg1','leg2'],
  outputs:['idle','fire_right_thruster','fire_main','fire_left_thruster'],
  shapes:{W1:[hn,8],b1:[hn],W2:[4,hn],b2:[4]},
  W1:round(p.W1), b1:round(p.b1), W2:round(p.W2), b2:round(p.b2),
};
fs.writeFileSync('weights-solve.json', JSON.stringify(out,null,1));
const st=a=>{const A=Array.from(a);const m=A.reduce((x,y)=>x+y,0)/A.length;
  return `n=${A.length} mean=${m.toFixed(3)} sd=${Math.sqrt(A.reduce((s,v)=>s+(v-m)**2,0)/A.length).toFixed(3)} min=${Math.min(...A).toFixed(3)} max=${Math.max(...A).toFixed(3)}`};
console.log('\nweight tensors:');
console.log('  W1 [64x8]  '+st(p.W1));
console.log('  b1 [64]    '+st(p.b1));
console.log('  W2 [4x64]  '+st(p.W2));
console.log('  b2 [4]     ['+round(p.b2).join(', ')+']');
console.log('\nwritten to weights-solve.json ('+(fs.statSync('weights-solve.json').size/1024).toFixed(1)+' KB)');
