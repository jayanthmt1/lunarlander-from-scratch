const C=require('./core.js');
const N=25000, SEEDS=[1,2,3];
// Prior data: entropy decay dominates escape time, monotonically (beta=0 fastest).
// Open questions: does beta=0 cost robustness, and does batch/lr help on top of it?
const CONFIGS=[
  ['baseline (HL=2000)',      {}],
  ['beta=0',                  {entropy0:0, entropyMin:0}],
  ['beta=0 + batch 16',       {entropy0:0, entropyMin:0, batch:16}],
  ['beta=0 + lr 4e-3',        {entropy0:0, entropyMin:0, lrPolicy:4e-3}],
  ['HL=500',                  {entropyHalfLife:500}],
  ['beta=0 + batch16 + lr4e-3',{entropy0:0, entropyMin:0, batch:16, lrPolicy:4e-3}],
];
function run(over,seed){
  const t=new C.Trainer(Object.assign({seed},over));
  let escape=-1, auc=0, nAuc=0, mark=0;
  while(t.episode<N){
    t.tick(60);
    if(t.episode>=mark+500){
      mark=Math.floor(t.episode/500)*500;
      let L=0,M=0,n=0;
      for(let i=Math.max(0,t.ghost.n-500);i<t.ghost.n;i++){
        if(t.ghost.outcome[i%C.CURVE_CAP]===1)L++;
        M+=t.ghost.ret[i%C.CURVE_CAP]; n++;
      }
      if(escape<0 && 100*L/n>50) escape=mark;
      auc+=M/n; nAuc++;
    }
  }
  let L=0,M=0,n=0;
  for(let i=t.ghost.n-1000;i<t.ghost.n;i++){
    if(t.ghost.outcome[i%C.CURVE_CAP]===1)L++; M+=t.ghost.ret[i%C.CURVE_CAP]; n++;
  }
  return {escape, land:100*L/n, mean:M/n, auc:auc/nAuc};
}
console.log(`tuning to ${N} episodes, ${SEEDS.length} seeds each\n`);
console.log('CONFIG                        ESCAPE(avg)  LAND%   MEAN   AUC   solved/seeds');
const results=[];
for(const [label,over] of CONFIGS){
  const rs=SEEDS.map(s=>run(over,s));
  const ok=rs.filter(r=>r.escape>0);
  const avgEsc=ok.length? ok.reduce((a,b)=>a+b.escape,0)/ok.length : -1;
  const land=rs.reduce((a,b)=>a+b.land,0)/rs.length;
  const mean=rs.reduce((a,b)=>a+b.mean,0)/rs.length;
  const auc=rs.reduce((a,b)=>a+b.auc,0)/rs.length;
  results.push({label,avgEsc,land,mean,auc,ok:ok.length});
  console.log(`${label.padEnd(30)}${(avgEsc<0?'never':Math.round(avgEsc)).toString().padStart(9)}  ${land.toFixed(0).padStart(5)}%  ${mean.toFixed(0).padStart(5)}  ${auc.toFixed(0).padStart(4)}   ${ok.length}/${SEEDS.length}`);
}
console.log('\nranked by mean return:');
results.sort((a,b)=>b.mean-a.mean).forEach((r,i)=>
  console.log(`  ${i+1}. ${r.label.padEnd(30)} mean ${r.mean.toFixed(0)}  land ${r.land.toFixed(0)}%  escape ${r.avgEsc<0?'never':Math.round(r.avgEsc)}`));
