const C=require('./core.js');
const N=26000;
function esc(over,label){
  const out=[];
  for(const seed of [1,2]){
    const t=new C.Trainer(Object.assign({seed},over));
    let escape=null; const marks=[];
    let mark=0;
    while(t.episode<N){
      t.tick(60);
      if(t.episode>=mark+2000){
        mark=Math.floor(t.episode/2000)*2000;
        let L=0,M=0,n=0;
        for(let i=Math.max(0,t.ghost.n-500);i<t.ghost.n;i++){
          if(t.ghost.outcome[i%C.CURVE_CAP]===1)L++;
          M+=t.ghost.ret[i%C.CURVE_CAP]; n++;
        }
        if(escape===null && 100*L/n>50) escape=mark;
        if(mark%6000===0) marks.push(`${mark/1000}k:${(100*L/n).toFixed(0)}%`);
      }
    }
    let L=0,M=0,n=0;
    for(let i=t.ghost.n-500;i<t.ghost.n;i++){
      if(t.ghost.outcome[i%C.CURVE_CAP]===1)L++; M+=t.ghost.ret[i%C.CURVE_CAP]; n++;
    }
    out.push({seed,escape,final:(100*L/n).toFixed(0),mean:(M/n).toFixed(0),marks:marks.join(' ')});
  }
  for(const o of out)
    console.log(`${label.padEnd(30)} seed ${o.seed}  escape@${String(o.escape===null?'never':o.escape/1000+'k').padStart(6)}  final ${o.final.padStart(3)}% / ${o.mean.padStart(4)}   ${o.marks}`);
}
console.log('212 weights, 26k episodes, escape = first 500-ep window above 50% landing');
esc({}, 'stock');
esc({stiffLegs:1}, 'stiff legs');
esc({startRandom:1}, 'random start');
esc({startRandom:1,stiffLegs:1}, 'random start + stiff legs');
