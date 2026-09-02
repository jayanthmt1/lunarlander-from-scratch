const C = require('./core.js');
const N = 9000;
function run(over,label){
  let L=0,Cr=0,T=0,M=0,n=0;
  for(const seed of [1,2]){
    const t=new C.Trainer(Object.assign({seed},over));
    while(t.episode<N) t.tick(50);
    for(let i=t.ghost.n-500;i<t.ghost.n;i++){
      const o=t.ghost.outcome[i%C.CURVE_CAP];
      if(o===1)L++;else if(o===0)Cr++;else T++;
      M+=t.ghost.ret[i%C.CURVE_CAP]; n++;
    }
  }
  console.log(`${label.padEnd(30)} mean ${(M/n).toFixed(0).padStart(5)}   land ${(100*L/n).toFixed(0).padStart(3)}%  crash ${(100*Cr/n).toFixed(0).padStart(3)}%  hover ${(100*T/n).toFixed(0).padStart(3)}%`);
}
console.log('--- 212 weights (hidden 16), 9k episodes, 2 seeds ---');
run({}, 'centred pad (stock)');
run({padChunk:2}, 'pad far left (x=4)');
run({padChunk:8}, 'pad far right (x=16)');
run({padRandom:1}, 'random pad each episode');
console.log('--- 420 weights (hidden 32) ---');
run({hidden:32}, 'centred pad');
run({hidden:32,padRandom:1}, 'random pad each episode');
