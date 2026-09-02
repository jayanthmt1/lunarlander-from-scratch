const C=require('./core.js');
const N=60000;
for(const seed of [1,2]){
  const t=new C.Trainer({seed});
  const row=[]; let mark=0;
  const t0=Date.now();
  while(t.episode<N){
    t.tick(60);
    if(t.episode>=mark+5000){
      mark=Math.floor(t.episode/5000)*5000;
      let L=0,M=0,n=0;
      for(let i=Math.max(0,t.ghost.n-500);i<t.ghost.n;i++){
        if(t.ghost.outcome[i%C.CURVE_CAP]===1)L++;
        M+=t.ghost.ret[i%C.CURVE_CAP]; n++;
      }
      row.push(`${mark/1000}k:${(100*L/n).toFixed(0)}%/${(M/n).toFixed(0)}`);
    }
  }
  console.log(`seed ${seed} (${((Date.now()-t0)/1000).toFixed(0)}s)  land%/meanReturn`);
  console.log('  '+row.join('  '));
}
