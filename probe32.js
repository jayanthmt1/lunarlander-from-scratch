const C=require('./core.js');
for(const over of [{hidden:32},{hidden:32,padRandom:1}]){
  const lbl = over.padRandom ? 'hidden 32, random pad' : 'hidden 32, centred';
  for(const seed of [1,2]){
    const t=new C.Trainer(Object.assign({seed},over));
    const row=[]; let mark=0;
    while(t.episode<14000){
      t.tick(50);
      if(t.episode>=mark+2000){ mark=Math.floor(t.episode/2000)*2000;
        let L=0,n=0; for(let i=Math.max(0,t.ghost.n-500);i<t.ghost.n;i++){ if(t.ghost.outcome[i%C.CURVE_CAP]===1)L++; n++; }
        row.push(`${mark/1000}k:${(100*L/n).toFixed(0)}%`);
      }
    }
    console.log(`${lbl} seed ${seed}  landing rate ->  ${row.join('  ')}`);
  }
}
