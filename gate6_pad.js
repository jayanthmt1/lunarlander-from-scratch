const C = require('./core.js');
function heuristic(s){let at=s[0]*0.5+s[2]*1.0;at=Math.max(-0.4,Math.min(0.4,at));
  const ht=0.55*Math.abs(s[0]);let ad=(at-s[4])*0.5-s[5]*1.0,hd=(ht-s[1])*0.5-s[3]*0.5;
  if(s[6]||s[7]){ad=0;hd=-s[3]*0.5;} let a=0;
  if(hd>Math.abs(ad)&&hd>0.05)a=2;else if(ad<-0.05)a=3;else if(ad>0.05)a=1;return a;}
function evalPad(over,label){
  const cfg=Object.assign({fLeg:65,timePenalty:0,padChunk:5,padRandom:0},over);
  let land=0,crash=0,trunc=0,tot=0;
  for(let ep=0;ep<100;ep++){
    const e=new C.LunarEnv(new C.RNG(1000+ep),cfg); let s=e.observe(),R=0,done=false;
    for(let t=0;t<1000&&!done;t++){const r=e.step(heuristic(s));s=r.obs;R+=r.reward;
      if(r.terminated){done=true;r.outcome===1?land++:crash++;}}
    if(!done)trunc++; tot+=R;
  }
  console.log(`${label.padEnd(26)} landed ${String(land).padStart(3)}  crashed ${String(crash).padStart(3)}  trunc ${String(trunc).padStart(3)}  mean ${(tot/100).toFixed(0).padStart(5)}`);
}
console.log('--- gymnasium heuristic, pad moved (pad x = 2*chunk) ---');
for(const pc of [2,3,4,5,6,7,8]) evalPad({padChunk:pc}, `pad chunk ${pc} (x=${2*pc})`);
evalPad({padRandom:1}, 'random pad each episode');
