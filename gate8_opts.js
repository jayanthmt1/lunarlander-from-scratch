const C=require('./core.js');
function heuristic(s){let at=s[0]*0.5+s[2]*1.0;at=Math.max(-0.4,Math.min(0.4,at));
  const ht=0.55*Math.abs(s[0]);let ad=(at-s[4])*0.5-s[5]*1.0,hd=(ht-s[1])*0.5-s[3]*0.5;
  if(s[6]||s[7]){ad=0;hd=-s[3]*0.5;} let a=0;
  if(hd>Math.abs(ad)&&hd>0.05)a=2;else if(ad<-0.05)a=3;else if(ad>0.05)a=1;return a;}
const base={fLeg:65,timePenalty:0,padChunk:5,padRandom:0,startRandom:0,stiffLegs:0};
function ev(over,label){
  const cfg=Object.assign({},base,over); let L=0,Cr=0,Tr=0,tot=0;
  for(let ep=0;ep<150;ep++){
    const e=new C.LunarEnv(new C.RNG(1000+ep),cfg); let s=e.observe(),R=0,done=false;
    for(let t=0;t<1000&&!done;t++){const r=e.step(heuristic(s));s=r.obs;R+=r.reward;
      if(r.terminated){done=true;r.outcome===1?L++:Cr++;}}
    if(!done)Tr++; tot+=R;
  }
  console.log(`${label.padEnd(34)} landed ${String(L).padStart(3)}/150  crashed ${String(Cr).padStart(3)}  trunc ${String(Tr).padStart(2)}  mean ${(tot/150).toFixed(0).padStart(5)}`);
}
// vertical impact speed a leg survives
function crashSpeed(over){
  const cfg=Object.assign({},base,over);
  for(let v=0.2;v<20;v+=0.1){
    const e=new C.LunarEnv(new C.RNG(11),cfg);
    e.cx=(e.helipadX1+e.helipadX2)/2; e.cy=3.3+0.5386+C.COM_OFF+0.001;
    e.vx=0;e.vy=-v;e.theta=0;e.omega=0;e.phi=[C.PHI_MIN,C.PHI_MIN];
    e.contact=[0,0];e.prevShaping=null;e.gameOver=false;e.sleepTimer=0;
    for(let t=0;t<400;t++){const r=e.step(0); if(r.terminated) { if(r.outcome===0) return v; break; }}
  }
  return null;
}
console.log('--- gymnasium heuristic, 150 episodes each ---');
ev({}, 'stock (bending legs, fixed start)');
ev({stiffLegs:1}, 'stiff legs');
ev({startRandom:1}, 'random start');
ev({startRandom:1,stiffLegs:1}, 'random start + stiff legs');
ev({startRandom:1,stiffLegs:1,padRandom:1}, 'random start + stiff + random pad');
console.log('\n--- survivable vertical impact speed (straight drop) ---');
const a=crashSpeed({}), b=crashSpeed({stiffLegs:1});
console.log(`bending legs (F_LEG 65): crashes above ${a?a.toFixed(1)+' m/s':'never'}`);
console.log(`stiff legs             : crashes above ${b?b.toFixed(1)+' m/s':'never — survives any impact'}`);
