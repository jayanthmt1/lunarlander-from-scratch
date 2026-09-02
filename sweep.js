const C = require('./core.js');
function heuristic(s){let at=s[0]*0.5+s[2]*1.0;at=Math.max(-0.4,Math.min(0.4,at));
  const ht=0.55*Math.abs(s[0]);let ad=(at-s[4])*0.5-s[5]*1.0,hd=(ht-s[1])*0.5-s[3]*0.5;
  if(s[6]||s[7]){ad=0;hd=-s[3]*0.5;} let a=0;
  if(hd>Math.abs(ad)&&hd>0.05)a=2;else if(ad<-0.05)a=3;else if(ad>0.05)a=1;return a;}

function evalF(fLeg, n){
  const cfg={fLeg,timePenalty:0}; let land=0,crash=0,trunc=0,tot=0;
  for(let ep=0;ep<n;ep++){
    const e=new C.LunarEnv(new C.RNG(1000+ep),cfg); let s=e.observe(),R=0,done=false;
    for(let t=0;t<1000&&!done;t++){const r=e.step(heuristic(s));s=r.obs;R+=r.reward;
      if(r.terminated){done=true;r.outcome===1?land++:crash++;}}
    if(!done)trunc++; tot+=R;
  }
  return {land,crash,trunc,mean:tot/n};
}
// straight-down drop test: what impact speed does a leg survive?
function crashSpeed(fLeg){
  const cfg={fLeg,timePenalty:0};
  for(let v=0.2;v<6;v+=0.05){
    const e=new C.LunarEnv(new C.RNG(11),cfg);
    e.cx=(e.helipadX1+e.helipadX2)/2; e.cy=3.3+0.5386+0.0984+0.001;
    e.vx=0;e.vy=-v;e.theta=0;e.omega=0;e.phi=[C.PHI_MIN,C.PHI_MIN];
    e.contact=[0,0];e.prevShaping=null;e.gameOver=false;
    let crashed=false;
    for(let t=0;t<300;t++){const r=e.step(0); if(r.terminated){crashed=r.outcome===0;break;}}
    if(crashed) return v;
  }
  return 99;
}
console.log('F_LEG  crashSpd  landed  crashed  trunc   meanReturn   (gym heuristic: ~85-95 land, ~+230)');
for(const f of [40,50,60,70,85,100,120,150]){
  const r=evalF(f,100), cs=crashSpeed(f);
  console.log(`${String(f).padStart(5)}  ${cs.toFixed(2).padStart(7)}  ${String(r.land).padStart(6)}  ${String(r.crash).padStart(7)}  ${String(r.trunc).padStart(5)}  ${r.mean.toFixed(1).padStart(10)}`);
}
