const C = require('./core.js');
const cfg = { fLeg: 40.0, timePenalty: 0 };
function heuristic(s){let at=s[0]*0.5+s[2]*1.0;at=Math.max(-0.4,Math.min(0.4,at));
  const ht=0.55*Math.abs(s[0]);let ad=(at-s[4])*0.5-s[5]*1.0,hd=(ht-s[1])*0.5-s[3]*0.5;
  if(s[6]||s[7]){ad=0;hd=-s[3]*0.5;} let a=0;
  if(hd>Math.abs(ad)&&hd>0.05)a=2;else if(ad<-0.05)a=3;else if(ad>0.05)a=1;return a;}

// gentle drop for assertion B (slow descent so discretisation error is small)
{
  const env = new C.LunarEnv(new C.RNG(3), cfg);
  env.cx=(env.helipadX1+env.helipadX2)/2; env.cy=3.3+0.5386+0.0984+0.02;
  env.vx=0;env.vy=-0.05;env.theta=0;env.omega=0;env.phi=[C.PHI_MIN,C.PHI_MIN];
  env.contact=[0,0];env.prevShaping=null;
  for(let i=0;i<200;i++){const r=env.step(0); if(r.obs[6]||r.obs[7]){
    console.log(`B(gentle). state[1] at 1st contact = ${r.obs[1].toFixed(4)}  (expect ~-0.0142)`);break;}}
}

// trace one heuristic episode's final 25 steps
const env = new C.LunarEnv(new C.RNG(1000), cfg);
let s = env.observe(); const hist=[];
for(let t=0;t<1000;t++){
  const pre={vy:env.vy,vx:env.vx,phi:[...env.phi],th:env.theta,om:env.omega,cy:env.cy};
  const r=env.step(heuristic(s)); s=r.obs;
  hist.push({t,vy:pre.vy,phi0:pre.phi[0],phi1:pre.phi[1],c:[...env.contact],
             th:pre.th,om:pre.om,go:env.gameOver,rw:r.reward,term:r.terminated,oc:r.outcome});
  if(r.terminated) break;
}
console.log(`\nepisode ended at t=${hist.length}, outcome=${hist[hist.length-1].oc}`);
console.log('  t     vy      phi0   phi1  c0 c1   theta   omega   reward');
for(const h of hist.slice(-25))
  console.log(`${String(h.t).padStart(4)} ${h.vy.toFixed(3).padStart(7)} ${h.phi0.toFixed(3).padStart(6)} ${h.phi1.toFixed(3).padStart(6)}  ${h.c[0]} ${h.c[1]} ${h.th.toFixed(4).padStart(8)} ${h.om.toFixed(3).padStart(7)} ${h.rw.toFixed(2).padStart(8)}`);

// touchdown speed distribution + crash cause
let firstContactVy=[], crashPhi=[], crashTheta=[];
for(let ep=0;ep<60;ep++){
  const e=new C.LunarEnv(new C.RNG(1000+ep),cfg); let ss=e.observe(); let fc=null;
  for(let t=0;t<1000;t++){
    const pvy=e.vy; const r=e.step(heuristic(ss)); ss=r.obs;
    if(fc===null&&(e.contact[0]||e.contact[1])){fc=pvy;firstContactVy.push(pvy);}
    if(r.terminated){ if(r.outcome===0){crashPhi.push(Math.max(e.phi[0],e.phi[1]));crashTheta.push(e.theta);} break;}
  }
}
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
firstContactVy.sort((a,b)=>a-b);
console.log(`\ntouchdown vy: mean ${mean(firstContactVy).toFixed(3)}  median ${firstContactVy[30].toFixed(3)}  worst ${firstContactVy[0].toFixed(3)}`);
console.log(`crash-time phi (max leg): mean ${mean(crashPhi).toFixed(3)}  (PHI_MAX=0.9, hull hits at ~0.605)`);
console.log(`crash-time |theta|: mean ${mean(crashTheta.map(Math.abs)).toFixed(3)} rad`);
