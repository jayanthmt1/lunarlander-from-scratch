const C=require('./core.js');
const rng=new C.RNG(1);
const cfg=Object.assign({},C.DEFAULTS);
const env=new C.LunarEnv(rng,cfg);
const pol=new C.PolicyNet(16,rng), val=new C.ValueNet(16,rng);
const s=env.observe();
function bench(label,n,fn){
  fn(); // warm the JIT
  const t0=process.hrtime.bigint();
  for(let i=0;i<n;i++) fn();
  const ns=Number(process.hrtime.bigint()-t0)/n;
  console.log(`  ${label.padEnd(34)} ${ns.toFixed(0).padStart(5)} ns/call`);
  return ns;
}
console.log('per-call cost of each piece (single CPU core, V8 JIT):');
const a=bench('physics step',            2e6, ()=>{ if(env.gameOver||env.t>800) env.reset(); env.step(2); });
const b=bench('policy forward (208 MACs)',5e6, ()=>pol.forward(s));
const c=bench('value  forward (161 MACs)',5e6, ()=>val.forward(s));
const d=bench('policy backward',          5e6, ()=>pol.backward(s,1,0.5,0.01));
const e=bench('value  backward',          5e6, ()=>val.backward(s,0.5));
console.log(`  ${'TOTAL per env step'.padEnd(34)} ${(a+b+c+d+e).toFixed(0).padStart(5)} ns`);
console.log('\nMath.tanh vs a Pade approximation (32 calls per step):');
let acc=0;
const t1=bench('32x Math.tanh',           2e6, ()=>{ for(let i=0;i<32;i++) acc+=Math.tanh(i*0.03-0.5); });
const t2=bench('32x Pade x(27+x2)/(27+9x2)',2e6,()=>{ for(let i=0;i<32;i++){const x=i*0.03-0.5,x2=x*x;acc+=x*(27+x2)/(27+9*x2);} });
console.log(`  -> tanh is ${(t1/t2).toFixed(1)}x the cost of Pade; ${(100*t1/(a+b+c+d+e)).toFixed(0)}% of a full step`);
