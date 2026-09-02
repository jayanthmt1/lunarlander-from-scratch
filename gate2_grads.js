const C = require('./core.js');
const rng = new C.RNG(42);

function randState(){ const s=new Float64Array(8);
  for(let i=0;i<6;i++) s[i]=rng.uniform(-1.2,1.2);
  s[6]=rng.next()<0.3?1:0; s[7]=rng.next()<0.3?1:0; return s; }

// a batch of samples, so we also test gradient ACCUMULATION across steps
const N=7, beta=0.02;
const batch=[]; for(let i=0;i<N;i++) batch.push({s:randState(),a:(rng.next()*4)|0,A:rng.uniform(-2,2),tgt:rng.uniform(-2,2)});

function policyLoss(net){ let L=0;
  for(const b of batch){ net.forward(b.s); L += -b.A*net.lp[b.a] - beta*net.entropy; } return L; }
function valueLoss(net){ let L=0;
  for(const b of batch){ const v=net.forward(b.s); L += 0.5*(v-b.tgt)*(v-b.tgt); } return L; }

function check(name, net, lossFn, analytic){
  net.zeroGrad(); analytic();
  const T=net.tensors(), G=net.grads();
  const labels=['W1','b1','W2','b2'];
  let worst=0, worstAt='', n=0, eps=1e-5;
  for(let i=0;i<T.length;i++){
    for(let j=0;j<T[i].length;j++){
      const o=T[i][j];
      T[i][j]=o+eps; const Lp=lossFn(net);
      T[i][j]=o-eps; const Lm=lossFn(net);
      T[i][j]=o;
      const fd=(Lp-Lm)/(2*eps), an=G[i][j];
      const rel=Math.abs(fd-an)/Math.max(1e-6,Math.abs(fd)+Math.abs(an));
      if(rel>worst){worst=rel;worstAt=`${labels[i]}[${j}] fd=${fd.toExponential(4)} an=${an.toExponential(4)}`;}
      n++;
    }
  }
  const ok = worst < 1e-4;
  console.log(`${ok?'PASS':'FAIL'}  ${name}: ${n} params, worst rel err ${worst.toExponential(3)}`);
  if(!ok) console.log(`      worst at ${worstAt}`);
  return ok;
}

let allOk = true;
for (const hn of [16, 64]) {
  const pr = new C.RNG(7);
  const pol = new C.PolicyNet(hn, pr), val = new C.ValueNet(hn, pr);
  console.log(`\n--- hidden = ${hn}  (policy ${pol.nParams} params, value ${val.nParams} params) ---`);
  allOk &= check('PolicyNet (REINFORCE + entropy)', pol, policyLoss,
    () => { for(const b of batch){ pol.forward(b.s); pol.backward(b.s,b.a,b.A,beta); } });
  allOk &= check('ValueNet  (MSE)', val, valueLoss,
    () => { for(const b of batch){ val.forward(b.s); val.backward(b.s,b.tgt); } });
}
// also verify entropy gradient in isolation (beta only, A=0)
{
  const pr=new C.RNG(9); const pol=new C.PolicyNet(16,pr);
  const only=(net)=>{let L=0;for(const b of batch){net.forward(b.s);L+=-1.0*net.entropy;}return L;};
  console.log('');
  allOk &= check('PolicyNet entropy term alone', pol, only,
    () => { for(const b of batch){ pol.forward(b.s); pol.backward(b.s,b.a,0.0,1.0); } });
}
console.log(allOk ? '\nGATE 2 PASSED' : '\nGATE 2 FAILED');
process.exit(allOk?0:1);
