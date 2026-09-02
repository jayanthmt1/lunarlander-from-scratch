// How much arithmetic is actually in one training step?
const NIN=8, HN=16, NOUT=4;
const fwdPolicy = HN*NIN + NOUT*HN;          // 128 + 64
const fwdValue  = HN*NIN + HN;               // 128 + 16
const bwd       = 2*(fwdPolicy+fwdValue);    // backward ~2x forward
const macs = fwdPolicy+fwdValue+bwd;
const flops = macs*2;
console.log('ARITHMETIC PER ENV STEP');
console.log(`  policy forward MACs : ${fwdPolicy}`);
console.log(`  value  forward MACs : ${fwdValue}`);
console.log(`  backward MACs (~2x) : ${bwd}`);
console.log(`  total               : ${macs} MACs = ${flops} FLOPs\n`);

console.log('WORK NEEDED TO AMORTISE ONE GPU KERNEL LAUNCH');
const launchUs = [5,10,20];
const gpuTflops = 10e12;
for(const l of launchUs){
  const breakeven = gpuTflops*(l*1e-6);
  console.log(`  ${String(l).padStart(2)} us launch @ 10 TFLOP/s -> need ${(breakeven/1e6).toFixed(1)}M FLOPs to break even`);
  console.log(`                          we have ${flops} -> ${(breakeven/flops).toExponential(1)}x too small`);
}

console.log('\nHOW BIG WOULD THE BATCH HAVE TO BE?');
for(const l of [5,10]){
  const need = gpuTflops*(l*1e-6)/flops;
  console.log(`  to amortise a ${l}us launch you'd need ~${Math.round(need).toLocaleString()} envs stepping in lockstep`);
}
console.log('  (we run 8-32, and they de-synchronise the moment one episode ends)');

console.log('\nTHE SEQUENTIAL CONSTRAINT');
console.log('  s_t -> a_t -> s_t+1 is a strict chain. Within an episode there is');
console.log('  NOTHING to parallelise: step t+1 cannot start until t finishes.');
console.log('  A 1000-step episode = 1000 unavoidable round trips.');
const perStepCpu = 3.06;
for(const l of [5,10,20]){
  console.log(`  1000 steps: CPU ${(1000*perStepCpu/1000).toFixed(1)} ms vs GPU >= ${(1000*l/1000).toFixed(1)} ms of launch overhead ALONE (${(l/perStepCpu).toFixed(1)}x slower)`);
}
