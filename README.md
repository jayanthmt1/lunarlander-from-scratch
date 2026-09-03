# Not Crashing Comes First

A neural network with **212 weights** learns to land on the moon — trained live in the
browser, with hand-written backpropagation and no ML library.

Open `lunar-reinforce.html` in any browser. No server, no build step, no dependencies.

Everything is written from scratch in vanilla JS: the matmuls, the backward pass, the Adam
optimiser, and a reimplementation of Gymnasium's `LunarLander-v3` physics (Box2D is a Python
C-extension, so it can't run in a browser).

- Policy: `8 → 16 tanh → 4 softmax` = **212 parameters**
- Value baseline: `8 → 16 tanh → 1` = 161 parameters
- REINFORCE with a value baseline, normalised advantages, entropy bonus, Adam
- Trains ~300,000 environment steps/second on **one CPU core**

## Layout

| file | what it is |
|---|---|
| `core.js` | training core — RNG, `LunarEnv`, nets, Adam, `Trainer`, buffers. No DOM. |
| `template.html` | page shell + canvas renderer, with a `/*__CORE__*/` marker |
| `build.sh` | inlines `core.js` into the template, syntax-checks, writes the artifact |
| `lunar-reinforce.html` | **the deliverable** — single self-contained file |
| `gate*.js` | verification gates (see below) |
| `sweep*.js`, `probe*.js`, `longrun.js` | experiments behind the findings |
| `profile.js`, `gpu_case.js` | performance analysis |

```sh
./build.sh          # rebuild the artifact from core.js + template.html
node gate1_physics.js   # physics fidelity
node gate2_grads.js     # gradient correctness
```

## Verification gates

These are the reason to trust any number in here.

**Gate 1 — physics fidelity.** Fly Gymnasium's own `heuristic()` controller through this
reimplementation. It scores **90/100 landed, mean +233.1** against the real environment's
documented ~90 and ~+230. Observation checks: `state[1]` is 1.4098 at reset (expected 1.41)
and −0.0143 at first pad contact (expected −0.0142).

**Gate 2 — gradient correctness.** Every one of the 373 parameters is checked against a
central finite difference. Worst relative error **6.7e-9** against a 1e-4 threshold,
including the entropy term `∂H/∂z = −p(lp + H)`.

## Presets

| preset | network | what it does |
|---|---|---|
| **Faithful** *(default)* | 8-16-4, **212 weights** | reproduces the hovering plateau — the story the page tells |
| **Solve** | 8-64-4, **836 weights** | **99.50% landing**, 0.03% crash (see caveat below) |

`Solve` is `entropy0 = 0`, `lrPolicy = 4e-3`, hidden 64. It is a *different, larger*
network — the "212 weights" headline belongs to the faithful default only.

## Landing accuracy: sampled vs greedy

Training samples actions from the softmax because exploration needs randomness. A deployed
controller takes the `argmax`. The gap between those two is large, and it depends on capacity
in a way that inverts the usual intuition:

| hidden | params | sampled | **greedy** | greedy mean |
|---|---|---|---|---|
| 16 | 212 | 92.0% | **84.5%** | +203 |
| 16 | 212 | 87.7% | 93.1% | +233 |
| 32 | 420 | 91.3% | 97.2% | +256 |
| 32 | 420 | 91.0% | 92.7% | +235 |
| 64 | 836 | 88.0% | **99.3%** | +268 |
| 64 | 836 | 92.7% | **99.4%** | +266 |

*(1,000 fresh evaluation episodes each, after 30k training episodes)*

### Caveat: greedy accuracy oscillates, it does not converge

Those are single measurements at one checkpoint. Tracking the same run over training shows the
greedy landing rate wandering rather than settling:

```
5k:99.8%  10k:99.0%  15k:96.6%  20k:98.4%  25k:98.6%
30k:99.2%  35k:91.2%  40k:98.0%  45k:97.8%  50k:99.2%
```

Vanilla REINFORCE has no mechanism to stabilise a converged policy — the gradient keeps pushing.
**Training longer does not monotonically help**, and where you stop determines what you get.

There is also a real accuracy/efficiency trade-off between checkpoints. Head to head on the same
4,000 fresh episodes:

| checkpoint | landing on pad | crash | mean return |
|---|---|---|---|
| **5k** *(shipped)* | **99.50%** | **0.03%** | +195.8 |
| 50k | 97.35% | 1.32% | +271.9 |

The early policy lands almost every time but burns roughly 3x the fuel; the late one flies
efficiently and crashes ~40x more often. `weights-solve.json` ships the 5k checkpoint because the
goal was landing accuracy — return only breaks ties.

A caution worth recording: picking the best of eight checkpoints on a 1,500-episode sample gave
99.13%, which fell to 98.35% on 4,000 fresh episodes. That gap is selection-on-noise, not a real
difference. Every headline number here is from a sample the selection never saw.

Sampled accuracy is flat across every width (~88-92%). Greedy accuracy climbs steeply with
capacity. **At hidden 16 going deterministic actively loses 7.5 points** — the policy is too
soft to run without noise, and sampling occasionally rescues states it handles badly. The extra
weights do not buy a better stochastic policy; they buy one sharp enough to deploy.

This is why capacity looked irrelevant in training-rate measurements: those measure the wrong
quantity for a deployed controller.

**The residual ~0.5% is irreducible.** Of 6 failures in 1,000, the initial velocities sit at the
very top of the achievable range (median `|vx0|` 0.680, p90 0.786, hard max 0.806) — spawns
thrown groundward at ~3.4 m/s by the `INITIAL_RANDOM` impulse. That is the task's own tail.

## Hyperparameter tuning

25,000 episodes, 3 seeds, escape = first 500-episode window above 50% landing:

| config | escape | landing | mean | AUC |
|---|---|---|---|---|
| **beta=0 + lr 4e-3** | **6,667** | 90% | **+251** | **196** |
| beta=0 + batch16 + lr4e-3 | 9,667 | 91% | +251 | 173 |
| beta=0 | 10,500 | 91% | +250 | 159 |
| HL=500 | 13,000 | 88% | +245 | 150 |
| baseline (HL=2000) | 14,167 | 88% | +244 | 145 |
| beta=0 + batch 16 | 17,167 | 88% | +240 | 106 |

Two results worth noting: removing the entropy bonus entirely beats every decay schedule, and
**larger batches make this worse, not better** (batch 16 pushed escape from 10.5k to 17.2k) —
the opposite of the usual variance-reduction advice.

## Findings

**The 212-weight net does solve LunarLander — it just takes a long time.** It sits at 0%
landing and mean ~+134 for over 10,000 episodes, then escapes:

| episodes | 5k | 10k | 15k | 20k | 40k | 60k |
|---|---|---|---|---|---|---|
| landing rate | 0% | 0% | 85% | 88% | 94% | 90% |

The plateau is a long **metastable phase**, not a capacity ceiling. Widening the hidden layer
moves the escape earlier (32 units → 8–12k) but is not required.

**The entropy bonus is what pins it there.** Escape episode tracks the entropy decay schedule
directly:

| entropy schedule | escape episode | final landing |
|---|---|---|
| β = 0 (no bonus) | **10,488 / 10,667** | — |
| half-life 500 | 12,078 / 12,596 / 13,685 | ~92% |
| half-life 2000 *(default)* | 13,434 / 13,520 / 15,198 | ~91% |
| half-life 8000 | 22,015 / 26,407 / 28,425 | ~91% |
| **β pinned at 0.02, never decays** | **never escapes** | **0%** |

A deliberately stochastic policy cannot execute the precise final touchdown. Hold β high and
the agent hovers forever; let it decay and the policy sharpens and lands.

**Why it hovers in the first place.** Landing is worth ~+260, hovering out the 1000-step clock
~+140, crashing on a landing attempt ~−40. Hovering beats crashing, and the gap between
hovering and landing is a few precisely-timed steps that random exploration rarely finds.

## Known approximations

- **Leg strength (`F_LEG = 65 N`)** is calibrated, not derived. Solving the joint geometry gives
  31.8 N, but a constant-force strut has no equivalent of Box2D's joint *limit*, which supplies
  unbounded torque. 65 N is the value at which Gymnasium's own heuristic reproduces its
  documented landing rate. Exposed as a slider.
- **Sleep tolerance** is relaxed from Box2D's 0.01 m/s to 0.05 m/s and additionally requires both
  legs in contact — a hand-rolled contact solver jitters at 1e-2, and if the lander can never
  fall asleep the +100 is unreachable.
- **Contact solving** uses 8 sequential-impulse iterations. One iteration is not enough: each
  leg's effective mass includes rotational compliance that cancels between the two legs.

## Non-standard options

All off by default; the footer names any that are on. Movable/random landing pad, random start
position, rigid legs, per-step time penalty, hover penalty, hidden width.

Note that `s[0]` is measured **from the pad**, not the viewport centre, so a policy trained on a
centred pad follows a moved one immediately. With the pad centred this is identical to stock.
