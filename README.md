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
