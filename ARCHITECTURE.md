# Model architecture

Two small multilayer perceptrons, both written from scratch. No ML library: the matmuls,
the backward pass and the optimiser are all hand-written in `core.js`, and every gradient
is verified against central finite differences (worst relative error **6.7e-9**).

```
                        ┌──────────────────────────┐
   8 observations ─────►│  POLICY (actor)          │──► 4 action probabilities
                        │  8 → H tanh → 4 softmax  │
                        └──────────────────────────┘
                        ┌──────────────────────────┐
   8 observations ─────►│  VALUE (critic baseline) │──► 1 scalar V(s)
                        │  8 → H tanh → 1 linear   │
                        └──────────────────────────┘
```

The two networks are **completely separate** — no shared trunk. With only a few hundred
parameters there is nothing to gain from sharing, and a shared trunk lets the value loss
(which has much larger gradients early on) dominate the policy.

---

## 1. Input — 8 numbers

Exactly Gymnasium's `LunarLander-v3` observation, with one deliberate change (marked ★):

| # | symbol | meaning | formula |
|---|---|---|---|
| 0 | `x` | horizontal offset **from the pad** ★ | `(origin.x − padCentre) / 10` |
| 1 | `y` | height above the pad surface | `(origin.y − 3.9333) / 6.6667` |
| 2 | `vx` | horizontal velocity | `vx · 0.2` |
| 3 | `vy` | vertical velocity | `vy · 0.13333` |
| 4 | `θ` | tilt, radians | `angle` |
| 5 | `ω` | angular velocity | `ω · 0.4` |
| 6 | `leg1` | left leg touching | `0` or `1` |
| 7 | `leg2` | right leg touching | `0` or `1` |

★ Stock LunarLander measures `x` from the **viewport centre**. Because the pad is always
centred there, the two are identical in the default configuration. Measuring from the pad
instead is what makes a movable/random pad learnable at all — otherwise the shaping reward
pulls the lander to the screen centre while the +100 sits somewhere else, and no input tells
the network where the pad is.

No input normalisation layer is needed: every component is already order-1.

## 2. Policy network (actor)

```
z1[k] = Σᵢ W1[k][i]·s[i] + b1[k]          k = 0 … H−1
h[k]  = tanh(z1[k])
z2[j] = Σₖ W2[j][k]·h[k] + b2[j]          j = 0 … 3
p     = softmax(z2)
```

Softmax is computed in log space with the max subtracted, which avoids overflow and gives
`log p` directly for the gradient:

```
m     = maxⱼ z2[j]
lp[j] = z2[j] − m − log Σⱼ' exp(z2[j'] − m)
p[j]  = exp(lp[j])
```

**Parameter count**

| tensor | shape | H = 16 | H = 64 |
|---|---|---|---|
| `W1` | H × 8 | 128 | 512 |
| `b1` | H | 16 | 64 |
| `W2` | 4 × H | 64 | 256 |
| `b2` | 4 | 4 | 4 |
| **total** | | **212** | **836** |

The four outputs are `[idle, fire right thruster, fire main engine, fire left thruster]`.

**Why tanh.** Zero-centred and bounded, so a 2-layer net stays well-conditioned without any
normalisation layer, and its derivative is `1 − h²` — computable from the output alone, with
no need to cache pre-activations.

**Initialisation.** `W1 ~ U(−0.5, 0.5)` (≈ Xavier for fan-in 8, fan-out H), `W2 ~ U(−0.1, 0.1)`,
biases zero. `W2` is deliberately **not** zeroed: that would give a perfectly uniform policy,
but then `dh = W2ᵀ·dz2 = 0` and the first layer receives no gradient on the first update.

## 3. Value network (critic)

```
zv1[k] = Σᵢ Wv1[k][i]·s[i] + bv1[k]
hv[k]  = tanh(zv1[k])
V      = Σₖ Wv2[k]·hv[k] + bv2
```

Linear output (a value can be negative). **161 parameters** at H = 16, **641** at H = 64.
It is only a variance-reduction baseline — it never selects actions.

## 4. Objective

REINFORCE (vanilla policy gradient) with a learned value baseline and an entropy bonus:

```
L = −Aₜ · log π(aₜ | sₜ)  −  β · H(π(·|sₜ))        (policy)
Lᵥ = ½ (V(sₜ) − Ĝₜ)²                                (value)
```

**Returns and advantages.** Rewards are scaled by `0.01` first (raw returns are order 200;
this keeps `V` order 1 without needing to normalise the value targets):

```
Gₜ = 0.01·rₜ + γ·Gₜ₊₁          γ = 0.995
G_T = 0 if terminated, else V(s_T)      ← bootstrap on truncation
Aₜ = Gₜ − V(sₜ)
```

Bootstrapping on truncation is correct, not a hack: `truncated` exists precisely to signal
that the episode was cut off rather than ended.

**Advantages are normalised across the whole batch** — `A ← (A − mean) / (std + 1e−8)`.
With a hand-rolled optimiser this is the difference between training and diverging. The value
targets `Ĝ` are *not* normalised; `REWARD_SCALE` already handles their magnitude.

## 5. Backward pass, hand-derived

**Output layer.** The policy-gradient term for a log-softmax is the standard `p − onehot`.
The entropy term derives as `∂H/∂z_m = −p_m(lp_m + H)`, and since the loss carries `−β·H`
its contribution flips sign:

```
dz2[j] = Aₜ·(p[j] − 1{j = aₜ})  +  β·p[j]·(lp[j] + H)
```

**Everything below it** (accumulated, summed over every step in the batch):

```
gW2[j][k] += dz2[j]·h[k]
gb2[j]    += dz2[j]
dh[k]      = Σⱼ W2[j][k]·dz2[j]
dz1[k]     = dh[k]·(1 − h[k]²)          ← tanh′
gW1[k][i] += dz1[k]·s[i]
gb1[k]    += dz1[k]
```

`W2` must be read before it is updated — automatic here, since Adam runs once per batch.

**Value network**, with `dV = V − Ĝ`:

```
gWv2[k] += dV·hv[k]
gbv2    += dV
dzv1[k]  = Wv2[k]·dV·(1 − hv[k]²)
gWv1[k][i] += dzv1[k]·s[i]
```

## 6. Optimiser — Adam, from scratch

Two independent optimiser states, one per network.

```
g ← g / N_steps_in_batch                       mean over the batch
if ‖g‖₂ > 1.0:  g ← g · (1.0 / ‖g‖₂)           global-norm clip
t ← t + 1
m ← 0.9·m   + 0.1·g
v ← 0.999·v + 0.001·g²
θ ← θ − lr · (m / (1 − 0.9ᵗ)) / ( √(v / (1 − 0.999ᵗ)) + 1e−8 )
```

`0.9ᵗ` and `0.999ᵗ` are kept as running scalars rather than calling `Math.pow` per parameter.

## 7. Training loop

A batch is 8 episodes. Because a batch of 8 × 1000 steps of backward is ~24 ms in one shot —
enough to drop a frame on its own — the trainer is a **resumable state machine** driven from
`requestAnimationFrame` with a time budget:

| phase | work | yields |
|---|---|---|
| `COLLECT` | run episodes, cache `(s, a, r, V)` per step | on budget, and every 200 steps |
| `ADVANTAGE` | discounted returns, advantages, batch normalisation | ~0.1 ms, runs whole |
| `BACKWARD` | accumulate gradients | between episodes |
| `ADAM` | one update | µs |

Because advantages are precomputed, the `BACKWARD` episodes are independent and interruptible.

Cost is about **3.06 µs per environment step** on one CPU core (physics 963 ns, policy forward
752 ns, value forward 603 ns, both backwards 745 ns) — roughly 300,000 steps/second. There is no
GPU path and it would not help: one step is 2,016 FLOPs, while amortising a single kernel launch
needs 50–200 million.

## 8. Action selection — the part that decides accuracy

| mode | rule | used for |
|---|---|---|
| **sampled** | `a ~ p` | training (exploration) |
| **greedy** | `a = argmaxⱼ p[j]` | deployment, and the Evaluate button |

The gap between these is large and **depends on capacity in a way that inverts the usual
intuition**:

| hidden | params | sampled | greedy |
|---|---|---|---|
| 16 | 212 | 92.0% | **84.5%** |
| 32 | 420 | 91.3% | 97.2% |
| 64 | 836 | 88.0% | **99.3%** |

Those figures are single 1,000-episode measurements at one checkpoint. Greedy accuracy
**oscillates over training between roughly 91% and 99.9% rather than converging** — vanilla
REINFORCE has nothing that stabilises a policy once it is good, so the shipped weights are a
selected checkpoint, not an endpoint. See the README for the checkpoint table.

Sampled accuracy is flat across every width. Greedy accuracy climbs steeply. At H = 16 going
deterministic actively *loses* accuracy — the policy is too soft to run without noise, and
sampling occasionally rescues states it handles badly. **The extra weights do not buy a better
stochastic policy; they buy one sharp enough to deploy.**

## 9. Hyperparameters

| | Faithful (default in `core.js`) | Solve (default in the page) |
|---|---|---|
| hidden units | 16 → **212 weights** | 64 → **836 weights** |
| policy lr | 2e-3 | 4e-3 |
| value lr | 5e-3 | 5e-3 |
| discount γ | 0.995 | 0.995 |
| episodes per update | 8 | 8 |
| entropy β₀ | 0.02, halving every 2000 eps, floor 0.001 | **0** |
| gradient clip | 1.0 global L2 | 1.0 |
| reward scale | 0.01 | 0.01 |
| outcome | hovers indefinitely | **99.50% landing, 0.03% crash** (greedy, 5k checkpoint, 4000 episodes) |

γ = 0.995 rather than 0.99 for a concrete reason: at γ = 0.99 a +100 landing bonus 500 steps in
the future is worth `100 × 0.99⁵⁰⁰ = 0.66` — effectively invisible from altitude. At 0.995 it is
worth 8.2.

The entropy bonus turned out to be what pins the agent in the hovering plateau. Escape episode
tracks the decay schedule directly, and pinning β at 0.02 prevents escape entirely across every
seed tested — a deliberately stochastic policy cannot execute the precise final touchdown.
