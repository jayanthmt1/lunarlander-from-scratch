'use strict';
/* =============================================================================
 * LunarLander REINFORCE — training core.  NO DOM REFERENCES IN THIS FILE.
 * Sections: 1 rng/helpers  2 LunarEnv  3 Nets  4 Adam  5 Trainer  6 GhostBuffer
 * ========================================================================== */

/* ---------- 1. RNG + helpers ---------------------------------------------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
class RNG {
  constructor(seed) { this.seed = seed >>> 0; this.next = mulberry32(this.seed); }
  uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
  reset(seed) { this.seed = seed >>> 0; this.next = mulberry32(this.seed); }
}

/* =============================================================================
 * 2. LunarEnv — Gymnasium LunarLander-v3 reimplemented without Box2D.
 * Constants taken from gymnasium/envs/box2d/lunar_lander.py.
 * ========================================================================== */
const SCALE = 30.0, FPS = 50, DT = 1 / FPS;
const VIEWPORT_W = 600, VIEWPORT_H = 400;
const W = VIEWPORT_W / SCALE;          // 20.0 m
const H = VIEWPORT_H / SCALE;          // 13.3333 m
const GRAVITY = -10.0;
const MAIN_ENGINE_POWER = 13.0, SIDE_ENGINE_POWER = 0.6;
const MAIN_ENGINE_Y_LOCATION = 4, SIDE_ENGINE_AWAY = 12, SIDE_ENGINE_HEIGHT = 14;
const INITIAL_RANDOM = 1000.0;
const LEG_DOWN = 18;
const CHUNKS = 11;

// Lumped rigid body (derived from LANDER_POLY @ density 5 + 2 legs @ density 1)
const MASS = 4.9589;            // kg
const INERTIA = 0.85;           // kg m^2 about COM
const COM_OFF = 0.09840;        // COM sits this far ABOVE the body origin (local +y)

const LANDER_POLY = [[-14,17],[-17,0],[-17,-10],[17,-10],[17,0],[14,17]]
  .map(p => [p[0] / SCALE, p[1] / SCALE]);

const HELIPAD_Y = H / 4;                          // 3.33333
const OBS_Y_REF = HELIPAD_Y + LEG_DOWN / SCALE;   // 3.93333

// Leg strut geometry (revolute joint solved to a point-foot)
const PHI_MIN = 0.4, PHI_MAX = 0.9, PHI_SPEED = 0.3;
const DYDPHI = 0.95154;         // dy/dphi at phi = PHI_MIN
const MU = 0.14;                // mixed friction, legs vs moon
const BAUMGARTE = 0.2, SLOP = 0.005;
const CONTACT_ITERS = 8;

// Sleep detection (relaxed vs Box2D's 0.01 — a hand-rolled solver jitters at 1e-2)
const LIN_TOL = 0.05, ANG_TOL = 0.1, SLEEP_TIME = 0.5;

const MAX_VEL = 100.0, MAX_OMEGA = 0.5 / DT;

function legFootLocal(i, phi) {   // relative to BODY ORIGIN, in body-local frame
  const c = Math.cos(phi), s = Math.sin(phi);
  return [ -i * (0.66667 * c + 0.86667 * s),
           -(-0.66667 * s + 0.86667 * c) ];
}

class LunarEnv {
  constructor(rng, cfg) { this.rng = rng; this.cfg = cfg; this.reset(); }

  generateTerrain() {
    const h = new Float64Array(CHUNKS + 1);
    for (let i = 0; i <= CHUNKS; i++) h[i] = this.rng.uniform(0, H / 2);
    this.chunkX = new Float64Array(CHUNKS);
    for (let i = 0; i < CHUNKS; i++) this.chunkX[i] = (W / (CHUNKS - 1)) * i;
    const cf = this.cfg || {};
    let pc = cf.padChunk === undefined ? (CHUNKS >> 1) : Math.round(cf.padChunk);
    if (cf.padRandom) pc = 2 + Math.floor(this.rng.uniform(0, CHUNKS - 4));  // 2..8
    pc = Math.max(2, Math.min(CHUNKS - 3, pc));      // keep the 5-chunk flat span in range
    this.padChunk = pc;
    this.helipadX1 = this.chunkX[pc - 1];            // 8.0 when centred
    this.helipadX2 = this.chunkX[pc + 1];            // 12.0 when centred
    this.padCenter = this.chunkX[pc];                // 10.0 when centred
    for (let i = pc - 2; i <= pc + 2; i++) h[i] = HELIPAD_Y;
    this.smoothY = new Float64Array(CHUNKS);
    for (let i = 0; i < CHUNKS; i++) {
      const prev = h[i === 0 ? CHUNKS : i - 1];        // python height[-1] wraps
      this.smoothY[i] = 0.33 * (prev + h[i] + h[i + 1]);
    }
  }

  terrainAt(x) {  // returns [height, normal_x, normal_y]
    const step = W / (CHUNKS - 1);
    let xc = x; if (xc < 0) xc = 0; if (xc > W) xc = W;
    let i = Math.floor(xc / step); if (i > CHUNKS - 2) i = CHUNKS - 2; if (i < 0) i = 0;
    const x1 = this.chunkX[i], x2 = this.chunkX[i + 1];
    const y1 = this.smoothY[i], y2 = this.smoothY[i + 1];
    const t = (xc - x1) / (x2 - x1);
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
    return [y1 + t * dy, -dy / L, dx / L];
  }

  // origin = com + R(theta) . (0, -COM_OFF)
  origin() {
    const c = Math.cos(this.theta), s = Math.sin(this.theta);
    return [ this.cx + s * COM_OFF, this.cy - c * COM_OFF ];
  }

  reset() {
    this.generateTerrain();
    // body origin starts at (W/2, H); com = origin + R(0).(0, +COM_OFF).
    // startRandom draws no RNG when off, so stock runs stay bit-identical.
    const cfr = this.cfg || {};
    if (cfr.startRandom) {
      this.cx = this.rng.uniform(0.12 * W, 0.88 * W);
      this.cy = this.rng.uniform(0.75 * H, H) + COM_OFF;
    } else {
      this.cx = W / 2; this.cy = H + COM_OFF;
    }
    this.vx = 0; this.vy = 0; this.theta = 0; this.omega = 0;
    this.phi = [PHI_MIN, PHI_MIN];
    this.contact = [0, 0];
    this.sleepTimer = 0;
    this.gameOver = false;
    this.prevShaping = null;
    this.t = 0;
    // INITIAL_RANDOM: a force applied for exactly one step
    this.vx += this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM) * DT / MASS;
    this.vy += this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM) * DT / MASS;
    // gymnasium's reset() returns self.step(0)[0]
    return this.step(0).obs;
  }

  observe() {
    const [ox, oy] = this.origin();
    return [
      (ox - this.padCenter) / (W / 2),   // pad-relative; == stock when padCenter is 10
      (oy - OBS_Y_REF) / (H / 2),
      this.vx * (W / 2) / FPS,
      this.vy * (H / 2) / FPS,
      this.theta,
      20.0 * this.omega / FPS,
      this.contact[0], this.contact[1],
    ];
  }

  hullTouchesGround() {
    const c = Math.cos(this.theta), s = Math.sin(this.theta);
    const [ox, oy] = this.origin();
    for (let k = 0; k < LANDER_POLY.length; k++) {
      const lx = LANDER_POLY[k][0], ly = LANDER_POLY[k][1];
      const wx = ox + lx * c - ly * s, wy = oy + lx * s + ly * c;
      if (wy < this.terrainAt(wx)[0]) return true;
    }
    return false;
  }

  step(action) {
    const rng = this.rng, cfg = this.cfg;
    const d0 = rng.uniform(-1, 1) / SCALE, d1 = rng.uniform(-1, 1) / SCALE;
    const tipx = Math.sin(this.theta), tipy = Math.cos(this.theta);
    const sidex = -tipy, sidey = tipx;
    const [orx, ory] = this.origin();

    let Jx = 0, Jy = 0, Jw = 0, mPower = 0, sPower = 0;
    const push = (ix, iy, px, py) => {
      Jx += ix; Jy += iy;
      Jw += (px - this.cx) * iy - (py - this.cy) * ix;
    };

    if (action === 2) {
      mPower = 1.0;
      const ox = tipx * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * d0) + sidex * d1;
      const oy = -tipy * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * d0) - sidey * d1;
      push(-ox * MAIN_ENGINE_POWER * mPower, -oy * MAIN_ENGINE_POWER * mPower,
           orx + ox, ory + oy);
    } else if (action === 1 || action === 3) {
      sPower = 1.0;
      const dir = action - 2;
      const b = 3 * d1 + dir * SIDE_ENGINE_AWAY / SCALE;
      const ox = tipx * d0 + sidex * b;
      const oy = -tipy * d0 - sidey * b;
      // NOTE: 17 (not SIDE_ENGINE_HEIGHT=14) — reproduces gymnasium's own bug
      push(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower,
           orx + ox - tipx * 17 / SCALE, ory + oy + tipy * SIDE_ENGINE_HEIGHT / SCALE);
    }

    Jy += GRAVITY * MASS * DT;

    this.vx += Jx / MASS; this.vy += Jy / MASS; this.omega += Jw / INERTIA;

    // ---- leg contacts: iterative sequential impulses ----------------------
    // One iteration is not enough: each leg's effective mass includes rotational
    // compliance that actually cancels between the two legs, so a single pass
    // removes only ~1/3 of the impact velocity and the struts ratchet down.
    const c = Math.cos(this.theta), s = Math.sin(this.theta);
    const F_LEG = cfg.fLeg;
    const legs = [];
    for (let li = 0; li < 2; li++) {
      const iSign = li === 0 ? 1 : -1;
      const fl = legFootLocal(iSign, this.phi[li]);
      const px = orx + fl[0] * c - fl[1] * s, py = ory + fl[0] * s + fl[1] * c;
      const [gh, nx, ny] = this.terrainAt(px);
      const d = gh - py;
      if (d <= 0) {
        this.contact[li] = 0;
        this.phi[li] = Math.max(PHI_MIN, this.phi[li] - PHI_SPEED * DT);
        continue;
      }
      this.contact[li] = 1;
      const rx = px - this.cx, ry = py - this.cy;
      const tx = ny, ty = -nx;
      const rn = rx * ny - ry * nx, rt = rx * ty - ry * tx;
      legs.push({
        li, rx, ry, nx, ny, tx, ty, d,
        mN: 1 / (1 / MASS + rn * rn / INERTIA),
        mT: 1 / (1 / MASS + rt * rt / INERTIA),
        bias: BAUMGARTE * Math.max(0, d - SLOP) / DT,
        jCap: F_LEG * DT,
        rigid: !!cfg.stiffLegs || this.phi[li] >= PHI_MAX,
        an: 0, at: 0,
      });
    }
    for (let it = 0; it < CONTACT_ITERS; it++) {
      for (let q = 0; q < legs.length; q++) {
        const L = legs[q];
        const ux = this.vx - this.omega * L.ry, uy = this.vy + this.omega * L.rx;
        const vn = ux * L.nx + uy * L.ny;
        let na = L.an + L.mN * (-vn + L.bias);
        if (na < 0) na = 0;
        if (!L.rigid && na > L.jCap) na = L.jCap;
        const dJ = na - L.an; L.an = na;
        this.vx += dJ * L.nx / MASS; this.vy += dJ * L.ny / MASS;
        this.omega += (L.rx * dJ * L.ny - L.ry * dJ * L.nx) / INERTIA;
      }
      for (let q = 0; q < legs.length; q++) {
        const L = legs[q];
        const ux = this.vx - this.omega * L.ry, uy = this.vy + this.omega * L.rx;
        const vt = ux * L.tx + uy * L.ty;
        const lim = MU * L.an;
        let na = L.at - L.mT * vt;
        if (na > lim) na = lim; if (na < -lim) na = -lim;
        const dJ = na - L.at; L.at = na;
        this.vx += dJ * L.tx / MASS; this.vy += dJ * L.ty / MASS;
        this.omega += (L.rx * dJ * L.ty - L.ry * dJ * L.tx) / INERTIA;
      }
    }
    // a strut only collapses once its force capacity actually saturated
    if (!cfg.stiffLegs) {
      for (let q = 0; q < legs.length; q++) {
        const L = legs[q];
        if (!L.rigid && L.an >= L.jCap - 1e-12)
          this.phi[L.li] = Math.min(PHI_MAX, this.phi[L.li] + L.d / DYDPHI);
      }
    }

    // ---- clamp + integrate -------------------------------------------------
    const sp = Math.hypot(this.vx, this.vy);
    if (sp > MAX_VEL) { this.vx *= MAX_VEL / sp; this.vy *= MAX_VEL / sp; }
    if (this.omega > MAX_OMEGA) this.omega = MAX_OMEGA;
    if (this.omega < -MAX_OMEGA) this.omega = -MAX_OMEGA;
    this.cx += this.vx * DT; this.cy += this.vy * DT; this.theta += this.omega * DT;
    this.t++;

    if (this.hullTouchesGround()) this.gameOver = true;

    // ---- sleep -------------------------------------------------------------
    if (mPower > 0 || sPower > 0) this.sleepTimer = 0;
    else if (Math.hypot(this.vx, this.vy) < LIN_TOL && Math.abs(this.omega) < ANG_TOL
             && this.contact[0] && this.contact[1]) this.sleepTimer += DT;
    else this.sleepTimer = 0;
    const asleep = this.sleepTimer >= SLEEP_TIME;

    // ---- observation + reward ---------------------------------------------
    const obs = this.observe();
    const shaping =
      -100 * Math.sqrt(obs[0] * obs[0] + obs[1] * obs[1])
      -100 * Math.sqrt(obs[2] * obs[2] + obs[3] * obs[3])
      -100 * Math.abs(obs[4]) + 10 * obs[6] + 10 * obs[7];
    let reward = 0;
    if (this.prevShaping !== null) reward = shaping - this.prevShaping;
    this.prevShaping = shaping;
    reward -= 0.30 * mPower;
    reward -= 0.03 * sPower;

    let terminated = false, outcome = -1;
    // |s0| >= 1 is exactly "x outside [0, W]" when the pad is centred; expressing it
    // in absolute terms keeps the playfield fixed when the pad moves.
    const oxEnd = this.origin()[0];
    if (this.gameOver || oxEnd <= 0 || oxEnd >= W) { terminated = true; reward = -100; outcome = 0; }
    else if (asleep) { terminated = true; reward = +100; outcome = 1; }

    // rawReward is always the true Gymnasium reward, so the displayed return and
    // the "+200 = SOLVED" line stay meaningful even with a shaping hatch enabled.
    const rawReward = reward;
    // non-Gymnasium escape hatches, off by default
    if (!terminated && cfg.timePenalty) reward -= cfg.timePenalty;

    return { obs, reward, rawReward, terminated, outcome, mPower, sPower };
  }
}

/* =============================================================================
 * 3. Networks — hand-written forward and hand-derived backward.
 * Policy 8 -> Hn tanh -> 4 softmax   (Hn=16 => 212 params)
 * Value  8 -> Hn tanh -> 1           (Hn=16 => 161 params)
 * ========================================================================== */
const NIN = 8, NOUT = 4;

function tanhArr(z, out, n) { for (let i = 0; i < n; i++) out[i] = Math.tanh(z[i]); }

class PolicyNet {
  constructor(hn, rng) {
    this.hn = hn;
    this.W1 = new Float64Array(hn * NIN); this.b1 = new Float64Array(hn);
    this.W2 = new Float64Array(NOUT * hn); this.b2 = new Float64Array(NOUT);
    for (let i = 0; i < this.W1.length; i++) this.W1[i] = rng.uniform(-0.5, 0.5);
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = rng.uniform(-0.1, 0.1);
    this.h = new Float64Array(hn); this.p = new Float64Array(NOUT);
    this.lp = new Float64Array(NOUT);
    this.gW1 = new Float64Array(hn * NIN); this.gb1 = new Float64Array(hn);
    this.gW2 = new Float64Array(NOUT * hn); this.gb2 = new Float64Array(NOUT);
  }
  get nParams() { return this.W1.length + this.b1.length + this.W2.length + this.b2.length; }
  tensors() { return [this.W1, this.b1, this.W2, this.b2]; }
  grads() { return [this.gW1, this.gb1, this.gW2, this.gb2]; }
  zeroGrad() { for (const g of this.grads()) g.fill(0); }

  forward(s) {
    const hn = this.hn, h = this.h, W1 = this.W1, b1 = this.b1;
    for (let k = 0; k < hn; k++) {
      let z = b1[k], o = k * NIN;
      for (let i = 0; i < NIN; i++) z += W1[o + i] * s[i];
      h[k] = Math.tanh(z);
    }
    const p = this.p, lp = this.lp, W2 = this.W2, b2 = this.b2;
    let mx = -Infinity;
    for (let j = 0; j < NOUT; j++) {
      let z = b2[j], o = j * hn;
      for (let k = 0; k < hn; k++) z += W2[o + k] * h[k];
      p[j] = z; if (z > mx) mx = z;
    }
    let S = 0;
    for (let j = 0; j < NOUT; j++) { const e = Math.exp(p[j] - mx); lp[j] = p[j] - mx; p[j] = e; S += e; }
    const logS = Math.log(S);
    let ent = 0;
    for (let j = 0; j < NOUT; j++) { p[j] /= S; lp[j] -= logS; ent -= p[j] * lp[j]; }
    this.entropy = ent;
    return p;
  }

  // Deployment picks the argmax; sampling exists for exploration during training.
  // The gap is large and capacity-dependent: at hidden 16 going greedy LOSES
  // accuracy, at hidden 64 it gains ~11 points.
  greedy() {
    let b = 0;
    for (let j = 1; j < NOUT; j++) if (this.p[j] > this.p[b]) b = j;
    return b;
  }
  sample(rng) {
    const u = rng.next(); let acc = 0;
    for (let j = 0; j < NOUT; j++) { acc += this.p[j]; if (u < acc) return j; }
    return NOUT - 1;
  }

  // Accumulates grads of  L = -A*lp[a] - beta*entropy   (forward() must be current)
  backward(s, a, A, beta) {
    const hn = this.hn, h = this.h, p = this.p, lp = this.lp, ent = this.entropy;
    const dz2 = new Float64Array(NOUT);
    for (let j = 0; j < NOUT; j++) {
      dz2[j] = A * (p[j] - (j === a ? 1 : 0)) + beta * p[j] * (lp[j] + ent);
    }
    for (let j = 0; j < NOUT; j++) {
      const o = j * hn, d = dz2[j];
      for (let k = 0; k < hn; k++) this.gW2[o + k] += d * h[k];
      this.gb2[j] += d;
    }
    for (let k = 0; k < hn; k++) {
      let dh = 0;
      for (let j = 0; j < NOUT; j++) dh += this.W2[j * hn + k] * dz2[j];
      const dz1 = dh * (1 - h[k] * h[k]);
      const o = k * NIN;
      for (let i = 0; i < NIN; i++) this.gW1[o + i] += dz1 * s[i];
      this.gb1[k] += dz1;
    }
  }
}

class ValueNet {
  constructor(hn, rng) {
    this.hn = hn;
    this.W1 = new Float64Array(hn * NIN); this.b1 = new Float64Array(hn);
    this.W2 = new Float64Array(hn); this.b2 = new Float64Array(1);
    for (let i = 0; i < this.W1.length; i++) this.W1[i] = rng.uniform(-0.5, 0.5);
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = rng.uniform(-0.1, 0.1);
    this.h = new Float64Array(hn);
    this.gW1 = new Float64Array(hn * NIN); this.gb1 = new Float64Array(hn);
    this.gW2 = new Float64Array(hn); this.gb2 = new Float64Array(1);
  }
  get nParams() { return this.W1.length + this.b1.length + this.W2.length + 1; }
  tensors() { return [this.W1, this.b1, this.W2, this.b2]; }
  grads() { return [this.gW1, this.gb1, this.gW2, this.gb2]; }
  zeroGrad() { for (const g of this.grads()) g.fill(0); }

  forward(s) {
    const hn = this.hn, h = this.h;
    for (let k = 0; k < hn; k++) {
      let z = this.b1[k], o = k * NIN;
      for (let i = 0; i < NIN; i++) z += this.W1[o + i] * s[i];
      h[k] = Math.tanh(z);
    }
    let v = this.b2[0];
    for (let k = 0; k < hn; k++) v += this.W2[k] * h[k];
    this.v = v;
    return v;
  }
  // L = 0.5*(V - target)^2
  backward(s, target) {
    const hn = this.hn, h = this.h, dV = this.v - target;
    for (let k = 0; k < hn; k++) this.gW2[k] += dV * h[k];
    this.gb2[0] += dV;
    for (let k = 0; k < hn; k++) {
      const dz1 = this.W2[k] * dV * (1 - h[k] * h[k]);
      const o = k * NIN;
      for (let i = 0; i < NIN; i++) this.gW1[o + i] += dz1 * s[i];
      this.gb1[k] += dz1;
    }
  }
}

/* ---------- 4. Adam --------------------------------------------------------- */
class Adam {
  constructor(net, lr, clip) {
    this.net = net; this.lr = lr; this.clip = clip;
    this.b1 = 0.9; this.b2 = 0.999; this.eps = 1e-8;
    this.b1t = 1; this.b2t = 1;
    this.m = net.tensors().map(t => new Float64Array(t.length));
    this.v = net.tensors().map(t => new Float64Array(t.length));
  }
  step(scale) {
    const T = this.net.tensors(), G = this.net.grads();
    let sq = 0;
    for (let i = 0; i < G.length; i++) for (let j = 0; j < G[i].length; j++) {
      G[i][j] *= scale; sq += G[i][j] * G[i][j];
    }
    const gn = Math.sqrt(sq);
    const cs = (this.clip > 0 && gn > this.clip) ? this.clip / gn : 1;
    this.b1t *= this.b1; this.b2t *= this.b2;
    const c1 = 1 - this.b1t, c2 = 1 - this.b2t;
    for (let i = 0; i < T.length; i++) {
      const t = T[i], g = G[i], m = this.m[i], v = this.v[i];
      for (let j = 0; j < t.length; j++) {
        const gj = g[j] * cs;
        m[j] = this.b1 * m[j] + (1 - this.b1) * gj;
        v[j] = this.b2 * v[j] + (1 - this.b2) * gj * gj;
        t[j] -= this.lr * (m[j] / c1) / (Math.sqrt(v[j] / c2) + this.eps);
      }
    }
    return gn;
  }
}


/* =============================================================================
 * 5. Trainer — resumable state machine so no single frame is ever blocked.
 *    COLLECT -> ADVANTAGE -> BACKWARD -> ADAM
 * ========================================================================== */
const DEFAULTS = {
  hidden: 16, gamma: 0.995, batch: 8,
  lrPolicy: 2e-3, lrValue: 5e-3, clip: 1.0,
  entropy0: 0.02, entropyHalfLife: 2000, entropyMin: 0.001,
  rewardScale: 0.01,
  fLeg: 65.0,
  padChunk: 5, padRandom: 0,      // 5 = centred = stock LunarLander
  startRandom: 0, stiffLegs: 0,   // both off = stock LunarLander
  timePenalty: 0, truncPenalty: 0,     // non-Gymnasium escape hatches, off
  seed: 1,
};

function maxStepsFor(ep) {
  if (ep < 1500) return 400;
  if (ep >= 3000) return 1000;
  return Math.round(400 + 600 * (ep - 1500) / 1500);
}

/* ---------- 6. GhostBuffer: fixed memory, any episode count ---------------- */
const GHOST_SLOTS = 64, GHOST_PTS = 256, CURVE_CAP = 16384;
const HIST_BINS = 1024;
// The CURVE_CAP ring only holds recent episodes, so a chart drawn straight from it
// silently repeats recent data once training passes 16,384 episodes. This keeps the
// WHOLE history in fixed memory by halving the bin resolution whenever it fills.
class CurveHistory {
  constructor() {
    this.n = 0; this.binSize = 1; this.bins = 0;
    this.sum = new Float64Array(HIST_BINS); this.cnt = new Int32Array(HIST_BINS);
    this.mn = new Float32Array(HIST_BINS); this.mx = new Float32Array(HIST_BINS);
    this.land = new Int32Array(HIST_BINS);
  }
  push(ret, outcome) {
    let b = (this.n / this.binSize) | 0;
    if (b >= HIST_BINS) { this.compact(); b = (this.n / this.binSize) | 0; }
    if (this.cnt[b] === 0) { this.mn[b] = ret; this.mx[b] = ret; }
    else { if (ret < this.mn[b]) this.mn[b] = ret; if (ret > this.mx[b]) this.mx[b] = ret; }
    this.sum[b] += ret; this.cnt[b]++; if (outcome === 1) this.land[b]++;
    this.n++; this.bins = b + 1;
  }
  compact() {
    const h = HIST_BINS >> 1;
    for (let i = 0; i < h; i++) {
      const a = 2 * i, c = 2 * i + 1, ca = this.cnt[a], cc = this.cnt[c];
      this.sum[i] = this.sum[a] + this.sum[c];
      this.cnt[i] = ca + cc;
      if (ca && cc) { this.mn[i] = Math.min(this.mn[a], this.mn[c]); this.mx[i] = Math.max(this.mx[a], this.mx[c]); }
      else if (ca) { this.mn[i] = this.mn[a]; this.mx[i] = this.mx[a]; }
      else { this.mn[i] = this.mn[c]; this.mx[i] = this.mx[c]; }
      this.land[i] = this.land[a] + this.land[c];
    }
    for (let i = h; i < HIST_BINS; i++) { this.sum[i] = 0; this.cnt[i] = 0; this.mn[i] = 0; this.mx[i] = 0; this.land[i] = 0; }
    this.binSize *= 2; this.bins = h;
  }
}
class GhostBuffer {
  constructor() {
    this.xy = new Float32Array(GHOST_SLOTS * GHOST_PTS * 3);
    this.len = new Int32Array(GHOST_SLOTS);
    this.stride = new Int32Array(GHOST_SLOTS);
    this.meta = new Float32Array(GHOST_SLOTS * 3);   // return, outcome, epIdx
    this.phase = 0; this.slot = 0;
    this.ret = new Float32Array(CURVE_CAP);
    this.steps = new Float32Array(CURVE_CAP);
    this.outcome = new Float32Array(CURVE_CAP);
    this.hist = new CurveHistory();
    this.n = 0;
  }
  begin(ep) { this.slot = ep % GHOST_SLOTS; this.len[this.slot] = 0; this.stride[this.slot] = 1; this.phase = 0; }
  point(x, y, th) {
    const s = this.slot;
    if (this.phase++ % this.stride[s] !== 0) return;
    const base = s * GHOST_PTS * 3;
    if (this.len[s] === GHOST_PTS) {                 // halve in place
      for (let n = 0; n < GHOST_PTS >> 1; n++) {
        this.xy[base + 3 * n] = this.xy[base + 6 * n];
        this.xy[base + 3 * n + 1] = this.xy[base + 6 * n + 1];
        this.xy[base + 3 * n + 2] = this.xy[base + 6 * n + 2];
      }
      this.len[s] = GHOST_PTS >> 1; this.stride[s] *= 2; this.phase = 0;
    }
    const o = base + 3 * this.len[s]++;
    this.xy[o] = x; this.xy[o + 1] = y; this.xy[o + 2] = th;
  }
  end(ep, ret, outcome, steps) {
    const s = this.slot;
    this.meta[s * 3] = ret; this.meta[s * 3 + 1] = outcome; this.meta[s * 3 + 2] = ep;
    const k = ep % CURVE_CAP;
    this.ret[k] = ret; this.steps[k] = steps; this.outcome[k] = outcome;
    this.hist.push(ret, outcome);
    this.n = ep + 1;
  }
}

// Evaluate a policy on fresh episodes. Greedy by default: this is the number
// that describes a deployed controller, not the exploration-noisy training rate.
function evaluatePolicy(pol, cfg, n, seed, greedy) {
  if (greedy === undefined) greedy = true;
  let land = 0, crash = 0, trunc = 0, sum = 0;
  for (let e = 0; e < n; e++) {
    const rng = new RNG((seed * 7919 + e * 104729) >>> 0);
    const env = new LunarEnv(rng, cfg);
    let s = env.observe(), R = 0, done = false;
    for (let t = 0; t < 1000 && !done; t++) {
      pol.forward(s);
      const a = greedy ? pol.greedy() : pol.sample(rng);
      const r = env.step(a);
      s = r.obs; R += r.rawReward;
      if (r.terminated) { done = true; if (r.outcome === 1) land++; else crash++; }
    }
    if (!done) trunc++;
    sum += R;
  }
  return { land: 100 * land / n, crash: 100 * crash / n, trunc: 100 * trunc / n, mean: sum / n, n };
}

// Resumable greedy evaluator. Runs episodes in slices so a 200-episode screen
// (~50ms in one shot) never blocks a frame.
class Evaluator {
  constructor() { this.active = false; this.result = null; this.pol = null; }
  // weights: array of Float64Array matching PolicyNet.tensors()
  start(weights, cfg, n, seedBase, tag) {
    this.pol = new PolicyNet(cfg.hidden, new RNG(1));
    const dst = this.pol.tensors();
    for (let i = 0; i < dst.length; i++) dst[i].set(weights[i]);
    this.cfg = cfg; this.n = n; this.seedBase = seedBase; this.tag = tag;
    this.i = 0; this.onpad = 0; this.crash = 0; this.trunc = 0; this.sum = 0;
    this.active = true; this.result = null;
  }
  weights() { return this.pol.tensors().map(a => Float64Array.from(a)); }
  // returns true when the whole evaluation is finished
  tick(budgetMs) {
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const t0 = now();
    while (this.i < this.n) {
      const rng = new RNG((this.seedBase + this.i * 7919) >>> 0);
      const env = new LunarEnv(rng, this.cfg);
      let s = env.observe(), R = 0, done = false;
      for (let t = 0; t < 1000 && !done; t++) {
        this.pol.forward(s);
        const r = env.step(this.pol.greedy());
        s = r.obs; R += r.rawReward;
        if (r.terminated) {
          done = true;
          if (r.outcome === 1) {
            const x = env.origin()[0];
            // "landed" must mean between the flags, not merely come to rest
            if (x >= env.helipadX1 && x <= env.helipadX2) this.onpad++;
          } else this.crash++;
        }
      }
      if (!done) this.trunc++;
      this.sum += R; this.i++;
      if (now() - t0 > budgetMs) return false;
    }
    this.active = false;
    this.result = {
      land: 100 * this.onpad / this.n, crash: 100 * this.crash / this.n,
      trunc: 100 * this.trunc / this.n, mean: this.sum / this.n, n: this.n, tag: this.tag,
    };
    return true;
  }
  get progress() { return this.n ? this.i / this.n : 0; }
}

// Presets. FAITHFUL reproduces the hovering plateau (the default story);
// SOLVE is tuned for landing accuracy and is a different, larger network.
const PRESETS = {
  faithful: { hidden: 16, entropy0: 0.02, entropyHalfLife: 2000, entropyMin: 0.001,
              lrPolicy: 2e-3, lrValue: 5e-3, batch: 8, gamma: 0.995 },
  solve:    { hidden: 64, entropy0: 0, entropyHalfLife: 2000, entropyMin: 0,
              lrPolicy: 4e-3, lrValue: 5e-3, batch: 8, gamma: 0.995 },
};

class Trainer {
  constructor(cfg) {
    this.cfg = Object.assign({}, DEFAULTS, cfg || {});
    this.reset();
  }
  reset(newSeed) {
    const c = this.cfg;
    if (newSeed !== undefined) c.seed = newSeed;
    this.rng = new RNG(c.seed);
    this.netRng = new RNG(c.seed ^ 0x9e3779b9);
    this.pol = new PolicyNet(c.hidden, this.netRng);
    this.val = new ValueNet(c.hidden, this.netRng);
    this.polOpt = new Adam(this.pol, c.lrPolicy, c.clip);
    this.valOpt = new Adam(this.val, c.lrValue, c.clip);
    this.env = new LunarEnv(this.rng, c);
    this.ghost = new GhostBuffer();
    const cap = 32 * 1000;
    this.obs = new Float64Array(cap * NIN);
    this.act = new Int8Array(cap);
    this.rew = new Float64Array(cap);
    this.vpr = new Float64Array(cap);
    this.adv = new Float64Array(cap);
    this.ret = new Float64Array(cap);
    this.cap = cap;
    this.episode = 0; this.updates = 0; this.nSteps = 0;
    this.eps = [];                 // {start,end,terminated,finalObs}
    this.phase = 'COLLECT';
    this.cur = null;
    this.bestReturn = -Infinity; this.bestWeights = null;
    this.recent = [];              // last 50 returns
    this.best50 = -Infinity;
    this.liveObs = new Float64Array(NIN);
    this.liveP = new Float64Array(NOUT);
    this.lastReturn = 0;
    this.liveAction = 0;
    this.onEpisode = null;
    this.startEpisode();
  }
  get beta() {
    const c = this.cfg;
    return Math.max(c.entropyMin, c.entropy0 * Math.pow(0.5, this.episode / c.entropyHalfLife));
  }
  startEpisode() {
    const o = this.env.reset();
    this.cur = { start: this.nSteps, R: 0, t: 0, obs: o };
    this.ghost.begin(this.episode);
    const or = this.env.origin();
    this.ghost.point(or[0], or[1], this.env.theta);
  }
  // Runs one COLLECT/ADVANTAGE/BACKWARD/ADAM slice. Returns steps simulated.
  tick(budgetMs, stepLimit) {
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    let did = 0;
    if (stepLimit === undefined) stepLimit = Infinity;
    while (true) {
      if (this.phase === 'COLLECT') {
        const c = this.cfg;
        let guard = 0;
        while (true) {
          if (did >= stepLimit) return did;
          // NB: read this.cur / maxT fresh each step — startEpisode() replaces
          // this.cur, so hoisting either out of the loop corrupts episode bounds.
          const cur = this.cur, maxT = maxStepsFor(this.episode);
          const s = cur.obs;
          const i = this.nSteps * NIN;
          for (let k = 0; k < NIN; k++) this.obs[i + k] = s[k];
          this.pol.forward(s);
          for (let k = 0; k < NOUT; k++) this.liveP[k] = this.pol.p[k];
          for (let k = 0; k < NIN; k++) this.liveObs[k] = s[k];
          const a = this.pol.sample(this.rng);
          this.liveAction = a;
          this.vpr[this.nSteps] = this.val.forward(s);
          const r = this.env.step(a);
          this.act[this.nSteps] = a;
          this.rew[this.nSteps] = r.reward;
          this.nSteps++; cur.t++; cur.R += r.rawReward; cur.obs = r.obs;
          did++; guard++;
          const or = this.env.origin();
          this.ghost.point(or[0], or[1], this.env.theta);
          const truncated = !r.terminated && cur.t >= maxT;
          if (r.terminated || truncated || this.nSteps >= this.cap - 2) {
            // penalties shape learning only; cur.R stays the true Gymnasium return
            if (truncated && c.truncPenalty) this.rew[this.nSteps - 1] -= c.truncPenalty;
            const outcome = r.terminated ? (r.outcome === 1 ? 1 : 0) : 2;
            this.eps.push({ start: cur.start, end: this.nSteps, terminated: r.terminated, finalObs: r.obs });
            this.ghost.end(this.episode, cur.R, outcome, cur.t);
            this.lastReturn = cur.R;
            this.recent.push(cur.R); if (this.recent.length > 50) this.recent.shift();
            if (this.recent.length === 50) {
              const m = this.recent.reduce((x, y) => x + y, 0) / 50;
              if (m > this.best50) this.best50 = m;
            }
            if (cur.R > this.bestReturn) { this.bestReturn = cur.R; this.bestWeights = this.snapshot(); }
            this.episode++;
            if (this.onEpisode) this.onEpisode(cur.R, outcome, cur.t);
            if (this.eps.length >= c.batch) { this.phase = 'ADVANTAGE'; break; }
            this.startEpisode();
            if (now() - t0 > budgetMs) return did;
          }
          if (guard >= 200) { guard = 0; if (now() - t0 > budgetMs) return did; }
        }
      }
      if (this.phase === 'ADVANTAGE') {
        const c = this.cfg;
        for (const e of this.eps) {
          let G = e.terminated ? 0 : this.val.forward(e.finalObs);
          for (let t = e.end - 1; t >= e.start; t--) {
            G = c.rewardScale * this.rew[t] + c.gamma * G;
            this.ret[t] = G;
            this.adv[t] = G - this.vpr[t];
          }
        }
        let m = 0; for (let t = 0; t < this.nSteps; t++) m += this.adv[t];
        m /= this.nSteps;
        let v = 0; for (let t = 0; t < this.nSteps; t++) { const d = this.adv[t] - m; v += d * d; }
        const sd = Math.sqrt(v / this.nSteps) + 1e-8;
        for (let t = 0; t < this.nSteps; t++) this.adv[t] = (this.adv[t] - m) / sd;
        this.pol.zeroGrad(); this.val.zeroGrad();
        this.bwEp = 0; this.phase = 'BACKWARD';
      }
      if (this.phase === 'BACKWARD') {
        const beta = this.beta, sbuf = new Float64Array(NIN);
        while (this.bwEp < this.eps.length) {
          const e = this.eps[this.bwEp];
          for (let t = e.start; t < e.end; t++) {
            const i = t * NIN;
            for (let k = 0; k < NIN; k++) sbuf[k] = this.obs[i + k];
            this.pol.forward(sbuf); this.pol.backward(sbuf, this.act[t], this.adv[t], beta);
            this.val.forward(sbuf); this.val.backward(sbuf, this.ret[t]);
          }
          this.bwEp++;
          if (now() - t0 > budgetMs) return did;
        }
        this.phase = 'ADAM';
      }
      if (this.phase === 'ADAM') {
        const sc = 1 / Math.max(1, this.nSteps);
        this.polOpt.lr = this.cfg.lrPolicy; this.valOpt.lr = this.cfg.lrValue;
        this.gradNorm = this.polOpt.step(sc);
        this.valOpt.step(sc);
        this.updates++;
        this.eps.length = 0; this.nSteps = 0;
        this.phase = 'COLLECT';
        this.startEpisode();
        if (now() - t0 > budgetMs || did >= stepLimit) return did;
      }
    }
  }
  snapshot() {
    return this.pol.tensors().map(t => Float64Array.from(t));
  }
  loadSnapshot(sn) {
    const T = this.pol.tensors();
    for (let i = 0; i < T.length; i++) T[i].set(sn[i]);
  }
  meanRecent(n) {
    const a = this.ghost.ret, N = this.ghost.n;
    const k = Math.min(n, N); if (!k) return 0;
    let s = 0; for (let i = N - k; i < N; i++) s += a[i % CURVE_CAP];
    return s / k;
  }
}

if (typeof module !== 'undefined') module.exports = {
  RNG, mulberry32, LunarEnv, PolicyNet, ValueNet, Adam,
  SCALE, FPS, DT, W, H, HELIPAD_Y, OBS_Y_REF, LANDER_POLY, CHUNKS,
  MASS, INERTIA, COM_OFF, legFootLocal, PHI_MIN, PHI_MAX, NIN, NOUT,
  Trainer, evaluatePolicy, Evaluator, PRESETS, GhostBuffer, CurveHistory, HIST_BINS, DEFAULTS, maxStepsFor, GHOST_SLOTS, GHOST_PTS, CURVE_CAP,
};
