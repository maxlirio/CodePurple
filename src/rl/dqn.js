// A small but complete Deep Q-Network, from scratch in pure JS (no deps, so
// it still runs on the static GitHub Pages site). MLP with ReLU hidden
// layers and a linear Q head, Adam optimiser, experience replay, a target
// network and Double-DQN targets. Used by both cubes as independent agents.

// 9 discrete actions: 8 compass directions + stay put. Velocity easing in
// world.js keeps the resulting motion from looking grid-locked.
export const ACTIONS = (() => {
  const a = [[0, 0]];
  for (let i = 0; i < 8; i++) {
    const ang = (i / 8) * Math.PI * 2;
    a.push([Math.cos(ang), Math.sin(ang)]);
  }
  return a; // length 9 (index 0 = stay, 1..8 = compass directions)
})();

// One extra discrete action beyond movement: index 9 = "interact" (grab a
// nearby block, or drop the carried one). 18 sensor inputs.
export const INTERACT = ACTIONS.length; // = 9
export const N_ACT = ACTIONS.length + 1; // = 10
const SIZES = [18, 48, 48, N_ACT];
const GAMMA = 0.99;
const LR = 5e-4;
const B1 = 0.9, B2 = 0.999, EPS = 1e-8;

function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeNet(sizes = SIZES) {
  const L = sizes.length - 1;
  const W = [], b = [], mW = [], vW = [], mb = [], vb = [];
  for (let l = 0; l < L; l++) {
    const ni = sizes[l], no = sizes[l + 1];
    const w = new Float64Array(ni * no);
    const s = Math.sqrt(2 / ni); // He init
    for (let i = 0; i < w.length; i++) w[i] = gauss() * s;
    W.push(w); b.push(new Float64Array(no));
    mW.push(new Float64Array(ni * no)); vW.push(new Float64Array(ni * no));
    mb.push(new Float64Array(no)); vb.push(new Float64Array(no));
  }
  return { sizes, L, W, b, mW, vW, mb, vb, t: 0 };
}

// Forward pass; keeps activations & pre-activations for backprop.
function forward(net, x) {
  const acts = [x], zs = [];
  let a = x;
  for (let l = 0; l < net.L; l++) {
    const ni = net.sizes[l], no = net.sizes[l + 1];
    const w = net.W[l], bb = net.b[l];
    const z = new Float64Array(no);
    for (let o = 0; o < no; o++) {
      let s = bb[o];
      const base = o * ni;
      for (let i = 0; i < ni; i++) s += w[base + i] * a[i];
      z[o] = s;
    }
    zs.push(z);
    if (l < net.L - 1) {
      const r = new Float64Array(no);
      for (let o = 0; o < no; o++) r[o] = z[o] > 0 ? z[o] : 0; // ReLU
      a = r;
    } else {
      a = z; // linear Q head
    }
    acts.push(a);
  }
  return { acts, zs, out: a };
}

export function predict(net, x) {
  return forward(net, x).out;
}

export function argmax(v) {
  let bi = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i;
  return bi;
}

function copyInto(dst, src) {
  for (let l = 0; l < src.L; l++) {
    dst.W[l].set(src.W[l]);
    dst.b[l].set(src.b[l]);
  }
}

// One Adam step from accumulated gradients (already mean over the batch).
function adam(net, gW, gb) {
  net.t++;
  const lrt = LR * Math.sqrt(1 - Math.pow(B2, net.t)) /
    (1 - Math.pow(B1, net.t));
  for (let l = 0; l < net.L; l++) {
    const W = net.W[l], mW = net.mW[l], vW = net.vW[l], gw = gW[l];
    for (let i = 0; i < W.length; i++) {
      mW[i] = B1 * mW[i] + (1 - B1) * gw[i];
      vW[i] = B2 * vW[i] + (1 - B2) * gw[i] * gw[i];
      W[i] -= lrt * mW[i] / (Math.sqrt(vW[i]) + EPS);
    }
    const b = net.b[l], mb = net.mb[l], vb = net.vb[l], gbb = gb[l];
    for (let o = 0; o < b.length; o++) {
      mb[o] = B1 * mb[o] + (1 - B1) * gbb[o];
      vb[o] = B2 * vb[o] + (1 - B2) * gbb[o] * gbb[o];
      b[o] -= lrt * mb[o] / (Math.sqrt(vb[o]) + EPS);
    }
  }
}

export function createAgent() {
  const net = makeNet();
  const target = makeNet();
  copyInto(target, net);
  return {
    net, target,
    buf: [], cap: 60000, head: 0,
    steps: 0, sync: 1000, batch: 32, minStart: 1000,
  };
}

export function remember(ag, s, a, r, s2, done) {
  const e = { s, a, r, s2, done };
  if (ag.buf.length < ag.cap) ag.buf.push(e);
  else { ag.buf[ag.head] = e; ag.head = (ag.head + 1) % ag.cap; }
}

export function act(ag, state, eps) {
  if (Math.random() < eps) return (Math.random() * ACTIONS.length) | 0;
  return argmax(predict(ag.net, state));
}

// Sample a minibatch and take one gradient step (Double-DQN, Huber loss).
export function learn(ag) {
  if (ag.buf.length < ag.minStart) return;
  const net = ag.net, tgt = ag.target;
  const gW = net.W.map((w) => new Float64Array(w.length));
  const gb = net.b.map((b) => new Float64Array(b.length));

  for (let n = 0; n < ag.batch; n++) {
    const e = ag.buf[(Math.random() * ag.buf.length) | 0];
    let y = e.r;
    if (!e.done) {
      const aStar = argmax(predict(net, e.s2)); // online picks action
      y += GAMMA * predict(tgt, e.s2)[aStar];   // target values it
    }
    const fwd = forward(net, e.s);
    let err = fwd.out[e.a] - y;
    if (err > 1) err = 1; else if (err < -1) err = -1; // Huber grad clip

    // Backprop: only the taken action has non-zero output gradient.
    let dz = new Float64Array(net.sizes[net.L]);
    dz[e.a] = err;
    for (let l = net.L - 1; l >= 0; l--) {
      const ni = net.sizes[l], no = net.sizes[l + 1];
      const aPrev = fwd.acts[l], w = net.W[l];
      const gw = gW[l], gbb = gb[l];
      for (let o = 0; o < no; o++) {
        const d = dz[o];
        if (d === 0) continue;
        gbb[o] += d;
        const base = o * ni;
        for (let i = 0; i < ni; i++) gw[base + i] += d * aPrev[i];
      }
      if (l > 0) {
        const dPrev = new Float64Array(ni);
        const zPrev = fwd.zs[l - 1];
        for (let i = 0; i < ni; i++) {
          if (zPrev[i] <= 0) continue; // ReLU derivative
          let s = 0;
          for (let o = 0; o < no; o++) s += w[o * ni + i] * dz[o];
          dPrev[i] = s;
        }
        dz = dPrev;
      }
    }
  }
  const inv = 1 / ag.batch;
  for (let l = 0; l < net.L; l++) {
    for (let i = 0; i < gW[l].length; i++) gW[l][i] *= inv;
    for (let o = 0; o < gb[l].length; o++) gb[l][o] *= inv;
  }
  adam(net, gW, gb);

  if (++ag.steps % ag.sync === 0) copyInto(tgt, net);
}

// Behavior cloning: supervised step that pushes the net to pick the
// demonstrated action (softmax cross-entropy over the Q outputs treated as
// logits). `batch` is an array of { s, a } from a scripted teacher.
export function bcLearn(ag, batch) {
  const net = ag.net;
  const gW = net.W.map((w) => new Float64Array(w.length));
  const gb = net.b.map((b) => new Float64Array(b.length));

  for (const e of batch) {
    const fwd = forward(net, e.s);
    const out = fwd.out;
    let mx = out[0];
    for (let i = 1; i < out.length; i++) if (out[i] > mx) mx = out[i];
    let sum = 0;
    const p = new Float64Array(out.length);
    for (let i = 0; i < out.length; i++) { p[i] = Math.exp(out[i] - mx); sum += p[i]; }
    let dz = new Float64Array(out.length);
    for (let i = 0; i < out.length; i++) dz[i] = p[i] / sum;
    dz[e.a] -= 1; // softmax-CE gradient

    for (let l = net.L - 1; l >= 0; l--) {
      const ni = net.sizes[l], no = net.sizes[l + 1];
      const aPrev = fwd.acts[l], w = net.W[l];
      const gw = gW[l], gbb = gb[l];
      for (let o = 0; o < no; o++) {
        const dd = dz[o];
        if (dd === 0) continue;
        gbb[o] += dd;
        const base = o * ni;
        for (let i = 0; i < ni; i++) gw[base + i] += dd * aPrev[i];
      }
      if (l > 0) {
        const dPrev = new Float64Array(ni);
        const zPrev = fwd.zs[l - 1];
        for (let i = 0; i < ni; i++) {
          if (zPrev[i] <= 0) continue;
          let s = 0;
          for (let o = 0; o < no; o++) s += w[o * ni + i] * dz[o];
          dPrev[i] = s;
        }
        dz = dPrev;
      }
    }
  }
  const inv = 1 / batch.length;
  for (let l = 0; l < net.L; l++) {
    for (let i = 0; i < gW[l].length; i++) gW[l][i] *= inv;
    for (let o = 0; o < gb[l].length; o++) gb[l][o] *= inv;
  }
  adam(net, gW, gb);
  copyInto(ag.target, ag.net); // keep target aligned with the cloned policy
}

// ---- Persistence (rl-seed.json + localStorage) ----
export function serializeAgent(ag) {
  return {
    sizes: ag.net.sizes,
    W: ag.net.W.map((w) => Array.from(w)),
    b: ag.net.b.map((b) => Array.from(b)),
    steps: ag.steps,
  };
}

export function deserializeAgent(d) {
  if (!d || !d.W) return null;
  // Reject brains from an older action/observation shape.
  if (d.sizes && (d.sizes[0] !== SIZES[0] ||
      d.sizes[d.sizes.length - 1] !== N_ACT)) return null;
  const ag = createAgent();
  for (let l = 0; l < ag.net.L; l++) {
    ag.net.W[l] = Float64Array.from(d.W[l]);
    ag.net.b[l] = Float64Array.from(d.b[l]);
  }
  copyInto(ag.target, ag.net);
  ag.steps = d.steps || 0;
  return ag;
}
