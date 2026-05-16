// The shared game/physics simulation. No rendering here — the browser draws
// the state this module produces, and the headless seed trainer runs it
// thousands of times. Coordinates are on the X/Z ground plane.

import { forward, ARCH } from "./nn.js";

export const CFG = {
  A: 16,            // arena half-extent (32 x 32 floor)
  AGENT_R: 0.7,     // agent radius
  CATCH_R: 1.4,     // distance at which blue "catches" red
  SPEED_E: 6.4,     // red (evader) top speed, units / second
  SPEED_P: 7.2,     // blue (pursuer) is a touch faster, so skill matters
  MAX_SPEED: 7.2,   // used only to normalise sensor inputs
  ACCEL: 22.0,      // how fast an agent reaches its desired velocity
  BLOCK_H: 1.1,     // pushable block half-size
  PICK_R: 2.6,      // reach for picking up a block
  N_BLOCKS: 11,
  N_RAYS: 8,
  RAY_RANGE: 16,
  DT: 1 / 30,       // physics timestep
  ESCAPE_T: 120,    // red survives 2 minutes -> red wins, restart
};

function rand(rng, a, b) { return a + (b - a) * rng(); }

export function createWorld() {
  return {
    t: 0,
    // carry = the block object this cube is holding over its head, or null.
    evader: { x: 0, z: 0, vx: 0, vz: 0, carry: null }, // red
    pursuer: { x: 0, z: 0, vx: 0, vz: 0, carry: null }, // blue
    blocks: Array.from({ length: CFG.N_BLOCKS }, () => ({ x: 0, z: 0 })),
    // User sandbox objects. They persist across rounds (reset() leaves them
    // alone) and the AIs sense + collide with them just like procedural ones.
    userBlocks: [], // pushable squares the player drops in
    statics: [],    // immovable wall squares the player drops in
    over: false,
    outcome: null, // 'caught' | 'escaped'
  };
}

const clampInside = (p) => {
  const lim = CFG.A - CFG.BLOCK_H;
  p.x = Math.max(-lim, Math.min(lim, p.x));
  p.z = Math.max(-lim, Math.min(lim, p.z));
  return p;
};

// ---- Sandbox mutators (used by the browser, never by training) ----
export function addStatic(w, x, z) {
  const o = clampInside({ x, z });
  w.statics.push(o);
  return o;
}
export function addPusher(w, x, z) {
  const o = clampInside({ x, z });
  w.userBlocks.push(o);
  return o;
}
export function removeObject(w, ref) {
  for (const list of [w.statics, w.userBlocks]) {
    const i = list.indexOf(ref);
    if (i >= 0) { list.splice(i, 1); return true; }
  }
  return false;
}
export function clearUserObjects(w) {
  w.statics.length = 0;
  w.userBlocks.length = 0;
}

export function reset(w, rng = Math.random) {
  const A = CFG.A;
  w.t = 0;
  w.over = false;
  w.outcome = null;
  // Red and blue start on opposite sides.
  w.evader.x = rand(rng, -A * 0.7, -A * 0.3);
  w.evader.z = rand(rng, -A * 0.7, A * 0.7);
  w.evader.vx = w.evader.vz = 0;
  w.pursuer.x = rand(rng, A * 0.3, A * 0.7);
  w.pursuer.z = rand(rng, -A * 0.7, A * 0.7);
  w.pursuer.vx = w.pursuer.vz = 0;
  // Drop anything carried and un-hold every block for the new round.
  w.evader.carry = w.pursuer.carry = null;
  for (const b of w.blocks) b.held = false;
  for (const b of w.userBlocks) b.held = false;
  // Scatter blocks, clear of both spawns and not overlapping each other
  // (blocks are solid and never move on their own now).
  const lim = A - CFG.BLOCK_H - 0.5;
  const minGap = (2 * CFG.BLOCK_H + 0.5) ** 2;
  for (let i = 0; i < w.blocks.length; i++) {
    const b = w.blocks[i];
    for (let tries = 0; tries < 40; tries++) {
      b.x = rand(rng, -lim, lim);
      b.z = rand(rng, -lim, lim);
      if (dist2(b, w.evader) <= 9 || dist2(b, w.pursuer) <= 9) continue;
      let clash = false;
      for (let j = 0; j < i; j++) {
        if (dist2(b, w.blocks[j]) < minGap) { clash = true; break; }
      }
      if (!clash) break;
    }
  }
  return w;
}

function dist2(a, b) {
  const dx = a.x - b.x, dz = a.z - b.z;
  return dx * dx + dz * dz;
}

// Distance from a point along a unit ray to the nearest wall or square,
// clamped to RAY_RANGE and normalised to [0,1] (1 = nothing in range).
function rayClearance(ox, oz, dx, dz, w) {
  const A = CFG.A, R = CFG.RAY_RANGE, h = CFG.BLOCK_H;
  let best = R;
  // Walls (the four arena boundaries).
  if (dx > 1e-6) best = Math.min(best, (A - ox) / dx);
  else if (dx < -1e-6) best = Math.min(best, (-A - ox) / dx);
  if (dz > 1e-6) best = Math.min(best, (A - oz) / dz);
  else if (dz < -1e-6) best = Math.min(best, (-A - oz) / dz);
  // Every square (procedural, user pushable, user wall) via the slab method.
  for (const list of [w.blocks, w.userBlocks, w.statics]) {
    for (const b of list) {
      if (b.held) continue; // carried blocks are out of play
      const tx1 = (b.x - h - ox) / dx, tx2 = (b.x + h - ox) / dx;
      const tz1 = (b.z - h - oz) / dz, tz2 = (b.z + h - oz) / dz;
      const tmin = Math.max(Math.min(tx1, tx2), Math.min(tz1, tz2));
      const tmax = Math.min(Math.max(tx1, tx2), Math.max(tz1, tz2));
      if (tmax >= Math.max(tmin, 0) && tmin < best) best = Math.max(tmin, 0);
    }
  }
  return Math.max(0, Math.min(1, best / R));
}

// Is the straight line from (ax,az) to (bx,bz) blocked by any solid block
// or wall? Used to reward red for hiding behind cover it built.
export function lineOccluded(w, ax, az, bx, bz) {
  const h = CFG.BLOCK_H;
  let dx = bx - ax, dz = bz - az;
  if (Math.abs(dx) < 1e-6) dx = 1e-6;
  if (Math.abs(dz) < 1e-6) dz = 1e-6;
  for (const list of [w.blocks, w.userBlocks, w.statics]) {
    for (const o of list) {
      if (o.held) continue;
      const tx1 = (o.x - h - ax) / dx, tx2 = (o.x + h - ax) / dx;
      const tz1 = (o.z - h - az) / dz, tz2 = (o.z + h - az) / dz;
      const tmin = Math.max(Math.min(tx1, tx2), Math.min(tz1, tz2));
      const tmax = Math.min(Math.max(tx1, tx2), Math.max(tz1, tz2));
      if (tmax >= Math.max(tmin, 0) && tmin <= 1 && tmax >= 0) return true;
    }
  }
  return false;
}

// Nearest pickable (non-held) block to a point, or null. Statics and the
// agent's own carried block are not pickable.
export function nearestPickable(w, x, z) {
  let best = null, bd = Infinity;
  for (const list of [w.blocks, w.userBlocks]) {
    for (const b of list) {
      if (b.held) continue;
      const dd = (b.x - x) * (b.x - x) + (b.z - z) * (b.z - z);
      if (dd < bd) { bd = dd; best = b; }
    }
  }
  return best ? { ref: best, dist: Math.sqrt(bd) } : null;
}

// Build the 18-element sensor vector for one agent: opponent (3), own &
// foe velocity (4), 8 rangefinder rays, then 3 carry/terrain inputs:
// am-I-carrying, and the relative vector to the nearest pickable block.
export function sense(w, who) {
  const me = who === "evader" ? w.evader : w.pursuer;
  const foe = who === "evader" ? w.pursuer : w.evader;
  const A = CFG.A, S = CFG.MAX_SPEED;
  const dx = foe.x - me.x, dz = foe.z - me.z;
  const d = Math.hypot(dx, dz);
  const inp = [
    dx / (2 * A), dz / (2 * A), d / (2 * A * Math.SQRT2),
    me.vx / S, me.vz / S, foe.vx / S, foe.vz / S,
  ];
  for (let i = 0; i < CFG.N_RAYS; i++) {
    const a = (i / CFG.N_RAYS) * Math.PI * 2;
    inp.push(rayClearance(me.x, me.z, Math.cos(a), Math.sin(a), w));
  }
  const np = me.carry ? null : nearestPickable(w, me.x, me.z);
  inp.push(me.carry ? 1 : 0);
  inp.push(np ? (np.ref.x - me.x) / (2 * A) : 0);
  inp.push(np ? (np.ref.z - me.z) / (2 * A) : 0);
  return inp;
}

function moveAgent(ag, ox, oz, desX, desZ) {
  const dt = CFG.DT;
  // Ease velocity toward the network's desired velocity.
  ag.vx += (desX - ag.vx) * Math.min(1, CFG.ACCEL * dt);
  ag.vz += (desZ - ag.vz) * Math.min(1, CFG.ACCEL * dt);
  ag.x += ag.vx * dt;
  ag.z += ag.vz * dt;
  // Every block is a SOLID wall. Cubes cannot shove blocks — the only way
  // to move one is to pick it up and place it. So collision always pushes
  // the agent out; the block never moves.
  const ar = CFG.AGENT_R, h = CFG.BLOCK_H;
  const resolve = (b) => {
    if (b.held) return; // carried blocks don't collide
    const cx = Math.max(b.x - h, Math.min(ag.x, b.x + h));
    const cz = Math.max(b.z - h, Math.min(ag.z, b.z + h));
    let nx = ag.x - cx, nz = ag.z - cz;
    let dd = Math.hypot(nx, nz);
    if (dd < ar && dd > 1e-5) {
      const pen = ar - dd;
      nx /= dd; nz /= dd;
      ag.x += nx * pen;
      ag.z += nz * pen;
    }
  };
  for (const b of ox.blocks) resolve(b);
  for (const b of ox.userBlocks) resolve(b);
  for (const b of ox.statics) resolve(b);
  // Clamp agent inside the arena.
  const lim = CFG.A - ar;
  ag.x = Math.max(-lim, Math.min(lim, ag.x));
  ag.z = Math.max(-lim, Math.min(lim, ag.z));
  void oz; void desZ;
}

function settleBlocks(w) {
  const A = CFG.A, h = CFG.BLOCK_H, lim = A - h;
  // Pushables = procedural + user blocks, minus any being carried.
  const anyHeld = w.evader.carry || w.pursuer.carry;
  const bl = (!anyHeld && !w.userBlocks.length)
    ? w.blocks
    : w.blocks.concat(w.userBlocks).filter((b) => !b.held);
  const st = w.statics;
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < bl.length; i++) {
      for (let j = i + 1; j < bl.length; j++) {
        const a = bl[i], b = bl[j];
        const ox = 2 * h - Math.abs(a.x - b.x);
        const oz = 2 * h - Math.abs(a.z - b.z);
        if (ox > 0 && oz > 0) {
          if (ox < oz) {
            const s = (a.x < b.x ? -1 : 1) * ox * 0.5;
            a.x += s; b.x -= s;
          } else {
            const s = (a.z < b.z ? -1 : 1) * oz * 0.5;
            a.z += s; b.z -= s;
          }
        }
      }
      // Keep pushables out of immovable walls.
      for (let k = 0; k < st.length; k++) {
        const a = bl[i], s = st[k];
        const ox = 2 * h - Math.abs(a.x - s.x);
        const oz = 2 * h - Math.abs(a.z - s.z);
        if (ox > 0 && oz > 0) {
          if (ox < oz) a.x += (a.x < s.x ? -1 : 1) * ox;
          else a.z += (a.z < s.z ? -1 : 1) * oz;
        }
      }
      bl[i].x = Math.max(-lim, Math.min(lim, bl[i].x));
      bl[i].z = Math.max(-lim, Math.min(lim, bl[i].z));
    }
  }
}

// Advance the world one timestep using the two genomes (brains).
export function step(w, evaderGenome, pursuerGenome) {
  if (w.over) return;
  const oe = forward(evaderGenome, sense(w, "evader"), ARCH);
  const op = forward(pursuerGenome, sense(w, "pursuer"), ARCH);
  // Expose the raw control outputs so fitness can penalise jerky steering.
  w.evAct = [oe[0], oe[1]];
  w.puAct = [op[0], op[1]];
  moveAgent(w.evader, w, w, oe[0] * CFG.SPEED_E, oe[1] * CFG.SPEED_E);
  moveAgent(w.pursuer, w, w, op[0] * CFG.SPEED_P, op[1] * CFG.SPEED_P);
  settleBlocks(w);
  w.t += CFG.DT;
  const d = Math.hypot(w.evader.x - w.pursuer.x, w.evader.z - w.pursuer.z);
  if (d < CFG.CATCH_R) { w.over = true; w.outcome = "caught"; }
  else if (w.t >= CFG.ESCAPE_T) { w.over = true; w.outcome = "escaped"; }
}

// Grab the nearest block within reach (if hands are empty). Returns true
// if a block was picked up.
function doPickup(w, ag) {
  if (ag.carry) return false;
  const np = nearestPickable(w, ag.x, ag.z);
  if (np && np.dist <= CFG.PICK_R) {
    np.ref.held = true;
    ag.carry = np.ref;
    return true;
  }
  return false;
}

// Is (x,z) a clear spot to drop a block (no wall/other block/agent there)?
function spotClear(w, b, x, z) {
  const h = CFG.BLOCK_H, A = CFG.A;
  if (Math.abs(x) > A - h || Math.abs(z) > A - h) return false;
  const overlaps = (o) =>
    o !== b && !o.held &&
    Math.abs(o.x - x) < 2 * h && Math.abs(o.z - z) < 2 * h;
  for (const o of w.blocks) if (overlaps(o)) return false;
  for (const o of w.userBlocks) if (overlaps(o)) return false;
  for (const o of w.statics)
    if (Math.abs(o.x - x) < 2 * h && Math.abs(o.z - z) < 2 * h) return false;
  const onAgent = (ag) =>
    Math.abs(ag.x - x) < h + CFG.AGENT_R &&
    Math.abs(ag.z - z) < h + CFG.AGENT_R;
  return !onAgent(w.evader) && !onAgent(w.pursuer);
}

// Set the carried block down on a clear spot in front of the cube (it drops
// toward the opponent when the cube is still — i.e. between them). Returns
// the placed block, or null.
function doPlace(w, ag, foe) {
  const b = ag.carry;
  if (!b) return null;
  let dirx = ag.vx, dirz = ag.vz;
  let m = Math.hypot(dirx, dirz);
  if (m < 0.3) { // barely moving: drop on the side facing the opponent
    dirx = foe.x - ag.x; dirz = foe.z - ag.z;
    m = Math.hypot(dirx, dirz) || 1;
  }
  dirx /= m; dirz /= m;
  const lim = CFG.A - CFG.BLOCK_H;
  const base = CFG.AGENT_R + CFG.BLOCK_H + 0.15;
  let px = ag.x + dirx * base, pz = ag.z + dirz * base;
  // Try a few distances/angles to find a non-overlapping spot.
  outer: for (const extra of [0, 0.7, 1.4]) {
    for (const ang of [0, 0.6, -0.6, 1.2, -1.2]) {
      const c = Math.cos(ang), s = Math.sin(ang);
      const ux = dirx * c - dirz * s, uz = dirx * s + dirz * c;
      const x = ag.x + ux * (base + extra), z = ag.z + uz * (base + extra);
      if (spotClear(w, b, x, z)) { px = x; pz = z; break outer; }
    }
  }
  b.x = Math.max(-lim, Math.min(lim, px));
  b.z = Math.max(-lim, Math.min(lim, pz));
  b.held = false;
  ag.carry = null;
  return b;
}

// Advance one timestep from explicit desired velocities instead of genomes.
// Used by the DQN path. `evInteract`/`puInteract` mean that cube spent the
// turn grabbing or dropping a block instead of moving. Returns which blocks
// (if any) were just placed, for reward shaping.
export function applyStep(w, evVx, evVz, puVx, puVz, evInteract, puInteract) {
  if (w.over) return { evPlaced: null, puPlaced: null };
  // Grabbing/placing is instant: an interacting cube keeps coasting on its
  // current momentum instead of stopping dead, so building costs almost no
  // tempo against the pursuer.
  moveAgent(w.evader, w, w,
    evInteract ? w.evader.vx : evVx, evInteract ? w.evader.vz : evVz);
  moveAgent(w.pursuer, w, w,
    puInteract ? w.pursuer.vx : puVx, puInteract ? w.pursuer.vz : puVz);

  let evPlaced = null, puPlaced = null;
  if (evInteract) {
    evPlaced = w.evader.carry
      ? doPlace(w, w.evader, w.pursuer) : (doPickup(w, w.evader), null);
  }
  if (puInteract) {
    puPlaced = w.pursuer.carry
      ? doPlace(w, w.pursuer, w.evader) : (doPickup(w, w.pursuer), null);
  }
  // Held blocks ride above their carrier (kept sane for any readers).
  if (w.evader.carry) { w.evader.carry.x = w.evader.x; w.evader.carry.z = w.evader.z; }
  if (w.pursuer.carry) { w.pursuer.carry.x = w.pursuer.x; w.pursuer.carry.z = w.pursuer.z; }

  settleBlocks(w);
  w.t += CFG.DT;
  const d = Math.hypot(w.evader.x - w.pursuer.x, w.evader.z - w.pursuer.z);
  if (d < CFG.CATCH_R) { w.over = true; w.outcome = "caught"; }
  else if (w.t >= CFG.ESCAPE_T) { w.over = true; w.outcome = "escaped"; }
  return { evPlaced, puPlaced };
}

// Run a full headless episode and return fitness for both sides.
export function runEpisode(evaderGenome, pursuerGenome, rng = Math.random) {
  const w = createWorld();
  reset(w, rng);
  // Continuous proximity signal: average gap between the cubes over the
  // whole round. Lower mean distance => blue gets a smoothly higher score
  // for *pressing in*, with no penalty for a lunge that misses. The catch
  // bonus then dwarfs everything, so finishing always beats shadowing.
  let sumD = 0;
  let steps = 0;
  // Accumulate "jerk": how much each cube's intended steering changes from
  // one step to the next. A smooth policy keeps this tiny; a twitchy one
  // racks it up. Penalising it makes smoothness genuinely *learned*.
  let jerkE = 0, jerkP = 0;
  let prevE = null, prevP = null;
  const maxSteps = Math.ceil(CFG.ESCAPE_T / CFG.DT);
  for (let s = 0; s < maxSteps && !w.over; s++) {
    step(w, evaderGenome, pursuerGenome);
    sumD += Math.hypot(w.evader.x - w.pursuer.x, w.evader.z - w.pursuer.z);
    if (prevE) {
      jerkE += Math.hypot(w.evAct[0] - prevE[0], w.evAct[1] - prevE[1]);
      jerkP += Math.hypot(w.puAct[0] - prevP[0], w.puAct[1] - prevP[1]);
    }
    prevE = w.evAct; prevP = w.puAct;
    steps++;
  }
  const escaped = w.outcome === "escaped";
  const meanD = steps ? sumD / steps : CFG.A;
  // Mean per-step steering change, scaled into a modest score penalty.
  const SMOOTH = 35;
  const penE = steps > 1 ? (jerkE / (steps - 1)) * SMOOTH : 0;
  const penP = steps > 1 ? (jerkP / (steps - 1)) * SMOOTH : 0;

  // Blue: a catch is worth far more than any amount of shadowing, and an
  // earlier catch is better. No catch -> graded purely on how close it
  // managed to stay, so closing distance is *always* rewarded.
  const pursuerFit = (w.outcome === "caught"
    ? 250 + (CFG.ESCAPE_T - w.t) * 2
    : (escaped ? 40 : 80) - meanD * 4) - penP;

  // Red: escaping is the jackpot; otherwise reward lasting long and having
  // kept blue at arm's length.
  const evaderFit = (escaped
    ? 300 + meanD * 2
    : w.t * 2.5 + meanD * 2) - penE;

  return { evaderFit, pursuerFit, time: w.t, outcome: w.outcome };
}
