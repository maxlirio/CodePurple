// The reinforcement-learning environment: wraps the shared world simulation
// with per-step rewards and the experience-collection loop. Shared by the
// headless trainer and the in-browser learning loop.

import {
  createWorld, reset, sense, applyStep, lineOccluded, CFG,
} from "../sim/world.js";
import { ACTIONS, INTERACT, act, remember, learn } from "./dqn.js";

export { INTERACT } from "./dqn.js";

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// Movement actions map to a velocity; the interact action means "don't
// move, grab/drop instead" so its velocity is zero.
export function actionVel(idx, speed) {
  if (idx === INTERACT) return [0, 0];
  const [dx, dz] = ACTIONS[idx];
  return [dx * speed, dz * speed];
}

const gap = (w) =>
  Math.hypot(w.evader.x - w.pursuer.x, w.evader.z - w.pursuer.z);

// Per-step rewards. Dense distance shaping (clipped) gives a smooth signal;
// the terminal +/-1 makes catching / escaping decisive.
export function rewards(d0, d1, outcome) {
  const close = clamp((d0 - d1) * 0.15, -0.5, 0.5);
  let rP = close - 0.003;   // blue: closing good, small time pressure
  let rE = -close + 0.003;  // red: opening good, small survival bonus
  if (outcome === "caught") { rP += 1; rE -= 1; }
  else if (outcome === "escaped") { rP -= 1; rE += 1; }
  return [rE, rP];
}

// How many blocks red has placed that are clustered around it right now —
// a proxy for "structure size" (a wall/enclosure, not one lone block).
function redStructure(w) {
  const R2 = (3.2 * CFG.BLOCK_H) ** 2;
  let n = 0;
  for (const list of [w.blocks, w.userBlocks]) {
    for (const b of list) {
      if (b.held || !b.byRed) continue;
      const dx = b.x - w.evader.x, dz = b.z - w.evader.z;
      if (dx * dx + dz * dz < R2) n++;
    }
  }
  return n;
}

// Reward red for using terrain as cover, scaled by how COMPLEX the cover
// is: the rising-edge payout grows with the size of the structure red has
// built around itself (2 blocks > 1, capped), so a real wall/enclosure
// pays far more than one lone block. Rising-edge only => no farming.
function buildReward(w, placedByRed, prevHidden, scale = 1) {
  const hidden = lineOccluded(
    w, w.pursuer.x, w.pursuer.z, w.evader.x, w.evader.z
  );
  let r = hidden ? 0.008 : 0;
  if (placedByRed && hidden && !prevHidden) {
    const struct = Math.min(redStructure(w), 5); // 1..5 blocks
    r += 0.12 * struct * scale; // escalates with structure complexity
  }
  return { r, hidden };
}

// Run one training episode: both agents explore (epsilon), collect
// transitions, and learn every few steps. Returns episode stats.
// Curriculum knobs (opt): blueScale scales blue's speed; noCatch lets the
// episode continue even when blue touches red (so during discovery, running
// can't save red and can't end the episode — only breaking line of sight
// relieves the pressure, which forces building); buildScale boosts the
// take-cover reward early. All default to "normal full-strength play".
export function trainEpisode(
  agE, agP, epsE, epsP, maxT, learnEvery = 6, opt = {}
) {
  const blueScale = opt.blueScale ?? 1;
  const noCatch = opt.noCatch ?? false;
  const buildScale = opt.buildScale ?? 1;
  const w = createWorld();
  reset(w);
  let sE = Float64Array.from(sense(w, "evader"));
  let sP = Float64Array.from(sense(w, "pursuer"));
  let steps = 0;
  let prevHidden = false;
  const maxSteps = Math.ceil(maxT / CFG.DT);

  while (!w.over && steps < maxSteps) {
    const aE = act(agE, sE, epsE);
    const aP = act(agP, sP, epsP);
    const d0 = gap(w);
    const [evx, evz] = actionVel(aE, CFG.SPEED_E);
    const [pvx, pvz] = actionVel(aP, CFG.SPEED_P * blueScale);
    const placed = applyStep(
      w, evx, evz, pvx, pvz, aE === INTERACT, aP === INTERACT
    );
    // Discovery: blue can't actually catch yet — undo the terminal so the
    // episode keeps going and red must learn cover, not just survive a hit.
    if (noCatch && w.outcome === "caught") {
      w.over = false;
      w.outcome = null;
    }
    let [rE, rP] = rewards(d0, gap(w), w.outcome);
    const bw = buildReward(w, !!placed.evPlaced, prevHidden, buildScale);
    rE += bw.r;
    // Sharp ongoing penalty for being EXPOSED (blue has line of sight).
    // This is the "scream": continuous pressure to get behind cover it
    // builds — tied to exposure (a state red can fix), not to the
    // unavoidable act of not-building (which collapsed training before).
    if (!bw.hidden) rE -= 0.012;
    prevHidden = bw.hidden;
    const nE = Float64Array.from(sense(w, "evader"));
    const nP = Float64Array.from(sense(w, "pursuer"));
    const done = w.over ? 1 : 0;
    remember(agE, sE, aE, rE, nE, done);
    remember(agP, sP, aP, rP, nP, done);
    sE = nE; sP = nP;
    steps++;
    if (steps % learnEvery === 0) { learn(agE); learn(agP); }
  }
  return { time: w.t, outcome: w.outcome, steps };
}
