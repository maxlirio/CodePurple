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

// Reward red for using terrain as cover: a one-time payout when a block it
// just placed breaks blue's line of sight, plus a small ongoing bonus for
// every step it stays hidden behind a wall. Escaping (+1 terminal) still
// dominates, so this guides discovery without distorting the goal.
function buildReward(w, placedByRed) {
  const hidden = lineOccluded(
    w, w.pursuer.x, w.pursuer.z, w.evader.x, w.evader.z
  );
  let r = hidden ? 0.006 : 0;
  if (placedByRed && hidden) r += 0.15; // built itself cover this step
  return r;
}

// Run one training episode: both agents explore (epsilon), collect
// transitions, and learn every few steps. Returns episode stats.
export function trainEpisode(agE, agP, epsE, epsP, maxT, learnEvery = 6) {
  const w = createWorld();
  reset(w);
  let sE = Float64Array.from(sense(w, "evader"));
  let sP = Float64Array.from(sense(w, "pursuer"));
  let steps = 0;
  const maxSteps = Math.ceil(maxT / CFG.DT);

  while (!w.over && steps < maxSteps) {
    const aE = act(agE, sE, epsE);
    const aP = act(agP, sP, epsP);
    const d0 = gap(w);
    const [evx, evz] = actionVel(aE, CFG.SPEED_E);
    const [pvx, pvz] = actionVel(aP, CFG.SPEED_P);
    const placed = applyStep(
      w, evx, evz, pvx, pvz, aE === INTERACT, aP === INTERACT
    );
    let [rE, rP] = rewards(d0, gap(w), w.outcome);
    rE += buildReward(w, !!placed.evPlaced);
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
