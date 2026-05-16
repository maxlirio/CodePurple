// Headless pre-trainer: B + A.
//  Phase 1 (B, imitation): a scripted builder/chaser generate demos;
//    red's net is behavior-cloned onto the builder so it is BORN knowing
//    the grab->carry->place chain it could never explore into.
//  Phase 2 (A, RL fine-tune): red is now permanently slower than blue
//    (set in world CFG) so running loses and building is necessary; both
//    agents fine-tune with RL from there.
// Usage: node src/rl/train-dqn.mjs [rlEpisodes] [maxSeconds]

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createAgent, bcLearn, serializeAgent } from "./dqn.js";
import { trainEpisode, actionVel, INTERACT } from "./env.js";
import { createWorld, reset, sense, applyStep, CFG } from "../sim/world.js";
import { scriptedEvader, scriptedPursuer } from "./scripted.js";

const RL_EPISODES = parseInt(process.argv[2] || "3000", 10);
const MAX_SECONDS = parseInt(process.argv[3] || "480", 10);
const EP_T = 25;
const DEMO_EPISODES = 220;
const BC_UPDATES = 6000;
const BC_BATCH = 64;
const out = join(
  dirname(fileURLToPath(import.meta.url)), "..", "rl-seed.json"
);

const agE = createAgent();
const agP = createAgent();
const start = Date.now();

// ---- Phase 1: collect demonstrations from the scripted teacher ----
const demos = [];
for (let ep = 0; ep < DEMO_EPISODES; ep++) {
  const w = createWorld();
  reset(w);
  const maxSteps = Math.ceil(EP_T / CFG.DT);
  for (let s = 0; s < maxSteps && !w.over; s++) {
    const aE = scriptedEvader(w);
    const aP = scriptedPursuer(w);
    demos.push({ s: Float64Array.from(sense(w, "evader")), a: aE });
    const [ex, ez] = actionVel(aE, CFG.SPEED_E);
    const [px, pz] = actionVel(aP, CFG.SPEED_P);
    applyStep(w, ex, ez, px, pz, aE === INTERACT, aP === INTERACT);
  }
}
console.log(`demos: ${demos.length} transitions from ${DEMO_EPISODES} eps`);

// ---- Phase 1: behavior-clone red onto the builder ----
for (let u = 1; u <= BC_UPDATES; u++) {
  const batch = [];
  for (let i = 0; i < BC_BATCH; i++)
    batch.push(demos[(Math.random() * demos.length) | 0]);
  bcLearn(agE, batch);
  if (u % 1500 === 0)
    console.log(`bc update ${u}/${BC_UPDATES} | ${
      ((Date.now() - start) / 1000).toFixed(0)}s`);
}

// ---- Phase 2: RL fine-tune (red slower; building is now necessary) ----
let caught = 0, sumT = 0, win = 0;
for (let ep = 1; ep <= RL_EPISODES; ep++) {
  // Low, decaying exploration so the cloned skill is refined, not erased.
  const eps = Math.max(0.05, 0.25 - 0.20 * (ep / RL_EPISODES));
  const r = trainEpisode(agE, agP, eps, eps, EP_T);
  if (r.outcome === "caught") caught++;
  sumT += r.time; win++;
  if (ep % 100 === 0) {
    const secs = (Date.now() - start) / 1000;
    process.stdout.write(
      `rl ${String(ep).padStart(4)} | eps ${eps.toFixed(2)} | ` +
      `catch ${(100 * caught / win).toFixed(0).padStart(3)}% | ` +
      `avg ${(sumT / win).toFixed(1).padStart(5)}s | ${secs.toFixed(0)}s\n`
    );
    caught = sumT = win = 0;
    if (secs > MAX_SECONDS) { console.log("time budget reached"); break; }
  }
}

writeFileSync(out, JSON.stringify({
  v: 1,
  evader: serializeAgent(agE),
  pursuer: serializeAgent(agP),
}));
console.log(`wrote ${out}`);
