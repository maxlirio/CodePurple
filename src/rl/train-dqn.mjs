// Headless DQN pre-trainer with a construction-forcing CURRICULUM.
//
// The hard part isn't catching — it's getting red to discover the long
// grab->carry->place->survive chain. So during DISCOVERY blue runs at full
// speed (red cannot simply outrun it) but cannot catch yet, and taking
// cover is paid generously: the only way to relieve constant pressure is
// to wall blue off, which forces building. Then catching is switched on and
// the cover bonus decays, so the honest base reward (a solid wall really
// does delay a full-speed blue) keeps the behaviour alive.
//
// Usage: node src/rl/train-dqn.mjs [episodes] [maxSeconds]

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createAgent, serializeAgent } from "./dqn.js";
import { trainEpisode } from "./env.js";

const EPISODES = parseInt(process.argv[2] || "2600", 10);
const MAX_SECONDS = parseInt(process.argv[3] || "320", 10);
const EP_T = 25;
const out = join(
  dirname(fileURLToPath(import.meta.url)), "..", "rl-seed.json"
);

// p in [0,1]:
//  discovery p<0.40 : full-speed blue, NO catch, big build bonus
//  ramp 0.40..0.80  : catch ON, build bonus 2.0 -> 1.0
//  mastery p>0.80   : honest full-strength play, low exploration
function schedule(p) {
  if (p < 0.40) {
    return { blueScale: 1, noCatch: true, buildScale: 2.0,
             eps: Math.max(0.4, 1.0 - p * (0.6 / 0.40)) };
  }
  if (p < 0.80) {
    const q = (p - 0.40) / 0.40;
    return { blueScale: 1, noCatch: false, buildScale: 2.0 - q,
             eps: 0.4 - q * 0.3 };
  }
  return { blueScale: 1, noCatch: false, buildScale: 1,
           eps: Math.max(0.05, 0.1 - (p - 0.80) * 0.25) };
}

const agE = createAgent();
const agP = createAgent();
const start = Date.now();
let caught = 0, sumT = 0, win = 0;

for (let ep = 1; ep <= EPISODES; ep++) {
  const p = ep / EPISODES;
  const { blueScale, noCatch, buildScale, eps } = schedule(p);
  const r = trainEpisode(agE, agP, eps, eps, EP_T, 6,
    { blueScale, noCatch, buildScale });
  if (r.outcome === "caught") caught++;
  sumT += r.time;
  win++;

  if (ep % 50 === 0) {
    const secs = (Date.now() - start) / 1000;
    process.stdout.write(
      `ep ${String(ep).padStart(4)} | ${noCatch ? "DISC" : "live"} | ` +
      `bld ${buildScale.toFixed(1)} | eps ${eps.toFixed(2)} | ` +
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
