// Headless DQN pre-trainer. Produces src/rl-seed.json so the cubes already
// behave sensibly on first load of the (branch-only) DQN build.
// Usage: node src/rl/train-dqn.mjs [episodes] [maxSeconds]

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createAgent, serializeAgent } from "./dqn.js";
import { trainEpisode } from "./env.js";

const EPISODES = parseInt(process.argv[2] || "600", 10);
const MAX_SECONDS = parseInt(process.argv[3] || "200", 10);
const EP_T = 25;          // seconds per training episode
const DECAY = 400;        // episodes to anneal epsilon 1.0 -> 0.05
const out = join(
  dirname(fileURLToPath(import.meta.url)), "..", "rl-seed.json"
);

const agE = createAgent();
const agP = createAgent();
const start = Date.now();
let caught = 0, escaped = 0, sumT = 0, win = 0;

for (let ep = 1; ep <= EPISODES; ep++) {
  const eps = Math.max(0.05, 1 - ep / DECAY);
  const r = trainEpisode(agE, agP, eps, eps, EP_T);
  if (r.outcome === "caught") caught++;
  else escaped++;
  sumT += r.time;
  win++;

  if (ep % 25 === 0) {
    const secs = (Date.now() - start) / 1000;
    process.stdout.write(
      `ep ${String(ep).padStart(4)} | eps ${eps.toFixed(2)} | ` +
      `blue-catch ${(100 * caught / win).toFixed(0).padStart(3)}% | ` +
      `avg ${ (sumT / win).toFixed(1).padStart(5) }s | ${secs.toFixed(0)}s\n`
    );
    caught = escaped = sumT = win = 0;
    if (secs > MAX_SECONDS) { console.log("time budget reached"); break; }
  }
}

writeFileSync(out, JSON.stringify({
  v: 1,
  evader: serializeAgent(agE),
  pursuer: serializeAgent(agP),
}));
console.log(`wrote ${out}`);
