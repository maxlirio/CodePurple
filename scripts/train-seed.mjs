// Headless seed trainer. Runs co-evolution for a while so the cubes already
// look competent the first time the page loads, then writes src/seed.json.
// Usage: node scripts/train-seed.mjs [generations] [maxSeconds]

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createEvolution, trainStep, serialize } from "../src/sim/evolve.js";

const GENERATIONS = parseInt(process.argv[2] || "120", 10);
const MAX_SECONDS = parseInt(process.argv[3] || "180", 10);
const out = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "seed.json");

const s = createEvolution();
const start = Date.now();

while (s.gen <= GENERATIONS) {
  const { genCompleted } = trainStep(s);
  if (!genCompleted) continue;
  const secs = (Date.now() - start) / 1000;
  process.stdout.write(
    `gen ${String(s.gen - 1).padStart(3)} | ` +
    `red(gen/best) ${s.genBestEvaderFit.toFixed(1).padStart(7)} / ` +
    `${s.bestEvaderFit.toFixed(1).padStart(7)} | ` +
    `blue(gen/best) ${s.genBestPursuerFit.toFixed(1).padStart(7)} / ` +
    `${s.bestPursuerFit.toFixed(1).padStart(7)} | ${secs.toFixed(0)}s\n`
  );
  if (secs > MAX_SECONDS) { console.log("time budget reached"); break; }
}

writeFileSync(out, JSON.stringify(serialize(s)));
console.log(`wrote ${out} (gen ${s.gen})`);
