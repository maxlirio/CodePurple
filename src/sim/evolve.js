// Co-evolution: two separate populations (red evaders and blue pursuers)
// that improve against each other. To get a stable learning signal, every
// individual is scored against the *other side's current champion* (a fixed
// target for the whole generation) rather than a moving same-index partner.
// The same module powers the headless seed trainer and the live,
// watch-it-learn loop in the browser.

import { ARCH, randomGenome, mutate, crossover } from "./nn.js";
import { runEpisode } from "./world.js";

export const POP = 16;   // individuals per side
const ELITE = 2;         // best genomes copied unchanged into the next gen

function emptyPop(rng) {
  return Array.from({ length: POP }, () => ({
    genome: randomGenome(ARCH, rng),
    fit: -Infinity,
  }));
}

function buildQueue() {
  // All evaders this generation, then all pursuers.
  const q = [];
  for (let i = 0; i < POP; i++) q.push({ side: "e", i });
  for (let i = 0; i < POP; i++) q.push({ side: "p", i });
  return q;
}

export function createEvolution(rng = Math.random) {
  return {
    arch: ARCH,
    gen: 1,
    evaderPop: emptyPop(rng),
    pursuerPop: emptyPop(rng),
    // Stationary sparring partners used until real champions emerge.
    champEvader: randomGenome(ARCH, rng),
    champPursuer: randomGenome(ARCH, rng),
    bestEvaderFit: -Infinity,
    bestPursuerFit: -Infinity,
    genBestEvaderFit: -Infinity,
    genBestPursuerFit: -Infinity,
    queue: buildQueue(),
  };
}

// The two genomes shown duelling live in the browser: the reigning champions.
export function getShowcase(s) {
  return { evaderGenome: s.champEvader, pursuerGenome: s.champPursuer };
}

// Progress through the current generation, 0..1.
export function genProgress(s) {
  return 1 - s.queue.length / (POP * 2);
}

function tournament(pop, rng) {
  let best = pop[(rng() * pop.length) | 0];
  for (let i = 0; i < 2; i++) {
    const c = pop[(rng() * pop.length) | 0];
    if (c.fit > best.fit) best = c;
  }
  return best.genome;
}

function nextGeneration(pop, rng) {
  const sorted = [...pop].sort((a, b) => b.fit - a.fit);
  const next = [];
  for (let i = 0; i < ELITE; i++) {
    next.push({ genome: Float64Array.from(sorted[i].genome), fit: -Infinity });
  }
  while (next.length < POP) {
    const child = crossover(
      tournament(sorted, rng), tournament(sorted, rng), rng
    );
    next.push({ genome: mutate(child, 0.18, 0.4, rng), fit: -Infinity });
  }
  return { next, champion: Float64Array.from(sorted[0].genome), best: sorted[0].fit };
}

// Run exactly one evaluation episode and advance the evolution. Cheap and
// non-blocking, so the browser can call it many times per animation frame
// and the headless trainer can spin it in a tight loop.
export function trainStep(s, rng = Math.random) {
  const task = s.queue.shift();
  if (task.side === "e") {
    const r = runEpisode(s.evaderPop[task.i].genome, s.champPursuer, rng);
    s.evaderPop[task.i].fit = r.evaderFit;
  } else {
    const r = runEpisode(s.champEvader, s.pursuerPop[task.i].genome, rng);
    s.pursuerPop[task.i].fit = r.pursuerFit;
  }

  if (s.queue.length > 0) return { genCompleted: false, gen: s.gen };

  // Generation finished: breed both sides and crown new champions.
  const e = nextGeneration(s.evaderPop, rng);
  const p = nextGeneration(s.pursuerPop, rng);
  s.champEvader = e.champion;
  s.champPursuer = p.champion;
  s.genBestEvaderFit = e.best;
  s.genBestPursuerFit = p.best;
  if (e.best > s.bestEvaderFit) s.bestEvaderFit = e.best;
  if (p.best > s.bestPursuerFit) s.bestPursuerFit = p.best;
  s.evaderPop = e.next;
  s.pursuerPop = p.next;
  s.gen++;
  s.queue = buildQueue();
  return { genCompleted: true, gen: s.gen };
}

// ---- Persistence (localStorage + the committed seed.json) ----

const toArr = (g) => (g ? Array.from(g) : null);
const toF64 = (a) => (a ? Float64Array.from(a) : null);

export function serialize(s) {
  return {
    v: 2,
    arch: s.arch,
    gen: s.gen,
    bestEvaderFit: s.bestEvaderFit,
    bestPursuerFit: s.bestPursuerFit,
    champEvader: toArr(s.champEvader),
    champPursuer: toArr(s.champPursuer),
    evaderPop: s.evaderPop.map((x) => toArr(x.genome)),
    pursuerPop: s.pursuerPop.map((x) => toArr(x.genome)),
  };
}

export function deserialize(d) {
  if (!d || d.v !== 2) return null;
  const pop = (arr) => arr.map((g) => ({ genome: toF64(g), fit: -Infinity }));
  return {
    arch: d.arch || ARCH,
    gen: d.gen || 1,
    evaderPop: pop(d.evaderPop),
    pursuerPop: pop(d.pursuerPop),
    champEvader: toF64(d.champEvader),
    champPursuer: toF64(d.champPursuer),
    bestEvaderFit: d.bestEvaderFit ?? -Infinity,
    bestPursuerFit: d.bestPursuerFit ?? -Infinity,
    genBestEvaderFit: -Infinity,
    genBestPursuerFit: -Infinity,
    queue: buildQueue(),
  };
}
