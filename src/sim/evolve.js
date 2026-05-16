// Co-evolution: two separate populations (red evaders and blue pursuers)
// that improve against each other.
//
// To stop the classic co-evolutionary failure (each side overfits the
// other's *latest* champion, "forgets" general skill, and the two lineages
// cycle forever — which on screen looks like mutual avoidance), every
// individual is scored against a HALL OF FAME: the current champion plus a
// sample of past champions. Fitness is the mean across those opponents, so
// only brains that are robustly good survive.
//
// The same module powers the headless seed trainer and the live,
// watch-it-learn loop in the browser.

import { ARCH, randomGenome, mutate, crossover } from "./nn.js";
import { runEpisode } from "./world.js";

export const POP = 16;     // individuals per side
const ELITE = 2;           // best genomes copied unchanged into the next gen
const HOF_CAP = 8;         // how many past champions to remember per side
const HOF_SAMPLE = 2;      // how many of them to test against (plus champion)
const HOF_EVERY = 6;       // archive a champion into the HoF every N gens

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
    hofEvader: [],   // archived past evader champions
    hofPursuer: [],  // archived past pursuer champions
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

function sampleN(arr, n, rng) {
  if (arr.length <= n) return arr.slice();
  const pool = arr.slice(), out = [];
  for (let i = 0; i < n; i++) {
    out.push(pool.splice((rng() * pool.length) | 0, 1)[0]);
  }
  return out;
}

// Mean fitness of one genome against the champion + a Hall-of-Fame sample.
function evalEvader(s, genome, rng) {
  const foes = [s.champPursuer, ...sampleN(s.hofPursuer, HOF_SAMPLE, rng)];
  let sum = 0;
  for (const f of foes) sum += runEpisode(genome, f, rng).evaderFit;
  return sum / foes.length;
}
function evalPursuer(s, genome, rng) {
  const foes = [s.champEvader, ...sampleN(s.hofEvader, HOF_SAMPLE, rng)];
  let sum = 0;
  for (const f of foes) sum += runEpisode(f, genome, rng).pursuerFit;
  return sum / foes.length;
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
  return {
    next, champion: Float64Array.from(sorted[0].genome), best: sorted[0].fit,
  };
}

function archive(hof, genome) {
  hof.push(Float64Array.from(genome));
  if (hof.length > HOF_CAP) hof.shift();
}

// Run one individual's full evaluation (champion + HoF) and advance the
// evolution. Cheap and non-blocking, so the browser can call it many times
// per animation frame and the headless trainer can spin it in a tight loop.
export function trainStep(s, rng = Math.random) {
  // Backward-compat for brains saved before the Hall of Fame existed.
  if (!s.hofEvader) s.hofEvader = [];
  if (!s.hofPursuer) s.hofPursuer = [];

  const task = s.queue.shift();
  if (task.side === "e") {
    s.evaderPop[task.i].fit = evalEvader(s, s.evaderPop[task.i].genome, rng);
  } else {
    s.pursuerPop[task.i].fit =
      evalPursuer(s, s.pursuerPop[task.i].genome, rng);
  }

  if (s.queue.length > 0) return { genCompleted: false, gen: s.gen };

  // Generation finished: periodically archive the outgoing champions so the
  // HoF spans history, then breed and crown the new champions.
  if (s.gen % HOF_EVERY === 0) {
    archive(s.hofEvader, s.champEvader);
    archive(s.hofPursuer, s.champPursuer);
  }
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
    v: 3,
    arch: s.arch,
    gen: s.gen,
    bestEvaderFit: s.bestEvaderFit,
    bestPursuerFit: s.bestPursuerFit,
    champEvader: toArr(s.champEvader),
    champPursuer: toArr(s.champPursuer),
    hofEvader: s.hofEvader.map(toArr),
    hofPursuer: s.hofPursuer.map(toArr),
    evaderPop: s.evaderPop.map((x) => toArr(x.genome)),
    pursuerPop: s.pursuerPop.map((x) => toArr(x.genome)),
  };
}

export function deserialize(d) {
  if (!d || d.v !== 3) return null;
  const pop = (arr) => arr.map((g) => ({ genome: toF64(g), fit: -Infinity }));
  return {
    arch: d.arch || ARCH,
    gen: d.gen || 1,
    evaderPop: pop(d.evaderPop),
    pursuerPop: pop(d.pursuerPop),
    champEvader: toF64(d.champEvader),
    champPursuer: toF64(d.champPursuer),
    hofEvader: (d.hofEvader || []).map(toF64),
    hofPursuer: (d.hofPursuer || []).map(toF64),
    bestEvaderFit: d.bestEvaderFit ?? -Infinity,
    bestPursuerFit: d.bestPursuerFit ?? -Infinity,
    genBestEvaderFit: -Infinity,
    genBestPursuerFit: -Infinity,
    queue: buildQueue(),
  };
}
