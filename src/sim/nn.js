// Minimal feed-forward neural network encoded as a flat Float64 "genome".
// Pure ESM, no dependencies, runs identically in the browser and in Node.

// Network shape: 15 inputs -> 20 -> 16 -> 2 outputs (tanh everywhere).
export const ARCH = [15, 20, 16, 2];

// Number of weights+biases required for a given architecture.
export function genomeLength(arch = ARCH) {
  let n = 0;
  for (let l = 1; l < arch.length; l++) n += arch[l - 1] * arch[l] + arch[l];
  return n;
}

// Box-Muller gaussian.
export function gaussian(rng = Math.random) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function randomGenome(arch = ARCH, rng = Math.random) {
  const len = genomeLength(arch);
  const g = new Float64Array(len);
  for (let i = 0; i < len; i++) g[i] = gaussian(rng) * 0.6;
  return g;
}

// Forward pass. `genome` is a flat Float64Array; `arch` describes the layers.
export function forward(genome, inputs, arch = ARCH) {
  let act = inputs;
  let p = 0;
  for (let l = 1; l < arch.length; l++) {
    const nIn = arch[l - 1];
    const nOut = arch[l];
    const out = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let sum = genome[p + nIn * nOut + o]; // bias
      const base = o * nIn;
      for (let i = 0; i < nIn; i++) sum += act[i] * genome[p + base + i];
      out[o] = Math.tanh(sum);
    }
    p += nIn * nOut + nOut;
    act = out;
  }
  return act;
}

export function mutate(genome, rate = 0.15, std = 0.35, rng = Math.random) {
  const g = Float64Array.from(genome);
  for (let i = 0; i < g.length; i++) {
    if (rng() < rate) g[i] += gaussian(rng) * std;
  }
  return g;
}

export function crossover(a, b, rng = Math.random) {
  const g = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) g[i] = rng() < 0.5 ? a[i] : b[i];
  return g;
}
