# CodePurple

A 3D arena where two **co-evolving neural-network cubes** learn forever:

- 🟥 **Red** is an AI that learns to *run away and escape*.
- 🟦 **Blue** is an AI that learns to *hunt red down and catch it*.

They shove the grey blocks around as they move, carving passageways and
walls into the arena. When blue touches red, the round restarts. If red
survives **2 minutes**, red wins and the round restarts. The learning
**never stops** — you just open the page and watch them get better.

### Play with them (sandbox)

It's not just a spectator view. Use the **sandbox** toolbar (top-right):

- 🧱 **wall** — drop immovable walls to build mazes and dead-ends.
- 📦 **block** — drop extra pushable blocks.
- ✋ **drag** — grab and drag *anything*, including the red and blue cubes
  themselves, to mess with the chase in real time.
- ✕ **erase** / 🗑 **clear** — remove objects you've placed.

The AIs *sense and adapt to* whatever you build — their rangefinder
sensors see your walls and blocks. Your layout is saved in the browser
and persists across visits.

▶ **Live:** https://maxlirio.github.io/CodePurple/

## How it works

- **Rendering:** [Three.js](https://threejs.org) (loaded from a CDN). No build step.
- **Brains:** a tiny from-scratch neural net (`15 → 20 → 16 → 2`, tanh),
  one per cube, encoded as a flat weight vector — see `src/sim/nn.js`.
- **Learning:** neuroevolution. Two separate populations (evaders and
  pursuers) compete; the fittest are selected, crossed over and mutated
  into the next generation — see `src/sim/evolve.js`.
- **World:** continuous physics on a plane with pushable blocks, wall/block
  rangefinder sensors, catch detection and the 2-minute escape rule —
  see `src/sim/world.js`.
- **Persistence:** progress is saved to your browser's `localStorage`, so
  it resumes and keeps improving every visit. A pre-trained **seed brain**
  (`src/seed.json`) ships in the repo so the cubes already look competent
  on the very first load.

Because GitHub Pages is a static host, training runs in *your* browser
while the page is open — that is the "watch them learn" part. Use the
speed buttons (1×–16×) to fast-forward many generations of learning.

## Run locally

```bash
cd CodePurple
npm run serve        # then open http://localhost:5173
```

## Re-train the seed brain

```bash
npm run seed                 # ~80 generations (default)
node scripts/train-seed.mjs 150 300   # 150 gens, max 300s
```

This regenerates `src/seed.json`. Commit it to ship a smarter starting point.

## License

MIT
