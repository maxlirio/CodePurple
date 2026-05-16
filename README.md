# CodePurple

A 3D arena where two **reinforcement-learning neural-network cubes** learn forever:

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
- **Brains:** a from-scratch **Deep Q-Network** — MLP with backprop + Adam,
  experience replay, a target network and Double-DQN targets — one agent
  per cube, see `src/rl/dqn.js`.
- **Learning:** reinforcement learning. Each cube chooses from 9 discrete
  moves; dense per-step distance rewards plus a decisive catch/escape
  terminal reward. You watch them play near-greedily in real time while
  epsilon-greedy training episodes run in the background — see `src/rl/env.js`.
- **World:** continuous physics on a plane with pushable blocks, wall/block
  rangefinder sensors, catch detection and the 2-minute escape rule —
  see `src/sim/world.js`.
- **Persistence:** progress is saved to your browser's `localStorage`, so
  it resumes and keeps improving every visit. A pre-trained **seed brain**
  (`src/rl-seed.json`) ships in the repo so the cubes already look
  competent on the very first load.

Because GitHub Pages is a static host, learning runs in *your* browser
while the page is open — that is the "watch them learn" part. Use the
speed buttons to run more training episodes per frame.

## History: the neuroevolution version

CodePurple was originally built with **neuroevolution** (evolving
populations of `15 → 20 → 16 → 2` tanh nets via selection / crossover /
mutation, with a Hall of Fame). It worked but kept settling into a
passive, twitchy mutual-avoidance equilibrium — a known weakness of
two-population co-evolution for pursuit-evasion. The DQN above replaced
it as the live build. The full neuroevolution implementation is preserved
on the [`neuroevolution`](https://github.com/maxlirio/CodePurple/tree/neuroevolution)
branch (`src/sim/evolve.js`, `scripts/train-seed.mjs`).

## Run locally

```bash
cd CodePurple
npm run serve        # then open http://localhost:5173
```

## Re-train the seed brain

```bash
npm run seed:dqn                      # ~600 episodes (default)
node src/rl/train-dqn.mjs 1000 200    # 1000 episodes, max 200s
```

This regenerates `src/rl-seed.json`. Commit it to ship a smarter starting
point. (The archived neuroevolution trainer is `npm run seed` on the
`neuroevolution` branch.)

## License

MIT
