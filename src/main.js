// Browser entry point: renders the live arena with Three.js. You watch the
// reigning champion red vs champion blue duel at real time, while many
// generations of co-evolution run in the background every second. Learning
// never stops and is persisted to localStorage; a committed seed brain is
// used on the first visit.

import * as THREE from "three";
import { createWorld, reset, step, CFG } from "./sim/world.js";
import {
  createEvolution, trainStep, getShowcase, genProgress,
  serialize, deserialize, POP,
} from "./sim/evolve.js";

const LS_KEY = "codepurple.v2";

// ---- Load brains: saved progress > committed seed > random ----
async function loadState() {
  try {
    const saved = localStorage.getItem(LS_KEY);
    if (saved) {
      const s = deserialize(JSON.parse(saved));
      if (s) return { state: s, src: "your saved training" };
    }
  } catch (e) { /* ignore corrupt storage */ }
  try {
    const r = await fetch("./src/seed.json", { cache: "no-cache" });
    if (r.ok) {
      const s = deserialize(await r.json());
      if (s) return { state: s, src: "seed brain" };
    }
  } catch (e) { /* no seed committed yet */ }
  return { state: createEvolution(), src: "fresh (random)" };
}

const { state, src } = await loadState();

function save() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(serialize(state))); }
  catch (e) { /* storage full / blocked */ }
}

// ---- Scene ----
const app = document.getElementById("app");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0512);
scene.fog = new THREE.Fog(0x0a0512, 40, 95);

const camera = new THREE.PerspectiveCamera(
  55, innerWidth / innerHeight, 0.1, 500
);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

scene.add(new THREE.AmbientLight(0x6a4fae, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(18, 34, 14);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.left = key.shadow.camera.bottom = -CFG.A - 4;
key.shadow.camera.right = key.shadow.camera.top = CFG.A + 4;
scene.add(key);
scene.add(new THREE.PointLight(0xff5a7a, 0.5, 60));

const A = CFG.A;
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(A * 2, A * 2),
  new THREE.MeshStandardMaterial({ color: 0x140a26, roughness: 0.95 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(A * 2, 32, 0x6a4fae, 0x2c1a47);
grid.position.y = 0.01;
scene.add(grid);

// Glowing arena walls.
const wallMat = new THREE.MeshStandardMaterial({
  color: 0x3a2360, emissive: 0x5a3aa0, emissiveIntensity: 0.45,
  transparent: true, opacity: 0.5,
});
for (const [sx, sz, x, z] of [
  [A * 2, 0.4, 0, A], [A * 2, 0.4, 0, -A],
  [0.4, A * 2, A, 0], [0.4, A * 2, -A, 0],
]) {
  const wall = new THREE.Mesh(new THREE.BoxGeometry(sx, 3, sz), wallMat);
  wall.position.set(x, 1.5, z);
  scene.add(wall);
}

function cube(color, emissive) {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(CFG.AGENT_R * 2, CFG.AGENT_R * 2, CFG.AGENT_R * 2),
    new THREE.MeshStandardMaterial({
      color, emissive, emissiveIntensity: 0.6, roughness: 0.35,
    })
  );
  m.castShadow = true;
  m.position.y = CFG.AGENT_R;
  scene.add(m);
  return m;
}
const redMesh = cube(0xff3b63, 0x7a0f28);
const blueMesh = cube(0x3b8bff, 0x0f3a7a);

const blockMat = new THREE.MeshStandardMaterial({
  color: 0x8a78b0, roughness: 0.7, metalness: 0.1,
});
const blockMeshes = Array.from({ length: CFG.N_BLOCKS }, () => {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(CFG.BLOCK_H * 2, 2, CFG.BLOCK_H * 2), blockMat
  );
  m.castShadow = true;
  m.receiveShadow = true;
  m.position.y = 1;
  scene.add(m);
  return m;
});

// ---- The exhibition match (champion vs champion, real time) ----
const w = createWorld();
let show = getShowcase(state);
reset(w);
let acc = 0;            // real-time physics accumulator
let last = performance.now();
let holdUntil = 0;      // brief pause after a round ends
let learnX = 1;         // background learning throughput (1×–16×)
let paused = false;
let lastSavedGen = state.gen;

const $ = (id) => document.getElementById(id);
$("src").textContent = src;
const banner = $("banner");

function showBanner(text, color) {
  banner.textContent = text;
  banner.style.color = color;
  banner.style.display = "block";
}

function startRound() {
  show = getShowcase(state); // pick up any newly crowned champions
  reset(w);
  banner.style.display = "none";
}

// ---- Main loop ----
let frame = 0;
function animate(now) {
  requestAnimationFrame(animate);
  const dtReal = Math.min((now - last) / 1000, 0.1);
  last = now;

  if (!paused) {
    // Background learning: run trainStep many times per frame.
    for (let i = 0; i < learnX * 2; i++) {
      const { genCompleted } = trainStep(state);
      if (genCompleted && state.gen - lastSavedGen >= 5) {
        save();
        lastSavedGen = state.gen;
      }
    }

    // Foreground exhibition match advances at true real time.
    if (now >= holdUntil) {
      if (w.over) {
        startRound();
      } else {
        acc += dtReal;
        let guard = 0;
        while (acc >= CFG.DT && !w.over && guard++ < 8) {
          step(w, show.evaderGenome, show.pursuerGenome);
          acc -= CFG.DT;
        }
        if (w.over) {
          const escaped = w.outcome === "escaped";
          showBanner(
            escaped ? "RED ESCAPED — RED WINS" : "BLUE CAUGHT RED",
            escaped ? "#ff5a7a" : "#5ab4ff"
          );
          holdUntil = now + 1400;
        }
      }
    }
  }

  redMesh.position.set(w.evader.x, CFG.AGENT_R, w.evader.z);
  blueMesh.position.set(w.pursuer.x, CFG.AGENT_R, w.pursuer.z);
  redMesh.rotation.y += 0.03;
  blueMesh.rotation.y -= 0.03;
  for (let i = 0; i < blockMeshes.length; i++) {
    blockMeshes[i].position.x = w.blocks[i].x;
    blockMeshes[i].position.z = w.blocks[i].z;
  }

  // Slow cinematic orbit.
  const a = now * 0.00007;
  camera.position.set(Math.sin(a) * 40, 30, Math.cos(a) * 40);
  camera.lookAt(0, 0, 0);

  if ((frame++ & 7) === 0) {
    $("gen").textContent = state.gen.toLocaleString();
    $("ind").textContent = `${Math.round(genProgress(state) * 100)}% of gen`;
    $("surv").textContent = `${w.t.toFixed(1)}s / ${CFG.ESCAPE_T}s`;
    $("bestE").textContent =
      isFinite(state.bestEvaderFit) ? state.bestEvaderFit.toFixed(1) : "—";
    $("bestP").textContent =
      isFinite(state.bestPursuerFit) ? state.bestPursuerFit.toFixed(1) : "—";
  }
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// ---- Controls ----
document.querySelectorAll("#ctrl button[data-spd]").forEach((b) => {
  b.addEventListener("click", () => {
    learnX = +b.dataset.spd;
    document.querySelectorAll("#ctrl button[data-spd]")
      .forEach((x) => x.classList.toggle("on", x === b));
  });
});
$("pause").addEventListener("click", (e) => {
  paused = !paused;
  e.target.textContent = paused ? "▶ resume" : "⏸ pause";
});
$("reset").addEventListener("click", () => {
  if (!confirm("Erase this browser's learned brains and start over?")) return;
  localStorage.removeItem(LS_KEY);
  location.reload();
});
addEventListener("beforeunload", save);
setInterval(save, 20000); // periodic safety save
