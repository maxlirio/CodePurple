// Browser entry point (DQN build): renders the live arena with Three.js.
// You watch the two Deep Q-Network cubes play greedily at real time, while
// they keep learning by reinforcement (epsilon-greedy episodes) in the
// background. Learning never stops and is persisted to localStorage; a
// committed DQN seed is used on the first visit.

import * as THREE from "three";
import {
  createWorld, reset, sense, applyStep, CFG,
  addStatic, addPusher, removeObject, clearUserObjects,
} from "./sim/world.js";
import {
  createAgent, act, serializeAgent, deserializeAgent,
} from "./rl/dqn.js";
import { actionVel, trainEpisode, INTERACT } from "./rl/env.js";

// v2: cubes can now carry & place blocks (new action + 3 new sensors),
// so the brain shape changed and old saved brains are superseded.
const LS_KEY = "codepurple.rl.v2";

// ---- Load brains: saved progress > committed seed > fresh ----
async function loadAgents() {
  try {
    const saved = localStorage.getItem(LS_KEY);
    if (saved) {
      const d = JSON.parse(saved);
      const e = deserializeAgent(d.evader), p = deserializeAgent(d.pursuer);
      if (e && p) return { agE: e, agP: p, ep: d.ep || 0,
                           src: "your saved training" };
    }
  } catch (e) { /* ignore corrupt storage */ }
  try {
    const r = await fetch("./src/rl-seed.json", { cache: "no-cache" });
    if (r.ok) {
      const d = await r.json();
      const e = deserializeAgent(d.evader), p = deserializeAgent(d.pursuer);
      if (e && p) return { agE: e, agP: p, ep: 0, src: "seed brain" };
    }
  } catch (e) { /* no seed committed yet */ }
  return { agE: createAgent(), agP: createAgent(), ep: 0,
           src: "fresh (random)" };
}

const { agE, agP, ep: ep0, src } = await loadAgents();
let episodes = ep0;

function save() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      evader: serializeAgent(agE),
      pursuer: serializeAgent(agP),
      ep: episodes,
    }));
  } catch (e) { /* storage full / blocked */ }
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
reset(w);
let sE = Float64Array.from(sense(w, "evader"));
let sP = Float64Array.from(sense(w, "pursuer"));
let acc = 0;            // real-time physics accumulator
let last = performance.now();
let holdUntil = 0;      // brief pause after a round ends
let learnX = 1;         // background learning throughput (1×–16×)
let paused = false;
let savedAtEp = episodes;
const EXPLORE = 0.1;    // residual exploration while it keeps learning
let catches = 0, rounds = 0, sumCatchT = 0; // rolling exhibition stats
let orbitT = 0;         // camera orbit time (frozen while interacting)
let tool = "move";      // active sandbox tool
let pointerDown = false;
let dragObj = null;     // { kind, ref, limit }

// ---- Sandbox objects (user-placed walls & blocks) ----
redMesh.userData = { kind: "evader", ref: w.evader };
blueMesh.userData = { kind: "pursuer", ref: w.pursuer };

const wallMatU = new THREE.MeshStandardMaterial({
  color: 0x00e6c8, emissive: 0x0a5a50, emissiveIntensity: 0.5, roughness: 0.5,
});
const pushMatU = new THREE.MeshStandardMaterial({
  color: 0xffb13b, emissive: 0x6a3a00, emissiveIntensity: 0.45, roughness: 0.6,
});
const sandGeo = new THREE.BoxGeometry(CFG.BLOCK_H * 2, 2.2, CFG.BLOCK_H * 2);
const staticMeshes = []; // parallel to w.statics
const userMeshes = [];   // parallel to w.userBlocks

function spawnMesh(mat, ref, kind, list) {
  const m = new THREE.Mesh(sandGeo, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  m.position.set(ref.x, 1.1, ref.z);
  m.userData = { kind, ref };
  scene.add(m);
  list.push(m);
  return m;
}
const placeWall = (x, z) =>
  spawnMesh(wallMatU, addStatic(w, x, z), "static", staticMeshes);
const placeBlock = (x, z) =>
  spawnMesh(pushMatU, addPusher(w, x, z), "user", userMeshes);

function destroyMesh(mesh) {
  removeObject(w, mesh.userData.ref);
  for (const list of [staticMeshes, userMeshes]) {
    const i = list.indexOf(mesh);
    if (i >= 0) list.splice(i, 1);
  }
  scene.remove(mesh);
}

const PLAY_KEY = "codepurple.play.v1";
function saveLayout() {
  try {
    localStorage.setItem(PLAY_KEY, JSON.stringify({
      statics: w.statics.map((o) => ({ x: o.x, z: o.z })),
      blocks: w.userBlocks.map((o) => ({ x: o.x, z: o.z })),
    }));
  } catch (e) { /* ignore */ }
}
try {
  const d = JSON.parse(localStorage.getItem(PLAY_KEY) || "null");
  if (d) {
    (d.statics || []).forEach((o) => placeWall(o.x, o.z));
    (d.blocks || []).forEach((o) => placeBlock(o.x, o.z));
  }
} catch (e) { /* ignore */ }

const $ = (id) => document.getElementById(id);
$("src").textContent = src;
const banner = $("banner");

function showBanner(text, color) {
  banner.textContent = text;
  banner.style.color = color;
  banner.style.display = "block";
}

function startRound() {
  reset(w);
  sE = Float64Array.from(sense(w, "evader"));
  sP = Float64Array.from(sense(w, "pursuer"));
  banner.style.display = "none";
}

// ---- Main loop ----
let frame = 0;
function animate(now) {
  requestAnimationFrame(animate);
  const dtReal = Math.min((now - last) / 1000, 0.1);
  last = now;

  if (!paused) {
    // Background reinforcement learning: full epsilon-greedy episodes.
    for (let i = 0; i < learnX; i++) {
      trainEpisode(agE, agP, EXPLORE, EXPLORE, 15);
      episodes++;
      if (episodes - savedAtEp >= 20) { save(); savedAtEp = episodes; }
    }

    // Foreground exhibition match: the same brains, played near-greedily
    // at true real time. Freezes while you interact so the scene is stable.
    if (now >= holdUntil && !pointerDown) {
      if (w.over) {
        startRound();
      } else {
        acc += dtReal;
        let guard = 0;
        while (acc >= CFG.DT && !w.over && guard++ < 8) {
          const aE = act(agE, sE, 0.02);
          const aP = act(agP, sP, 0.02);
          const [evx, evz] = actionVel(aE, CFG.SPEED_E);
          const [pvx, pvz] = actionVel(aP, CFG.SPEED_P);
          applyStep(w, evx, evz, pvx, pvz, aE === INTERACT, aP === INTERACT);
          sE = Float64Array.from(sense(w, "evader"));
          sP = Float64Array.from(sense(w, "pursuer"));
          acc -= CFG.DT;
        }
        if (w.over) {
          const escaped = w.outcome === "escaped";
          rounds++;
          if (!escaped) { catches++; sumCatchT += w.t; }
          if (rounds >= 30) { rounds = catches = sumCatchT = 0; }
          showBanner(
            escaped ? "RED ESCAPED — RED WINS" : "BLUE CAUGHT RED",
            escaped ? "#ff5a7a" : "#5ab4ff"
          );
          holdUntil = now + 1400;
        }
      }
    }
  }

  // Visual-only smoothing: ease each cube's *rendered* position toward its
  // true simulation position. Removes the 30 Hz jitter to the eye without
  // touching the physics, sensors, catch detection or learning. Large jumps
  // (round reset, dragging) snap instead of sliding across the arena.
  const k = 1 - Math.exp(-16 * dtReal);
  const smooth = (mesh, x, z) => {
    const dx = x - mesh.position.x, dz = z - mesh.position.z;
    if (Math.hypot(dx, dz) > 3) mesh.position.set(x, CFG.AGENT_R, z);
    else { mesh.position.x += dx * k; mesh.position.z += dz * k; }
  };
  smooth(redMesh, w.evader.x, w.evader.z);
  smooth(blueMesh, w.pursuer.x, w.pursuer.z);
  redMesh.rotation.y += 0.03;
  blueMesh.rotation.y -= 0.03;
  // A carried block hovers over its carrier's head; otherwise it sits on
  // the ground at its sim position.
  const carrierMesh = (b) =>
    b === w.evader.carry ? redMesh
      : b === w.pursuer.carry ? blueMesh : null;
  const placeBlockMesh = (mesh, b) => {
    const cm = carrierMesh(b);
    if (cm) mesh.position.set(cm.position.x, 2.8, cm.position.z);
    else mesh.position.set(b.x, 1, b.z);
  };
  for (let i = 0; i < blockMeshes.length; i++)
    placeBlockMesh(blockMeshes[i], w.blocks[i]);
  for (let i = 0; i < userMeshes.length; i++)
    placeBlockMesh(userMeshes[i], w.userBlocks[i]);
  for (let i = 0; i < staticMeshes.length; i++) {
    staticMeshes[i].position.x = w.statics[i].x;
    staticMeshes[i].position.z = w.statics[i].z;
  }

  // Slow cinematic orbit, paused while you interact with the scene.
  if (!pointerDown && !paused) orbitT += dtReal;
  const a = orbitT * 0.07;
  camera.position.set(Math.sin(a) * 40, 30, Math.cos(a) * 40);
  camera.lookAt(0, 0, 0);

  if ((frame++ & 7) === 0) {
    $("gen").textContent = episodes.toLocaleString();
    $("ind").textContent = EXPLORE.toFixed(2);
    $("surv").textContent = `${w.t.toFixed(1)}s / ${CFG.ESCAPE_T}s`;
    $("bestP").textContent =
      rounds ? `${Math.round((100 * catches) / rounds)}%` : "—";
    $("bestE").textContent =
      catches ? `${(sumCatchT / catches).toFixed(1)}s` : "—";
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
// ---- Sandbox tools & drag interaction ----
const cv = renderer.domElement;
function setCursor() {
  cv.style.cursor =
    tool === "erase" ? "not-allowed" : tool === "move" ? "grab" : "crosshair";
}
document.querySelectorAll("#tools button[data-tool]").forEach((b) => {
  b.addEventListener("click", () => {
    tool = b.dataset.tool;
    document.querySelectorAll("#tools button[data-tool]")
      .forEach((x) => x.classList.toggle("on", x === b));
    setCursor();
  });
});
$("clearObj").addEventListener("click", () => {
  [...staticMeshes, ...userMeshes].forEach((m) => scene.remove(m));
  staticMeshes.length = 0;
  userMeshes.length = 0;
  clearUserObjects(w);
  saveLayout();
});

const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const hitPt = new THREE.Vector3();

function groundAt(ev) {
  const r = cv.getBoundingClientRect();
  ndc.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
  ray.setFromCamera(ndc, camera);
  return ray.ray.intersectPlane(groundPlane, hitPt) ? hitPt : null;
}
function pickObject() {
  const hits = ray.intersectObjects(
    [redMesh, blueMesh, ...staticMeshes, ...userMeshes], false
  );
  return hits.length ? hits[0].object : null;
}
const clampTo = (p, lim) => {
  p.x = Math.max(-lim, Math.min(lim, p.x));
  p.z = Math.max(-lim, Math.min(lim, p.z));
};

cv.addEventListener("pointerdown", (ev) => {
  const g = groundAt(ev);
  if (!g) return;
  pointerDown = true;
  try { cv.setPointerCapture(ev.pointerId); } catch (e) { /* ok */ }
  const obj = pickObject();

  if (tool === "erase") {
    if (obj && (obj.userData.kind === "static" || obj.userData.kind === "user")) {
      destroyMesh(obj);
      saveLayout();
    }
    return;
  }
  if (tool === "wall" || tool === "push") {
    const m = tool === "wall" ? placeWall(g.x, g.z) : placeBlock(g.x, g.z);
    dragObj = { kind: m.userData.kind, ref: m.userData.ref,
                lim: CFG.A - CFG.BLOCK_H };
    return;
  }
  if (obj) { // move tool: grab whatever is under the cursor
    const k = obj.userData.kind;
    dragObj = {
      kind: k, ref: obj.userData.ref,
      lim: (k === "evader" || k === "pursuer")
        ? CFG.A - CFG.AGENT_R : CFG.A - CFG.BLOCK_H,
    };
  }
});
cv.addEventListener("pointermove", (ev) => {
  if (!pointerDown || !dragObj) return;
  const g = groundAt(ev);
  if (!g) return;
  const ref = dragObj.ref;
  ref.x = g.x; ref.z = g.z;
  clampTo(ref, dragObj.lim);
  if (dragObj.kind === "evader" || dragObj.kind === "pursuer") {
    ref.vx = ref.vz = 0; // drop it cleanly; the brain resumes control
  }
});
function endDrag(ev) {
  if (!pointerDown) return;
  pointerDown = false;
  if (dragObj && (dragObj.kind === "static" || dragObj.kind === "user")) {
    saveLayout();
  }
  dragObj = null;
  try { cv.releasePointerCapture(ev.pointerId); } catch (e) { /* ok */ }
}
cv.addEventListener("pointerup", endDrag);
cv.addEventListener("pointercancel", endDrag);
setCursor();

addEventListener("beforeunload", save);
setInterval(save, 20000); // periodic safety save
