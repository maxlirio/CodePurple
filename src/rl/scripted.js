// Hand-written "teacher" policies, used only to generate demonstrations
// for behavior cloning (so red's net is born already knowing the long
// grab->carry->place chain it could never explore into on its own).

import { lineOccluded, nearestPickable, CFG } from "../sim/world.js";
import { ACTIONS, INTERACT } from "./dqn.js";

// Nearest of the 8 compass move-actions to a direction vector (0 = stay).
function dirAction(dx, dz) {
  if (dx * dx + dz * dz < 1e-6) return 0;
  const ang = Math.atan2(dz, dx);
  let best = 1, bd = Infinity;
  for (let k = 1; k < ACTIONS.length; k++) {
    const ka = ((k - 1) / 8) * Math.PI * 2;
    let diff = Math.abs(ang - ka) % (Math.PI * 2);
    if (diff > Math.PI) diff = Math.PI * 2 - diff;
    if (diff < bd) { bd = diff; best = k; }
  }
  return best;
}

// Builder: if exposed, fetch the nearest block and wall blue off; once
// carrying, place it (placement drops it toward blue -> cover). When
// already hidden, just flee directly away from blue. Repeatedly getting
// re-exposed makes it stack several blocks -> a structure.
export function scriptedEvader(w) {
  const me = w.evader, foe = w.pursuer;
  if (me.carry) return INTERACT; // place it (toward blue)
  const hidden = lineOccluded(w, foe.x, foe.z, me.x, me.z);
  if (!hidden) {
    const np = nearestPickable(w, me.x, me.z);
    if (np) {
      if (np.dist <= CFG.PICK_R) return INTERACT; // grab
      return dirAction(np.ref.x - me.x, np.ref.z - me.z); // go to block
    }
  }
  return dirAction(me.x - foe.x, me.z - foe.z); // flee
}

// Chaser: drive straight at red. Good enough as a sparring opponent for
// demonstrations.
export function scriptedPursuer(w) {
  return dirAction(w.evader.x - w.pursuer.x, w.evader.z - w.pursuer.z);
}
