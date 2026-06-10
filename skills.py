"""High-level manipulation skills (find_object, pick, place) for SimEnv.

Grasping does not rely on contact friction: pick() activates the "grasp_weld"
equality constraint between the Panda hand and the target body (with the weld
relpose set to the current hand-object relative pose so nothing snaps), and
place() deactivates it to release.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Gripper-pointing-down orientation (wxyz), matching the Panda home pose.
DOWN_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

BOX_HALF = 0.02  # half-size of the 4cm pickable box
PREGRASP_HEIGHT = 0.12
LIFT_HEIGHT = 0.20
PLACE_APPROACH_HEIGHT = 0.10
SAFE_Z = 0.84  # transit height: clears the shelf (top 0.66) and held boxes


def _wxyz_to_rot(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])  # scipy wants xyzw


def _rot_to_wxyz(r):
    return r.as_quat()[[3, 0, 1, 2]]


def find_object(env, name):
    """Return world (pos, quat[wxyz]) of a named body (or static geom)."""
    try:
        b = env.data.body(name)
        return b.xpos.copy(), b.xquat.copy()
    except KeyError:
        g = env.data.geom(name)
        quat = _rot_to_wxyz(Rotation.from_matrix(g.xmat.reshape(3, 3)))
        return g.xpos.copy(), quat


def _set_weld(env, object_name=None, active=False):
    """Toggle the grasp weld. When activating, retarget body2 to `object_name`
    and set the weld relpose to the CURRENT hand-object relative pose."""
    eq_id = env.model.eq("grasp_weld").id
    if not active:
        env.data.eq_active[eq_id] = 0
        return

    hand_pos = env.data.body("hand").xpos
    hand_rot = _wxyz_to_rot(env.data.body("hand").xquat)
    obj_pos = env.data.body(object_name).xpos
    obj_rot = _wxyz_to_rot(env.data.body(object_name).xquat)

    relpos = hand_rot.inv().apply(obj_pos - hand_pos)
    relquat = _rot_to_wxyz(hand_rot.inv() * obj_rot)

    env.model.eq_obj2id[eq_id] = env.model.body(object_name).id
    env.model.eq_data[eq_id, 0:3] = 0.0  # anchor
    env.model.eq_data[eq_id, 3:6] = relpos
    env.model.eq_data[eq_id, 6:10] = relquat
    env.data.eq_active[eq_id] = 1


def _safe_move(env, pos, quat=DOWN_QUAT):
    """Move the EE to `pos` via a transit waypoint at SAFE_Z so straight-line
    paths never cut through the shelf or other obstacles."""
    cur, _ = env.get_ee_pose()
    if cur[2] < SAFE_Z - 0.02:
        env.move_to_pose([cur[0], cur[1], SAFE_Z], quat, duration=1.5)
    env.move_to_pose([pos[0], pos[1], SAFE_Z], quat, duration=2.5)
    ok = env.move_to_pose(pos, quat, duration=2.5)
    env.step_settle(50)
    return ok


def pick(env, object_name):
    """Pick up `object_name`: pre-grasp above, descend, verify, weld + close,
    lift. Raises RuntimeError if the gripper could not reach the object."""
    env.set_gripper("open")

    obj_pos, _ = find_object(env, object_name)
    pregrasp = obj_pos + [0, 0, PREGRASP_HEIGHT]
    _safe_move(env, pregrasp)

    for attempt in range(2):
        obj_pos, _ = find_object(env, object_name)
        grasp = obj_pos + [0, 0, 0.005]  # TCP at object center
        env.move_to_pose(grasp, DOWN_QUAT, duration=2.5)
        env.step_settle(100)
        ee, _ = env.get_ee_pose()
        if np.linalg.norm(ee - grasp) < 0.02:
            break
        if attempt == 0:  # retry once from directly above
            env.move_to_pose(obj_pos + [0, 0, PREGRASP_HEIGHT], DOWN_QUAT, duration=2.0)
            env.step_settle(50)
    else:
        raise RuntimeError(f"Gripper could not reach {object_name!r}.")

    env.set_gripper("closed", settle_steps=200)
    # Sanity check before welding: the box must actually be between the fingers.
    ee, _ = env.get_ee_pose()
    obj_pos, _ = find_object(env, object_name)
    if np.linalg.norm(ee - obj_pos) > 0.04:
        env.set_gripper("open", settle_steps=100)
        raise RuntimeError(f"Grasp on {object_name!r} failed (object not in gripper).")
    _set_weld(env, object_name, active=True)
    env.step_settle(100)

    lift = obj_pos + [0, 0, LIFT_HEIGHT]
    env.move_to_pose(lift, DOWN_QUAT, duration=2.0)
    env.step_settle(100)


def _find_free_spot(env, location_name, clearance=0.065):
    """Return [x, y, surface_z] of a free spot on top of the location geom,
    or None if the surface is full. Scans grid candidates from the center out,
    keeping `clearance` distance from boxes already resting on the surface."""
    gid = env.model.geom(location_name).id
    center = env.data.geom(location_name).xpos
    half_x, half_y, half_z = env.model.geom_size[gid]
    surface_z = center[2] + half_z

    occupied = []
    for name in env.box_body_names():
        p = env.data.body(name).xpos
        if (
            surface_z - 0.005 < p[2] < surface_z + 0.05
            and abs(p[0] - center[0]) < half_x + 0.05
            and abs(p[1] - center[1]) < half_y + 0.05
        ):
            occupied.append(p[:2])

    margin = BOX_HALF + 0.01
    xs = np.arange(center[0] - half_x + margin, center[0] + half_x - margin + 1e-9, 0.03)
    ys = np.arange(center[1] - half_y + margin, center[1] + half_y - margin + 1e-9, 0.03)
    candidates = [(x, y) for x in xs for y in ys]
    candidates.sort(key=lambda c: np.hypot(c[0] - center[0], c[1] - center[1]))
    for x, y in candidates:
        if all(np.hypot(x - ox, y - oy) > clearance for ox, oy in occupied):
            return np.array([x, y, surface_z])
    return None


def place(env, location_name):
    """Place the held object on a free spot on `location_name`, then release.

    Raises RuntimeError if there is no free space left on the surface."""
    spot = _find_free_spot(env, location_name)
    if spot is None:
        raise RuntimeError(f"No free space left on {location_name!r}.")
    loc_pos = spot
    surface_z = spot[2]

    # EE carries the object at its TCP, so offset by the object half-height.
    approach = np.array(
        [loc_pos[0], loc_pos[1], surface_z + BOX_HALF + PLACE_APPROACH_HEIGHT]
    )
    _safe_move(env, approach)

    lower = np.array([loc_pos[0], loc_pos[1], surface_z + BOX_HALF + 0.01])
    env.move_to_pose(lower, DOWN_QUAT, duration=2.0)
    env.step_settle(100)

    _set_weld(env, active=False)
    env.set_gripper("open", settle_steps=200)

    env.move_to_pose(approach, DOWN_QUAT, duration=2.0)
    env.step_settle(100)
    cur, _ = env.get_ee_pose()
    env.move_to_pose([cur[0], cur[1], SAFE_Z], DOWN_QUAT, duration=1.5)
    env.step_settle(100)


if __name__ == "__main__":
    import time

    from sim_env import SimEnv

    env = SimEnv()
    env.step_settle(200)

    pick(env, "box_red")
    box_pos, _ = find_object(env, "box_red")
    print("after pick, box at", np.round(box_pos, 3))

    place(env, "shelf")
    env.step_settle(500)  # let the box come to rest

    box_pos, _ = find_object(env, "box_red")
    shelf_pos, _ = find_object(env, "shelf")
    shelf_top = shelf_pos[2] + 0.01
    on_shelf = (
        abs(box_pos[0] - shelf_pos[0]) < 0.15
        and abs(box_pos[1] - shelf_pos[1]) < 0.12
        and abs(box_pos[2] - (shelf_top + BOX_HALF)) < 0.01
    )
    print("box at", np.round(box_pos, 3), "shelf top z", shelf_top)
    print("SUCCESS: box resting on shelf" if on_shelf else "FAILURE: box not on shelf")

    time.sleep(1.0)
    env.close()
