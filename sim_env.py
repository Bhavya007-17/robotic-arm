"""Synchronous MuJoCo simulation environment for the Panda pick-and-place scene.

All methods run in the calling (main) thread: they step the simulation and sync
the passive viewer themselves. No background sim thread.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    import mink

    HAVE_MINK = True
except ImportError:
    HAVE_MINK = False

SCENE_XML = "models/scene.xml"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0


class SimEnv:
    def __init__(self, xml_path=SCENE_XML, show_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)

        self.ee_site_id = self.model.site("ee_site").id
        self.arm_qpos_ids = np.array(
            [self.model.joint(name).qposadr[0] for name in ARM_JOINTS]
        )
        self.arm_dof_ids = np.array(
            [self.model.joint(name).dofadr[0] for name in ARM_JOINTS]
        )
        # actuator1..actuator7 are position servos on the arm joints.
        self.arm_act_ids = np.array(
            [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self.gripper_act_id = self.model.actuator(GRIPPER_ACTUATOR).id

        self.viewer = (
            mujoco.viewer.launch_passive(self.model, self.data) if show_viewer else None
        )

    # ------------------------------------------------------------------ basics

    def _step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def step_settle(self, n):
        """Step the simulation n times, syncing the viewer each step."""
        for _ in range(n):
            self._step()

    def box_body_names(self):
        """Names of all pickable box bodies in the scene."""
        return [
            self.model.body(i).name
            for i in range(self.model.nbody)
            if self.model.body(i).name.startswith("box_")
        ]

    def randomize_boxes(self, seed=None, settle_steps=200):
        """Scatter the boxes to random, non-overlapping reachable table spots."""
        rng = np.random.default_rng(seed)
        placed = []
        for name in self.box_body_names():
            for _ in range(200):
                x = rng.uniform(0.33, 0.62)
                y = rng.uniform(-0.26, 0.24)
                if np.hypot(x, y) > 0.68:  # keep within comfortable reach
                    continue
                if all(np.hypot(x - px, y - py) > 0.09 for px, py in placed):
                    placed.append((x, y))
                    break
            joint = self.model.joint(f"{name}_free")
            adr, dof = joint.qposadr[0], joint.dofadr[0]
            self.data.qpos[adr : adr + 7] = [x, y, 0.43, 1, 0, 0, 0]
            self.data.qvel[dof : dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.step_settle(settle_steps)

    def get_ee_pose(self):
        """Return current end-effector (pos[3], quat[4] wxyz)."""
        pos = self.data.site_xpos[self.ee_site_id].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        return pos, quat

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ----------------------------------------------------------------- gripper

    def set_gripper(self, state, settle_steps=300):
        """Drive the gripper to 'open' or 'closed' and step until settled."""
        if state not in ("open", "closed"):
            raise ValueError(f"state must be 'open' or 'closed', got {state!r}")
        self.data.ctrl[self.gripper_act_id] = (
            GRIPPER_OPEN_CTRL if state == "open" else GRIPPER_CLOSED_CTRL
        )
        self.step_settle(settle_steps)

    # --------------------------------------------------------------------- IK

    def move_to_pose(self, pos, quat, duration=2.0, pos_tol=0.01, ori_tol=0.05):
        """Move the end-effector to (pos, quat[wxyz]) using differential IK.

        Runs a synchronous control loop: each iteration solves IK for joint
        velocities, integrates them into position targets for the arm's
        position actuators, steps the sim, and syncs the viewer. Stops when the
        measured EE pose is within tolerance or `duration` sim-seconds elapse.
        Returns True if the target was reached.
        """
        pos = np.asarray(pos, dtype=float)
        quat = np.asarray(quat, dtype=float)
        if HAVE_MINK:
            return self._move_mink(pos, quat, duration, pos_tol, ori_tol)
        return self._move_dls(pos, quat, duration, pos_tol, ori_tol)

    def _pose_error(self, pos, quat):
        cur_pos, cur_quat = self.get_ee_pose()
        pos_err = np.linalg.norm(pos - cur_pos)
        # Angle between quaternions.
        dot = min(1.0, abs(float(np.dot(quat, cur_quat))))
        ori_err = 2.0 * np.arccos(dot)
        return pos_err, ori_err

    def _move_mink(self, pos, quat, duration, pos_tol, ori_tol):
        dt = self.model.opt.timestep
        configuration = mink.Configuration(self.model)
        configuration.update(self.data.qpos.copy())

        ee_task = mink.FrameTask(
            frame_name="ee_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        ee_task.set_target(
            mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=quat), pos)
        )
        posture_task = mink.PostureTask(self.model, cost=1e-2)
        posture_task.set_target_from_configuration(configuration)
        tasks = [ee_task, posture_task]
        limits = [mink.ConfigurationLimit(self.model)]

        max_vel = 2.0  # rad/s clamp on arm joint speed for smooth motion
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            vel = mink.solve_ik(configuration, tasks, dt, "daqp", 1e-3, limits=limits)
            vel = vel.copy()
            # Only the arm moves: zero everything but the 7 arm dofs.
            mask = np.zeros_like(vel, dtype=bool)
            mask[self.arm_dof_ids] = True
            vel[~mask] = 0.0
            peak = np.max(np.abs(vel[self.arm_dof_ids]))
            if peak > max_vel:
                vel *= max_vel / peak
            configuration.integrate_inplace(vel, dt)

            self.data.ctrl[self.arm_act_ids] = configuration.q[self.arm_qpos_ids]
            self._step()

            pos_err, ori_err = self._pose_error(pos, quat)
            if pos_err < pos_tol and ori_err < ori_tol:
                return True
        return self._pose_error(pos, quat)[0] < pos_tol

    def _move_dls(self, pos, quat, duration, pos_tol, ori_tol, damping=1e-2, gain=2.0):
        """Fallback: damped-least-squares IK with mj_jacSite."""
        dt = self.model.opt.timestep
        q_target = self.data.qpos[self.arm_qpos_ids].copy()
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            cur_pos, cur_quat = self.get_ee_pose()
            pos_err = pos - cur_pos
            # Orientation error as a rotational velocity vector.
            quat_err = np.empty(4)
            quat_conj = np.empty(4)
            mujoco.mju_negQuat(quat_conj, cur_quat)
            mujoco.mju_mulQuat(quat_err, quat, quat_conj)
            ori_err = np.empty(3)
            mujoco.mju_quat2Vel(ori_err, quat_err, 1.0)

            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
                return True

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            J = np.vstack([jacp, jacr])[:, self.arm_dof_ids]
            err = gain * np.concatenate([pos_err, ori_err])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            dq = np.clip(dq, -2.0, 2.0)

            q_target = q_target + dq * dt
            self.data.ctrl[self.arm_act_ids] = q_target
            self._step()
        return self._pose_error(pos, quat)[0] < pos_tol


if __name__ == "__main__":
    env = SimEnv()
    print("IK backend:", "mink" if HAVE_MINK else "DLS fallback")
    env.step_settle(200)

    _, down_quat = env.get_ee_pose()  # home pose already points the gripper down
    box_pos = env.data.body("box_red").xpos.copy()
    print("box at", np.round(box_pos, 3))

    env.set_gripper("open")

    for label, target in [
        ("above box", box_pos + [0, 0, 0.15]),
        ("at box", box_pos + [0, 0, 0.01]),
        ("lift", box_pos + [0, 0, 0.20]),
    ]:
        ok = env.move_to_pose(target, down_quat, duration=3.0)
        env.step_settle(100)
        ee, _ = env.get_ee_pose()
        err = np.linalg.norm(ee - target)
        print(f"{label}: reached={ok} ee={np.round(ee, 3)} err={err * 1000:.1f} mm")

    time.sleep(1.0)
    env.close()
