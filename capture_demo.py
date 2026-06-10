"""Headless demo: run pick-and-place and save camera screenshots for the README."""

import os
from pathlib import Path

import mujoco
import PIL.Image

import skills
from sim_env import SimEnv

OUT = Path("docs/images")
CAMERAS = ("front_34", "wrist_cam")


def save_frame(env, camera, path, width=640, height=480):
    renderer = mujoco.Renderer(env.model, height, width)
    try:
        renderer.update_scene(env.data, camera=camera)
        img = PIL.Image.fromarray(renderer.render())
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        print(f"saved {path}")
    finally:
        renderer.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = SimEnv(show_viewer=False)
    env.step_settle(200)
    env.randomize_boxes(seed=42)

    save_frame(env, "front_34", OUT / "01_initial_scene.png")
    save_frame(env, "wrist_cam", OUT / "02_wrist_initial.png")

    print("picking red box...")
    skills.pick(env, "box_red")
    save_frame(env, "front_34", OUT / "03_holding_red.png")

    print("placing on shelf...")
    skills.place(env, "shelf")
    env.step_settle(300)
    save_frame(env, "front_34", OUT / "04_red_on_shelf.png")
    save_frame(env, "wrist_cam", OUT / "05_wrist_after_place.png")

    env.close()
    print("done — screenshots in docs/images/")


if __name__ == "__main__":
    main()
