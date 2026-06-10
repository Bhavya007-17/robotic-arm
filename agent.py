"""Robot arm agent controlled via natural language.

Uses a fast local parser for common pick/place commands, with NVIDIA NIM
(google/gemma-4-31b-it) as fallback for ambiguous requests.
Reads NVIDIA_API_KEY from the environment.
"""

import base64
import io
import json
import os
import re
import sys
import time

import mujoco
import PIL.Image
import requests
from langchain_core.tools import tool

import skills
from sim_env import SimEnv

ENV = None  # single shared SimEnv, set in main()
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = os.environ.get("NIM_MODEL", "google/gemma-4-31b-it")
NIM_API_KEY = None  # set in build_agent()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "look_at_scene",
            "description": (
                "Look at the workspace cameras and describe visible colored boxes."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_object",
            "description": "Return world position of a box color or shelf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "red, green, blue, yellow, or shelf",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Pick up a colored box with the gripper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "enum": ["red", "green", "blue", "yellow"],
                    }
                },
                "required": ["object_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": "Place the held box onto the shelf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {"type": "string", "enum": ["shelf"]}
                },
                "required": ["location_name"],
            },
        },
    },
]

BOX_COLORS = ("red", "green", "blue", "yellow")
CAMERAS = ("front_34", "wrist_cam")

SYSTEM_PROMPT = """You control a Franka Panda robot arm in a physics simulation.
On the table in front of the arm there are several small colored boxes and a
shelf mounted above the table.

You have four tools:
- look_at_scene(): capture images from the external camera and the wrist camera
  and get a visual description of the colored boxes currently visible.
- find_object(name): look up the exact position of an object.
- pick(object_name): pick up a box with the gripper.
- place(location_name): place the currently held box on a location.

Valid objects: "red", "green", "blue", "yellow" (the colored boxes).
Valid locations: "shelf".
Map user phrasing to these names (e.g. "the blue cube" -> "blue").

For pick/place by color name (red, green, blue, yellow), call pick or place
directly — do NOT call look_at_scene first. Only use look_at_scene when the user
asks what you see or which boxes are visible. Use find_object if you need exact
coordinates. Call tools in the correct order (you must pick before you can
place). Placement automatically finds a free spot on the target surface; if the
place tool reports the surface is full, tell the user instead of retrying.

IMPORTANT: keep calling tools until the user's ENTIRE request is fulfilled.
For example "put the green box on the shelf" requires TWO tool calls:
pick("green") and then place("shelf"). Never stop after picking without
placing. Only respond with plain text once everything is done (briefly report
what you did) or the request is impossible (say why).
"""


def _resolve_box(name):
    """Map 'blue' / 'box_blue' / 'blue box' to the body name 'box_blue'."""
    key = name.strip().lower().replace("box_", "").replace("box", "").strip(" _")
    if key in BOX_COLORS:
        return f"box_{key}"
    return None


def _capture_cameras_b64(cameras, width=320, height=240):
    # A fresh renderer per call: GL contexts are thread-bound and LangGraph may
    # run each tool call on a different thread, so a cached renderer breaks.
    # Keep images small — large PNGs make NVIDIA vision calls very slow.
    renderer = mujoco.Renderer(ENV.model, height, width)
    try:
        images = []
        for camera in cameras:
            renderer.update_scene(ENV.data, camera=camera)
            img = PIL.Image.fromarray(renderer.render())
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            images.append(base64.b64encode(buf.getvalue()).decode())
        return images
    finally:
        renderer.close()


@tool
def look_at_scene() -> str:
    """Look at the workspace through the external camera and the wrist camera,
    and return a description of the colored boxes that are visible."""
    # Gemma 4 performs best when images precede the text prompt.
    content = []
    for b64 in _capture_cameras_b64(CAMERAS):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                "These are two camera views of a robot workspace: first an "
                "external 3/4 view, second a wrist-mounted top-down view. "
                "List the small colored boxes you can see on the table or "
                "shelf (colors and rough locations, e.g. 'blue box near the "
                "front edge', 'red box on the shelf'). Be brief."
            ),
        }
    )
    timeout = float(os.environ.get("NIM_VISION_TIMEOUT", "90"))
    try:
        data = _nim_chat(
            [{"role": "user", "content": content}],
            tools=None,
            max_tokens=128,
            timeout=timeout,
        )
        return _as_text(data["choices"][0]["message"].get("content"))
    except Exception as e:
        return (
            f"Vision request failed ({e}). Box colors are red, green, blue, "
            "and yellow — use find_object or pick directly."
        )


@tool
def find_object(name: str) -> str:
    """Return the world position of an object or location (a box color like "blue", or "shelf")."""
    body = _resolve_box(name) or name
    pos, _ = skills.find_object(ENV, body)
    return f"{name} is at x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}"


@tool
def pick(object_name: str) -> str:
    """Pick up a colored box with the gripper. Valid objects: "red", "green", "blue", "yellow"."""
    body = _resolve_box(object_name)
    if body is None:
        return (
            f"Error: unknown object {object_name!r}. "
            f"Valid objects: {', '.join(BOX_COLORS)}."
        )
    try:
        skills.pick(ENV, body)
    except RuntimeError as e:
        return f"Could not pick up the {object_name} box: {e}"
    pos, _ = skills.find_object(ENV, body)
    return f"Picked up the {object_name} box; it is now held at z={pos[2]:.3f}."


@tool
def place(location_name: str) -> str:
    """Place the currently held box onto a free spot on the named location.
    Valid locations: "shelf". Reports an error if the location is full."""
    if location_name.strip().lower() != "shelf":
        return f"Error: unknown location {location_name!r}. Valid locations: shelf."
    try:
        skills.place(ENV, "shelf")
    except RuntimeError as e:
        return f"Could not place: {e} The held box is still in the gripper."
    return "Placed the held box on a free spot on the shelf and released it."


def _nim_chat(messages, tools=None, max_tokens=256, timeout=120):
    """Call NVIDIA NIM chat/completions directly (avoids LangGraph 504 timeouts)."""
    payload = {
        "model": NIM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(os.environ.get("NIM_TEMPERATURE", "0.2")),
        "top_p": float(os.environ.get("NIM_TOP_P", "0.95")),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = requests.post(
        NIM_URL,
        headers={
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Accept": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"NIM HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def _execute_tool(name, args):
    tools = {
        "look_at_scene": lambda a: look_at_scene.invoke(a or {}),
        "find_object": lambda a: find_object.invoke(a),
        "pick": lambda a: pick.invoke(a),
        "place": lambda a: place.invoke(a),
    }
    if name not in tools:
        return f"Error: unknown tool {name!r}"
    return tools[name](args or {})


_COLOR = r"(red|green|blue|yellow)"


def _local_plan(text):
    """Fast path for common commands — no API call needed."""
    s = text.lower().strip()
    steps = []

    if re.search(r"what do you see|look at (?:the )?scene|describe (?:the )?scene", s):
        return [("look_at_scene", {})]

    for m in re.finditer(
        rf"put (?:the )?{_COLOR}(?: box)? on (?:the )?shelf", s
    ):
        steps.extend(
            [
                ("pick", {"object_name": m.group(1)}),
                ("place", {"location_name": "shelf"}),
            ]
        )
    if steps:
        return steps

    for m in re.finditer(rf"pick(?: up)? (?:the )?{_COLOR}(?: box)?", s):
        steps.append(("pick", {"object_name": m.group(1)}))
    if steps:
        return steps

    if re.search(r"place (?:it )?on (?:the )?shelf|put (?:it )?on (?:the )?shelf", s):
        return [("place", {"location_name": "shelf"})]

    return None


def build_agent():
    global NIM_API_KEY
    NIM_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVAPI_API_KEY")
    if not NIM_API_KEY:
        sys.exit("Set NVIDIA_API_KEY before running agent.py")
    return object()  # placeholder; run_instruction is self-contained


def _as_text(content):
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content).strip()


def _default_emit(line):
    print(line, flush=True)


def _run_steps(steps, emit):
    last_result = None
    for name, args in steps:
        emit(f"  [tool call] {name}({args})")
        last_result = _execute_tool(name, args)
        emit(f"  [tool result] {last_result}")
    return last_result


def run_instruction(agent, text, attempts=3, backoff=10.0, emit=_default_emit):
    plan = _local_plan(text)
    if plan:
        emit("agent: understood (local)")
        result = _run_steps(plan, emit)
        emit(f"agent: {result}")
        return

    emit("agent: asking NVIDIA NIM... (may take 30-60s)")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    last_tool_result = None
    for i in range(attempts):
        try:
            for _ in range(6):
                data = _nim_chat(messages, tools=TOOL_SCHEMAS, max_tokens=256)
                msg = data["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    reply = _as_text(msg.get("content"))
                    if not reply and last_tool_result:
                        reply = last_tool_result
                    emit(f"agent: {reply}")
                    return
                messages.append(msg)
                for tc in tool_calls:
                    fn = tc["function"]
                    args = json.loads(fn.get("arguments") or "{}")
                    emit(f"  [tool call] {fn['name']}({args})")
                    last_tool_result = _execute_tool(fn["name"], args)
                    emit(f"  [tool result] {last_tool_result}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": last_tool_result,
                        }
                    )
            emit(f"agent: {last_tool_result or 'gave up after too many tool rounds'}")
            return
        except Exception as e:
            if i == attempts - 1:
                emit(f"agent: giving up after {attempts} attempts ({e})")
                return
            emit(f"  [retry {i + 1}/{attempts - 1}] {e}")
            time.sleep(backoff * (i + 1))


def main():
    global ENV
    ENV = SimEnv()
    ENV.step_settle(200)
    ENV.randomize_boxes()
    agent = build_agent()
    print("Robot agent ready. Type an instruction (Ctrl+C / empty line to quit).")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        run_instruction(agent, text)
    ENV.close()


if __name__ == "__main__":
    main()
