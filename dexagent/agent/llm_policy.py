"""A bench policy driven by an OpenRouter model: observe, ask, execute, repeat.

`dexagent.bench` loads this file as a policy (`run(api)`); the run is configured by the JSON file
named in DEXAGENT_LLM_CONFIG (written by `dexagent.agent.run_agent`):

    model, objective, camera, max_calls, max_tools_per_call, keep_images, temperature, max_tokens,
    log_dir, stub (path to a scripted-turns JSON; replaces the model)

Each turn the model sees a camera image, proprioception and contact forces (no object pose), and
answers with tool calls for the level's own commands. One OpenRouter request is one agent call.
The simulator advances only while a tool call executes. Counters are kept on `api._llm`, which
the bench writes into the case record.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from dexagent.agent import observe as O
from dexagent.agent import tools as T

FRAMES = """\
Frames. World: z up, table top at z = 0.75 m. Palm frame: +z runs along the fingers, +x is out of the palm face
(fingers flex toward +x), +y is toward the thumb. Fingers are indexed 0 thumb, 1 index, 2 middle, 3 ring, 4 little.
Arm: a UR5e carrying the hand; you command the PALM pose in the world frame and the arm follows by IK
(a target out of reach is rejected with an error)."""

LEVEL_NOTES = {
    "joints": "You command the 20 finger joint targets directly (no finger coordination is provided).",
    "fingertips": "You command finger pad-centre positions in the palm frame; per-finger IK turns them into joint targets.",
    "synergies": "You command whole-hand posture through k synergy coefficients; individual fingers cannot be commanded.",
    "contacts": ("You command object-centric contact primitives. Contact points are given in the OBJECT frame (origin = the object's "
                 "pose position, axes aligned with the object); the controller maps them through the object's true pose. You are NOT "
                 "told the object's pose: infer it from the image."),
    "layered": "All layers are available at once: joints, fingertips, synergies and contact primitives. Use whichever suits each step.",
    "force": "All layers are available, plus a per-finger grip-force servo (grip/ungrip) that keeps running through later motions.",
}

RULES = """\
How to work:
- You are given one or more camera images each turn (each labelled with its camera name), plus proprioception and contact
  forces. You are NOT given any object pose, in-palm pose, velocity or object joint angle — even if the task text says a
  pose is "readable". Infer where objects are from the images (each camera's eye, look-at and field of view are listed in
  the observation) and from contact forces. Use multiple views together when they are provided.
- Every reply MUST include tool calls (plain text alone is rejected). You may send several tool calls in one reply; they run
  in order, and a failing call cancels the rest. Nested arguments (arrays, objects) must be real JSON values, not strings
  that contain JSON. Each reply costs one agent call and you have a fixed budget.
- Motions take simulated time (`seconds`); the world only advances while a command runs.
- If a call returns an error, read it and change the command; do not repeat it unchanged.
- Call `finish` when the objective is achieved (or cannot be)."""


def load_config():
    path = os.environ.get("DEXAGENT_LLM_CONFIG")
    if not path:
        raise RuntimeError("DEXAGENT_LLM_CONFIG is not set; run through dexagent.agent.run_agent")
    return json.loads(Path(path).read_text())


def system_prompt(api, cfg):
    level = api.level
    parts = [f"You control a UR5e arm with a 20-DOF Tesollo DG5F-S right hand in a physics simulation, at the hand-command "
             f"representation '{level}'.", FRAMES, f"Representation: {LEVEL_NOTES[level]}"]
    if hasattr(api, "joint_limits"):
        lim = np.round(api.joint_limits, 2)
        parts.append("Finger joint limits [lo, hi] rad, entries 4i..4i+3 for finger i: " + json.dumps(lim.tolist()))
        parts.append("Open-hand joint posture: " + json.dumps(np.round(api.OPEN, 2).tolist()))
    if hasattr(api, "describe"):
        parts.append(f"Synergies: k = {api.k}. Coefficients are in standard deviations; an open hand is s = "
                     f"{json.dumps(np.round(api.open_coeffs, 2).tolist())}.\n{api.describe()}")
    parts.append(f"Scored objects: {list(api.objects)}.")
    parts.append(RULES)
    parts.append(f"Budget: at most {cfg['max_calls']} agent calls.")
    return "\n\n".join(parts)


def _prune_images(messages, keep):
    """Drop images from all but the last `keep` observation turns (each turn may contain several cameras)."""
    idx = [i for i, m in enumerate(messages) if isinstance(m.get("content"), list)
           and any(p.get("type") == "image_url" for p in m["content"])]
    drop = set(idx[:-keep]) if keep > 0 else set(idx)
    out = []
    for i, m in enumerate(messages):
        if i in drop:
            m = dict(m, content=[{"type": "text", "text": "[earlier image omitted]"} if p.get("type") == "image_url" else p
                                 for p in m["content"]])
        out.append(m)
    return out


def _user_obs(text, images):
    """`images` is a list of (camera_name, data_url)."""
    content = [{"type": "text", "text": text}]
    for name, url in images:
        content.append({"type": "text", "text": f"[camera: {name}]"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


def make_client(cfg, log_path):
    if cfg.get("stub"):
        from dexagent.agent.stub import ScriptedClient
        return ScriptedClient(json.loads(Path(cfg["stub"]).read_text()))
    from dexagent.agent.openrouter import OpenRouter
    return OpenRouter(cfg["model"], temperature=cfg.get("temperature", 0.2),
                      max_tokens=int(cfg.get("max_tokens", 8192)), log_path=log_path)


def run(api):
    cfg = load_config()
    level = api.level
    case = os.environ.get("DEXAGENT_CASE", "0:0").replace(":", "_")
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(cfg, str(log_dir / f"llm_{case}.jsonl"))
    tools = T.tool_specs(level)
    cameras = O.parse_cameras(cfg.get("cameras") or cfg.get("camera") or "closeup,overview")
    max_calls, max_tools = int(cfg["max_calls"]), int(cfg.get("max_tools_per_call", 6))
    keep_images = int(cfg.get("keep_images", 2))  # observation turns kept with images

    c = api._llm = {"level": level, "model": getattr(client, "model", "?"), "agent_calls": 0, "tool_calls": 0, "tool_errors": 0,
                    "rejected_calls": 0, "ik_failures": 0, "param_dims": 0, "tokens_in": 0, "tokens_out": 0,
                    "llm_wall_s": 0.0, "tool_counts": {}, "stopped": "running", "no_tool_replies": 0, "coerced_args": 0,
                    "cameras": list(cameras)}
    transcript = []

    def dump():
        (log_dir / f"transcript_{case}.json").write_text(json.dumps(transcript, indent=1, default=str))

    obs, images = O.observe(api, cameras)
    messages = [{"role": "system", "content": system_prompt(api, cfg)},
                _user_obs(f"OBJECTIVE:\n{cfg['objective']}\n\nInitial observation (agent call 1 of {max_calls}):\n"
                          + json.dumps(obs), images)]
    transcript.append({"system": messages[0]["content"], "objective": cfg["objective"], "cameras": list(cameras)})

    try:
        for _ in range(max_calls):
            t0 = time.time()
            reply = client.chat(_prune_images(messages, keep_images), tools)
            c["agent_calls"] += 1
            c["llm_wall_s"] = round(c["llm_wall_s"] + time.time() - t0, 2)
            u = reply.get("usage") or {}
            c["tokens_in"] += int(u.get("prompt_tokens", 0))
            c["tokens_out"] += int(u.get("completion_tokens", 0))
            msg = reply["message"]
            calls = msg.get("tool_calls") or []
            entry = {"call": c["agent_calls"], "t_sim": round(api.t, 2), "text": msg.get("content"), "tool_calls": []}
            transcript.append(entry)

            if not calls:
                c["no_tool_replies"] += 1
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                if c["no_tool_replies"] >= 3:
                    c["stopped"] = "no tool calls"
                    break
                messages.append({"role": "user", "content": "Reply with tool calls only. Call `finish` if you are done."})
                continue
            c["no_tool_replies"] = 0

            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            results, abort, done = [], False, False
            for i, tc in enumerate(calls):
                name = tc["function"]["name"]
                if i >= max_tools:
                    res = {"error": f"skipped: at most {max_tools} tool calls per reply"}
                elif abort:
                    res = {"error": "skipped: an earlier call in this reply failed"}
                else:
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                        args, coerced = T.coerce_args(args)
                    except json.JSONDecodeError as e:
                        args, res = None, {"error": f"arguments are not valid JSON: {e}"}
                    else:
                        if coerced:
                            c["coerced_args"] = c.get("coerced_args", 0) + coerced
                        err = T.validate(level, name, args)
                        if err:
                            c["rejected_calls"] += 1
                            res = {"error": err}
                        elif name == "finish":
                            done = True
                            res = {"ok": True, "note": "finishing"}
                        else:
                            c["tool_calls"] += 1
                            c["tool_counts"][name] = c["tool_counts"].get(name, 0) + 1
                            c["param_dims"] += T.count_dims(args)
                            try:
                                res = yield from T.execute(api, level, name, args)
                            except ValueError as e:  # arm IK out of reach, or a bad value from the api
                                c["ik_failures"] += 1
                                res = {"error": str(e)[:300]}
                            except Exception as e:  # noqa: BLE001 - the model must hear about any failed command
                                c["tool_errors"] += 1
                                res = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                    if isinstance(res, dict) and "error" in res and not done:
                        abort = True
                results.append(res)
                entry["tool_calls"].append({"name": name, "arguments": tc["function"].get("arguments"), "result": res})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
            dump()
            if done:
                c["stopped"] = "finish"
                break
            obs, images = O.observe(api, cameras)
            messages.append(_user_obs(f"Observation after your commands (agent call {c['agent_calls']} of {max_calls} used):\n"
                                      + json.dumps(obs), images))
        else:
            c["stopped"] = "call budget"
    finally:
        if c["stopped"] == "running":
            c["stopped"] = "error"
        dump()
    c["sim_s"] = round(api.t, 2)
