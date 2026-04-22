"""
CropOps Agent - Round 2 Inference Script
Upgraded from Round 1 CropDrop inference.py

Key changes:
  - Action is now: up/down/left/right/pickup/dropoff (not crop_id + route + zone)
  - LLM prompt now includes agent_position, grid map, and disruptions
  - Fallback logic uses BFS pathfinding (not just "pick paved route")
"""

import os
import sys
import json
import requests
from collections import deque
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       "https://suhailma-cropops-env.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = [
    {
        "name": "single_priority_delivery",
        "difficulty": "easy",
        "description": "Pick up 1 crop and deliver it to the correct warehouse zone before it spoils.",
        "max_steps": 20,
    },
    {
        "name": "multi_crop_prioritization",
        "difficulty": "medium",
        "description": "Deliver all 3 crops to correct zones. Prioritize the fast-spoiling tomato first.",
        "max_steps": 40,
    },
    {
        "name": "route_optimization",
        "difficulty": "hard",
        "description": "Deliver all crops efficiently despite road blocks, rain, and vehicle breakdowns.",
        "max_steps": 50,
    },
]

# Fixed positions (match environment)
WAREHOUSE_POSITIONS = {"zone_1": [4, 2], "zone_2": [4, 7]}
VILLAGE_POSITIONS   = {1: [1, 1], 2: [1, 8], 3: [8, 1], 4: [8, 8]}
GRID_SIZE = 10
ALL_ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff"]


# ── API helpers ───────────────────────────────────────────────
def call_env(endpoint, method="POST", payload=None):
    url = f"{ENV_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        if method == "POST":
            resp = requests.post(url, json=payload or {}, headers=headers, timeout=30)
        else:
            resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] {endpoint}: {e}", file=sys.stderr)
        return None


# ── BFS pathfinding fallback ──────────────────────────────────
def bfs_next_action(agent_pos, target_pos, blocked_tiles=None):
    """
    Returns the first action to take toward target_pos using BFS.
    blocked_tiles: list of [r,c] that are currently blocked.
    """
    blocked = set(tuple(t) for t in (blocked_tiles or []))
    start   = tuple(agent_pos)
    goal    = tuple(target_pos)

    if start == goal:
        return None  # already there

    queue = deque([(start, [])])
    visited = {start}
    move_map = {
        "up":    (-1, 0),
        "down":  (1,  0),
        "left":  (0, -1),
        "right": (0,  1),
    }

    while queue:
        pos, path = queue.popleft()
        for action, (dr, dc) in move_map.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            npos = (nr, nc)
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if npos in blocked or npos in visited:
                continue
            new_path = path + [action]
            if npos == goal:
                return new_path[0]   # first step toward goal
            visited.add(npos)
            queue.append((npos, new_path))

    return "right"   # last resort: just move


def fallback_action(obs):
    """Rule-based fallback when LLM fails."""
    agent = obs.get("agent_position", [5, 5])
    carried = obs.get("carried_crop")
    crops = obs.get("crops", [])
    blocked = []
    for d in obs.get("active_disruptions", []):
        if d.get("blocked_tile"):
            blocked.append(d["blocked_tile"])

    # If carrying → go to correct warehouse
    if carried:
        target_zone = carried["intended_zone"]
        target = WAREHOUSE_POSITIONS[target_zone]
        if agent == target:
            return "dropoff"
        return bfs_next_action(agent, target, blocked)

    # Pick the crop with least spoilage remaining
    pending = [c for c in crops if not c["picked_up"] and not c["delivered"]]
    if not pending:
        return "dropoff"   # all done, shouldn't happen

    urgent = min(pending, key=lambda c: c["spoilage_remaining"])
    target = urgent["position"]
    if agent == target:
        return "pickup"
    return bfs_next_action(agent, target, blocked)


# ── LLM decision ─────────────────────────────────────────────
def ask_llm(obs, task_info):
    agent    = obs.get("agent_position", [5, 5])
    carried  = obs.get("carried_crop")
    crops    = obs.get("crops", [])
    pending  = [c for c in crops if not c["delivered"]]
    warehouses = obs.get("warehouses", WAREHOUSE_POSITIONS)
    disruptions = obs.get("active_disruptions", [])

    crops_text = ""
    for c in pending:
        status = "CARRYING" if (carried and carried["id"] == c["id"]) else (
            "picked_up" if c["picked_up"] else "at village"
        )
        crops_text += (
            f"  Crop {c['id']} ({c['type']}): pos={c['position']}, "
            f"spoilage_left={c['spoilage_remaining']}, "
            f"deliver_to={c['intended_zone']}, status={status}\n"
        )

    wh_text = "\n".join(f"  {z}: {p}" for z, p in warehouses.items())

    dis_text = ""
    for d in disruptions:
        dis_text += f"  ⚠ {d['type']}: {d['message']}\n"
    if obs.get("breakdown_steps_remaining", 0) > 0:
        dis_text += f"  ⚠ BREAKDOWN: {obs['breakdown_steps_remaining']} steps immobilised\n"
    if not dis_text:
        dis_text = "  None\n"

    prompt = f"""You are a delivery agent on a 10x10 grid.
Task: {task_info['description']}

Your position: {agent}
Carrying: {carried['type'] + ' (crop ' + str(carried['id']) + ')' if carried else 'Nothing'}
Step: {obs.get('current_time', 0)} / {task_info['max_steps']}

Crops to deliver:
{crops_text if crops_text else '  All delivered!\n'}
Warehouses (drop-off points):
{wh_text}

Active disruptions:
{dis_text}

Rules:
- Move with: up / down / left / right
- When ON a village tile with an uncollected crop → use: pickup
- When ON a warehouse tile and carrying the correct crop → use: dropoff
- Avoid blocked tiles listed in disruptions
- Prioritize crops with lowest spoilage_remaining

Respond ONLY with one JSON: {{"action": "<up|down|left|right|pickup|dropoff>"}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        action_obj = json.loads(content.strip())
        action = action_obj.get("action", "").lower()
        if action in ALL_ACTIONS:
            return action
    except Exception as e:
        print(f"[WARN] LLM failed: {e}", file=sys.stderr)

    return fallback_action(obs)


# ── Task runner ───────────────────────────────────────────────
def run_task(task_info):
    name      = task_info["name"]
    max_steps = task_info["max_steps"]

    start_log = {
        "task": name, "difficulty": task_info["difficulty"],
        "max_steps": max_steps, "model": MODEL_NAME,
    }
    print(f"[START] {json.dumps(start_log)}")

    obs = call_env("reset", method="POST")
    if obs is None:
        print(f"[END] {json.dumps({'task': name, 'score': 0.0, 'error': 'reset failed'})}")
        return 0.0

    total_reward = 0.0
    steps = 0

    for step_num in range(1, max_steps + 1):
        # Stop if nothing left to deliver
        pending = obs.get("pending_orders", [])
        if not pending and not obs.get("carried_crop"):
            break

        action = ask_llm(obs, task_info)
        result = call_env("step", method="POST", payload={"action": action})

        if result is None:
            print(f"[STEP] {json.dumps({'step': step_num, 'action': action, 'error': 'step failed'})}")
            break

        reward = result.get("reward", 0.0)
        done   = result.get("done", False)
        obs    = result.get("observation", result)
        total_reward += reward
        steps = step_num

        print(f"[STEP] {json.dumps({'step': step_num, 'action': action, 'reward': round(reward, 4), 'total_reward': round(total_reward, 4), 'done': done, 'result': obs.get('last_action_result', '')})}")

        if done:
            break

    grader_result = call_env(f"grader/{name}", method="POST")
    score = grader_result.get("score", 0.0) if grader_result else 0.0

    print(f"[END] {json.dumps({'task': name, 'score': round(score, 4), 'total_reward': round(total_reward, 4), 'steps': steps})}")
    return score


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CropOps Agent Round 2 — Inference")
    print(f"Env: {ENV_URL}  |  Model: {MODEL_NAME}")
    print("=" * 60)

    all_scores = {}
    for task in TASKS:
        print(f"\n--- Task: {task['name']} ({task['difficulty']}) ---")
        score = run_task(task)
        all_scores[task["name"]] = score

    print("\n" + "=" * 60)
    print("FINAL SCORES:")
    for name, score in all_scores.items():
        print(f"  {name}: {score:.4f}")
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  AVERAGE: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()