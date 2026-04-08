"""
Baseline inference script for CropDrop Environment
Uses LLM via OpenAI Client to decide actions.
Calls the deployed HF Space via HTTP.
Emits structured [START], [STEP], [END] logs for evaluation.
"""

import os
import sys
import json
import requests
from openai import OpenAI

# ─── Configuration from environment variables ───
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "https://suhailma-cropdrop-env.hf.space")

# ─── OpenAI-compatible client ───
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── Task definitions ───
TASKS = [
    {
        "name": "single_priority_delivery",
        "difficulty": "easy",
        "description": "Deliver a single crop to the correct zone before it spoils.",
        "max_steps": 5,
    },
    {
        "name": "multi_crop_prioritization",
        "difficulty": "medium",
        "description": "Deliver all 3 crops to their correct zones, prioritizing fast-spoiling ones.",
        "max_steps": 10,
    },
    {
        "name": "route_optimization",
        "difficulty": "hard",
        "description": "Deliver all crops using optimal routes, minimizing time and avoiding muddy paths.",
        "max_steps": 15,
    },
]


def call_env(endpoint, method="POST", payload=None):
    """Call the HF Space environment API."""
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
        print(f"[ERROR] API call to {endpoint} failed: {e}", file=sys.stderr)
        return None


def ask_llm(observation, task_info):
    """Ask the LLM to choose the next action given the observation."""
    crops_info = ""
    for crop in observation.get("crops", []):
        crops_info += (
            f"  - Crop ID {crop['id']}: {crop['type']}, "
            f"spoilage_remaining={crop['spoilage_remaining']}, "
            f"intended_zone={crop['intended_zone']}\n"
        )

    routes_info = ""
    for route_name, info in observation.get("routes_status", {}).items():
        routes_info += (
            f"  - {route_name}: estimated_time={info.get('estimated_time', '?')}, "
            f"congestion={info.get('congestion', '?')}\n"
        )

    prompt = f"""You are an AI agent managing agricultural crop deliveries.
Task: {task_info['description']}
Difficulty: {task_info['difficulty']}
Current State:
- Time elapsed: {observation.get('current_time', 0)}
- Last action result: {observation.get('last_action_result', 'None')}
Available Crops to deliver:
{crops_info if crops_info else '  (none remaining)'}
Route Options:
{routes_info if routes_info else '  (unknown)'}
Choose the best action. You must respond with ONLY a JSON object:
{{"crop_id": <int>, "route": "<paved|dirt|muddy>", "destination_zone": "<zone_1|zone_2>"}}
Strategy tips:
- Match each crop to its intended_zone for maximum reward
- Use "paved" route for fast-spoiling crops (lowest travel time)
- Prioritize crops with lowest spoilage_remaining first
- Avoid "muddy" routes unless necessary (high time cost)
Respond with ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON from response - handle markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        action = json.loads(content)
        return action
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
        # Fallback: pick fastest-spoiling crop, paved route, correct zone
        crops = observation.get("crops", [])
        if crops:
            crop = min(crops, key=lambda c: c.get("spoilage_remaining", 999))
            return {
                "crop_id": crop["id"],
                "route": "paved",
                "destination_zone": crop.get("intended_zone", "zone_1"),
            }
        return {"crop_id": 1, "route": "paved", "destination_zone": "zone_1"}


def run_task(task_info):
    """Run a single task episode and return the score."""
    task_name = task_info["name"]
    max_steps = task_info["max_steps"]

    # ─── [START] log ───
    start_log = {
        "task": task_name,
        "difficulty": task_info["difficulty"],
        "max_steps": max_steps,
        "env_url": ENV_URL,
        "model": MODEL_NAME,
    }
    print(f"[START] {json.dumps(start_log)}")

    # Reset the environment
    obs = call_env("reset", method="POST")
    if obs is None:
        print(f"[END] {json.dumps({'task': task_name, 'score': 0.0, 'total_reward': 0.0, 'steps': 0, 'error': 'reset failed'})}")
        return 0.0

    total_reward = 0.0
    steps = 0

    for step_num in range(1, max_steps + 1):
        crops = obs.get("crops", [])
        if not crops:
            break

        # For easy task, only use first crop
        if task_info["difficulty"] == "easy" and len(crops) > 1:
            obs["crops"] = [crops[0]]

        # Ask LLM for action
        action = ask_llm(obs, task_info)

        # Execute action via API
        result = call_env("step", method="POST", payload=action)
        if result is None:
            step_log = {
                "step": step_num,
                "action": action,
                "reward": 0.0,
                "done": False,
                "error": "step call failed",
            }
            print(f"[STEP] {json.dumps(step_log)}")
            break

        # Parse response - OpenEnv returns observation, reward, done, info
        if isinstance(result, dict):
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", result)
            info = result.get("info", {})
        else:
            reward = 0.0
            done = False
            obs = {}
            info = {}

        total_reward += reward
        steps = step_num

        # ─── [STEP] log ───
        step_log = {
            "step": step_num,
            "action": action,
            "reward": round(reward, 4),
            "total_reward": round(total_reward, 4),
            "done": done,
            "last_action_result": obs.get("last_action_result", None),
        }
        print(f"[STEP] {json.dumps(step_log)}")

        if done:
            break

    # Get grader score
    grader_result = call_env(f"grader/{task_name}", method="POST")
    score = 0.0
    if grader_result and isinstance(grader_result, dict):
        score = grader_result.get("score", 0.0)

    # ─── [END] log ───
    end_log = {
        "task": task_name,
        "difficulty": task_info["difficulty"],
        "score": round(score, 4),
        "total_reward": round(total_reward, 4),
        "steps": steps,
    }
    print(f"[END] {json.dumps(end_log)}")

    return score


def main():
    """Run baseline inference on all 3 tasks."""
    print("=" * 60)
    print("CropDrop Environment - Baseline Inference")
    print(f"Environment: {ENV_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print("=" * 60)

    all_scores = {}

    for task in TASKS:
        print(f"\n--- Running Task: {task['name']} ({task['difficulty']}) ---")
        score = run_task(task)
        all_scores[task["name"]] = score
        print(f"--- Score: {score:.4f} ---\n")

    # Summary
    print("=" * 60)
    print("FINAL SCORES:")
    for task_name, score in all_scores.items():
        print(f"  {task_name}: {score:.4f}")
    avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0
    print(f"  AVERAGE: {avg_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
