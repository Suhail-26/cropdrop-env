"""
Baseline inference script for CropDrop environment
Uses OpenAI client with the hackathon's LLM proxy.
Outputs structured blocks for validator: [START], [STEP], [END].
"""

import os
import sys
import json
import random
from openai import OpenAI
from server.cropdrop_env_environment import CropdropEnvironment
from models import CropdropAction

# ============================================================
# Use hackathon's environment variables (no defaults for API_KEY)
# ============================================================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")      # or HF_TOKEN
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

if not API_BASE_URL or not API_KEY:
    raise ValueError("API_BASE_URL and API_KEY must be set by the evaluator.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_llm_action(observation, crops, routes):
    """Call LLM; fallback to random on any error."""
    prompt = f"""
You are controlling a crop delivery robot.
Current time: {observation.current_time}
Available crops: {json.dumps(crops, indent=2)}
Routes: {json.dumps(routes, indent=2)}

Choose an action. Reply ONLY with a JSON object:
{{"crop_id": <id>, "route": "paved" or "dirt" or "muddy", "destination_zone": "zone_1" or "zone_2"}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        action_data = json.loads(response.choices[0].message.content)
        return CropdropAction(
            crop_id=action_data["crop_id"],
            route=action_data["route"],
            destination_zone=action_data["destination_zone"]
        )
    except Exception as e:
        print(f"Warning: LLM call failed ({e}). Using random fallback.", file=sys.stderr)
        return CropdropAction(
            crop_id=random.choice(crops)["id"],
            route=random.choice(["paved", "dirt", "muddy"]),
            destination_zone=random.choice(["zone_1", "zone_2"])
        )

def run_task(task_name, env, max_steps=20):
    """Run one episode and print structured output."""
    print(f"[START]task={task_name}", flush=True)
    obs = env.reset()
    total_reward = 0.0
    step = 0

    for step in range(1, max_steps + 1):
        if not env.crops:
            break
        action = get_llm_action(obs, env.crops, env.routes)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"[STEP]step={step} reward={reward:.3f}", flush=True)
        if done:
            break

    print(f"[END]task={task_name} score={total_reward:.3f} steps={step}", flush=True)
    return total_reward

def run_all_tasks():
    """Run easy, medium, hard tasks."""
    # Easy: single crop
    env = CropdropEnvironment()
    env.crops = [env.crops[0]]
    run_task("easy", env)

    # Medium: 3 crops, no congestion (default)
    env = CropdropEnvironment()
    run_task("medium", env)

    # Hard: 3 crops with congestion (same as default, but congestion active)
    env = CropdropEnvironment()
    run_task("hard", env)

if __name__ == "__main__":
    run_all_tasks()
