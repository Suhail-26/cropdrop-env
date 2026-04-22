import os
import sys
import json
import random
from openai import OpenAI
from live_client import CropdropLiveClient

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "set MODEL_NAME=microsoft/phi-2")

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable must be set.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_llm_action(observation, crops, routes):
    prompt = f"""
You are controlling a crop delivery robot.
Current time: {observation.get('current_time', 0)}
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
        content = response.choices[0].message.content
        # Clean markdown if present
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        action_data = json.loads(content)
        # Ensure integer type
        action_data['crop_id'] = int(action_data['crop_id'])
        return action_data
    except Exception as e:
        print(f"Warning: LLM call failed ({e}). Using random fallback.", file=sys.stderr)
        if not crops:
            return {"crop_id": 1, "route": "paved", "destination_zone": "zone_1"}
        return {
            "crop_id": random.choice(crops)["id"],
            "route": random.choice(["paved", "dirt", "muddy"]),
            "destination_zone": random.choice(["zone_1", "zone_2"])
        }

def run_task(task_name, env_client, max_steps=20):
    print(f"[START] task={task_name}", flush=True)
    obs = env_client.reset()
    total_reward = 0.0
    steps = 0

    for step in range(1, max_steps + 1):
        crops = obs.get('crops', [])
        if not crops:
            break
        routes = obs.get('routes_status', {})
        action_dict = get_llm_action(obs, crops, routes)
        print(f"[DEBUG] Sending action: {action_dict}", flush=True)
        try:
            obs, reward, done, _ = env_client.step(action_dict)
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", flush=True)
            break
        total_reward += reward
        print(f"[STEP] step={step} reward={reward:.3f}", flush=True)
        steps = step
        if done:
            break

    score = max(0.001, min(0.999, total_reward / max(steps, 1)))
    print(f"[END] task={task_name} score={score:.3f} steps={steps}", flush=True)
    return score

def run_all_tasks():
    with CropdropLiveClient() as client:
        run_task("easy", client)
        run_task("medium", client)
        run_task("hard", client)

if __name__ == "__main__":
    run_all_tasks()