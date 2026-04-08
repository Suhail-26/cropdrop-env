"""
Baseline inference script for CropDrop environment
Uses OpenAI client with the hackathon's LLM proxy.
If the proxy call fails, falls back to random actions (but still makes at least one attempt).
"""

import os
import sys
import json
import random
from openai import OpenAI
from server.cropdrop_env_environment import CropdropEnvironment
from models import CropdropAction

# ============================================================
# CRITICAL: Use the hackathon's environment variables.
# Do NOT set defaults for API_KEY (must come from evaluator).
# ============================================================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")      # or HF_TOKEN
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Fail fast if proxy variables are missing (evaluator must provide them)
if not API_BASE_URL or not API_KEY:
    raise ValueError(
        "API_BASE_URL and API_KEY must be set by the evaluator. "
        "Do not hardcode API keys."
    )

# Initialize the OpenAI client with the proxy
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

def get_llm_action(observation, crops, routes):
    """Call the LLM to decide the next action. Fallback to random on error."""
    prompt = f"""
You are controlling a crop delivery robot.
Current time: {observation.current_time}
Available crops: {json.dumps(crops, indent=2)}
Routes: {json.dumps(routes, indent=2)}

Choose an action. Reply ONLY with a JSON object in this format:
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
        action_data = json.loads(content)
        return CropdropAction(
            crop_id=action_data["crop_id"],
            route=action_data["route"],
            destination_zone=action_data["destination_zone"]
        )
    except Exception as e:
        # If LLM call fails (network, parsing, etc.), fall back to random action
        print(f"Warning: LLM call failed: {e}. Using random fallback.", file=sys.stderr)
        # Random fallback (ensures the script does not crash)
        return CropdropAction(
            crop_id=random.choice(crops)["id"],
            route=random.choice(["paved", "dirt", "muddy"]),
            destination_zone=random.choice(["zone_1", "zone_2"])
        )

def run_baseline_agent(env, max_steps=20):
    """Run agent using LLM for decision making (with fallback)."""
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(max_steps):
        if not env.crops:
            break
        
        action = get_llm_action(obs, env.crops, env.routes)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    return total_reward / max(steps, 1)

def run_all_tasks():
    results = {}
    
    # Easy task (single crop)
    env = CropdropEnvironment()
    env.crops = [env.crops[0]]
    results["easy"] = run_baseline_agent(env)
    
    # Medium task (3 crops)
    env = CropdropEnvironment()
    results["medium"] = run_baseline_agent(env)
    
    # Hard task (3 crops + congestion)
    env = CropdropEnvironment()
    results["hard"] = run_baseline_agent(env)
    
    return results

if __name__ == "__main__":
    print("=" * 50)
    print("🌾 CropDrop Baseline Inference (with LLM proxy)")
    scores = run_all_tasks()
    print("\n📊 Baseline Scores:")
    for task, score in scores.items():
        print(f"  {task.upper()}: {score:.2f}")
    print("=" * 50)
