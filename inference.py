"""
Baseline inference script for CropDrop environment
Uses OpenAI client with the hackathon's LLM proxy
"""

import os
import json
from openai import OpenAI
from server.cropdrop_env_environment import CropdropEnvironment
from models import CropdropAction

# ============================================================
# CRITICAL: Use the hackathon's environment variables
# Do NOT set defaults for API_KEY (must come from evaluator)
# ============================================================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")      # or HF_TOKEN
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

if not API_BASE_URL or not API_KEY:
    raise ValueError(
        "API_BASE_URL and API_KEY must be set by the evaluator. "
        "Do not hardcode API keys."
    )

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

def get_llm_action(observation, crops, routes):
    """Call the LLM to decide the next action"""
    prompt = f"""
You are controlling a crop delivery robot.
Current time: {observation.current_time}
Available crops: {json.dumps(crops, indent=2)}
Routes: {json.dumps(routes, indent=2)}

Choose an action. Reply ONLY with a JSON object in this format:
{{"crop_id": <id>, "route": "paved" or "dirt" or "muddy", "destination_zone": "zone_1" or "zone_2"}}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    
    try:
        action_data = json.loads(response.choices[0].message.content)
        return CropdropAction(
            crop_id=action_data["crop_id"],
            route=action_data["route"],
            destination_zone=action_data["destination_zone"]
        )
    except:
        # Fallback safe action
        return CropdropAction(crop_id=crops[0]["id"], route="paved", destination_zone="zone_1")

def run_baseline_agent(env, max_steps=20):
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
    
    # Easy
    env = CropdropEnvironment()
    env.crops = [env.crops[0]]
    results["easy"] = run_baseline_agent(env)
    
    # Medium
    env = CropdropEnvironment()
    results["medium"] = run_baseline_agent(env)
    
    # Hard
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
