"""
Baseline inference script for CropDrop environment
Uses the hackathon's LiteLLM proxy for LLM calls.
"""

import os
import json
import random
from openai import OpenAI
from server.cropdrop_env_environment import CropdropEnvironment
from models import CropdropAction

# ------------------------------------------------------------------
# CRITICAL: Use the environment variables injected by the evaluator
# Do NOT set defaults for API_KEY (must come from the proxy)
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")          # or HF_TOKEN
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Validate that the proxy credentials are present
if not API_BASE_URL or not API_KEY:
    raise ValueError(
        "API_BASE_URL and API_KEY must be set by the evaluator. "
        "Do not hardcode API keys or use other providers."
    )

# Initialize the OpenAI client with the proxy
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ------------------------------------------------------------------
# Helper: ask the LLM to choose an action
# ------------------------------------------------------------------
def get_llm_action(observation, crops, routes_status):
    """Send the current state to the LLM and parse its action choice."""
    
    # Build a prompt that describes the environment and available options
    prompt = f"""
You are controlling a crop delivery robot in a logistics environment.

Current time: {observation.current_time}

Available crops (each has id, type, spoilage remaining, intended zone):
{json.dumps(crops, indent=2)}

Route status (congestion and estimated travel time):
{json.dumps(routes_status, indent=2)}

Your goal is to deliver crops to their correct zones before they spoil.
Choose an action. Reply ONLY with a JSON object in this exact format:
{{"crop_id": <id>, "route": "paved" or "dirt" or "muddy", "destination_zone": "zone_1" or "zone_2"}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        # Parse the LLM's response
        action_text = response.choices[0].message.content.strip()
        # Extract JSON from the response (may be surrounded by backticks)
        if "```json" in action_text:
            action_text = action_text.split("```json")[1].split("```")[0]
        elif "```" in action_text:
            action_text = action_text.split("```")[1].split("```")[0]
        action_data = json.loads(action_text)
        return CropdropAction(
            crop_id=int(action_data["crop_id"]),
            route=action_data["route"],
            destination_zone=action_data["destination_zone"]
        )
    except Exception as e:
        # Fallback to a safe default action if parsing fails
        print(f"LLM parsing error: {e}, using fallback action")
        return CropdropAction(
            crop_id=crops[0]["id"],
            route="paved",
            destination_zone=crops[0]["intended_zone"]
        )

# ------------------------------------------------------------------
# Run one episode with the LLM agent
# ------------------------------------------------------------------
def run_agent(env, max_steps=20):
    """Run a single episode using the LLM to choose actions."""
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    
    for _ in range(max_steps):
        if not env.crops:
            break
        # Get action from LLM (makes an API call to the proxy)
        action = get_llm_action(obs, env.crops, env.routes)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    
    # Return average reward per step
    return total_reward / max(steps, 1)

# ------------------------------------------------------------------
# Run all three tasks
# ------------------------------------------------------------------
def run_all_tasks():
    results = {}
    
    # Easy task: single crop
    env = CropdropEnvironment()
    env.crops = [env.crops[0]]   # keep only the first crop
    results["easy"] = run_agent(env)
    
    # Medium task: default 3 crops, no extra congestion (already default)
    env = CropdropEnvironment()
    results["medium"] = run_agent(env)
    
    # Hard task: same as medium (congestion already dynamic)
    env = CropdropEnvironment()
    results["hard"] = run_agent(env)
    
    return results

# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("🌾 CropDrop Baseline Inference (using LiteLLM proxy)")
    print("=" * 50)
    scores = run_all_tasks()
    print("\n📊 Baseline Scores:")
    for task, score in scores.items():
        print(f"  {task.upper()}: {score:.2f}")
    print("=" * 50)
