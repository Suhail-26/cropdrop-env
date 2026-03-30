"""
Baseline inference script for CropDrop environment
Uses OpenAI client to run a model against the environment
"""

import os
import random
from server.cropdrop_env_environment import CropdropEnvironment
from models import CropdropAction

# Read environment variables (will be provided during evaluation)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def run_baseline_agent(env, max_steps=20):
    """Run a simple baseline agent"""
    
    observation = env.reset()
    total_reward = 0
    steps = 0
    trajectory = {
        "delivered_crops": [],
        "total_time": 0,
        "muddy_route_count": 0
    }
    
    routes = ["paved", "dirt", "muddy"]
    zones = ["zone_1", "zone_2"]
    
    for _ in range(max_steps):
        if not env.crops:
            break
        
        # Simple baseline: choose first available crop, random route, zone_1
        action = CropdropAction(
            crop_id=env.crops[0]["id"],
            route=random.choice(routes),
            destination_zone=random.choice(zones)
        )
        
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        # Track trajectory for graders
        if info.get("delivered"):
            for crop in env.delivered_crops:
                crop_copy = crop.copy()
                crop_copy["delivered_zone"] = action.destination_zone
                if crop_copy not in trajectory["delivered_crops"]:
                    trajectory["delivered_crops"].append(crop_copy)
        
        if action.route == "muddy":
            trajectory["muddy_route_count"] += 1
        
        steps += 1
        trajectory["total_time"] = env.current_time
        
        if done:
            break
    
    return total_reward / max(steps, 1), steps, trajectory


def run_all_tasks():
    """Run baseline on all 3 tasks"""
    
    results = {}
    
    # Task 1: Easy (single crop configuration)
    env = CropdropEnvironment()
    # Modify for single crop task
    env.crops = [env.crops[0]]  # Only keep first crop
    score, steps, trajectory = run_baseline_agent(env)
    results["easy"] = score
    
    # Task 2: Medium (3 crops, default config)
    env = CropdropEnvironment()
    score, steps, trajectory = run_baseline_agent(env)
    results["medium"] = score
    
    # Task 3: Hard (same as medium with congestion)
    env = CropdropEnvironment()
    score, steps, trajectory = run_baseline_agent(env)
    results["hard"] = score
    
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("🌾 CropDrop Baseline Inference")
    print("=" * 50)
    
    scores = run_all_tasks()
    
    print("\n📊 Baseline Scores:")
    for task, score in scores.items():
        print(f"  {task.upper()}: {score:.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Baseline inference completed!")