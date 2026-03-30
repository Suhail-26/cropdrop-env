"""
Test CropDrop environment with 1000+ random episodes
Collects statistics on rewards, deliveries, and performance
"""

from server.cropdrop_env_environment import CropdropEnvironment
from models import CropdropAction
import random
import time
from datetime import datetime

def run_single_episode(episode_num):
    """Run one episode with random actions and return stats"""
    
    env = CropdropEnvironment()
    obs = env.reset()
    
    total_reward = 0
    steps = 0
    max_steps = 50
    deliveries = 0
    spoilage_count = 0
    route_usage = {"paved": 0, "dirt": 0, "muddy": 0}
    
    routes = ["paved", "dirt", "muddy"]
    zones = ["zone_1", "zone_2"]
    
    while steps < max_steps and env.crops:
        # Random action
        action = CropdropAction(
            crop_id=env.crops[0]["id"] if env.crops else 1,
            route=random.choice(routes),
            destination_zone=random.choice(zones)
        )
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        route_usage[action.route] += 1
        
        # Track deliveries
        if info.get("delivered"):
            deliveries += 1
        
        if done:
            break
    
    # Count spoiled crops
    for crop in env.crops:
        if crop.get("spoilage_remaining", 0) <= 0:
            spoilage_count += 1
    
    return {
        "episode": episode_num,
        "total_reward": total_reward,
        "steps": steps,
        "deliveries": deliveries,
        "spoiled": spoilage_count,
        "route_usage": route_usage,
        "crops_remaining": len(env.crops)
    }


def run_batch(num_episodes=1000):
    """Run multiple episodes and collect statistics"""
    
    print("=" * 60)
    print(f"🌾 CropDrop - Running {num_episodes} Episodes")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    for i in range(num_episodes):
        result = run_single_episode(i + 1)
        results.append(result)
        
        # Show progress every 100 episodes
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{num_episodes} episodes")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate statistics
    total_rewards = [r["total_reward"] for r in results]
    total_steps = [r["steps"] for r in results]
    total_deliveries = [r["deliveries"] for r in results]
    total_spoiled = [r["spoiled"] for r in results]
    
    # Route usage
    route_paved = sum(r["route_usage"]["paved"] for r in results)
    route_dirt = sum(r["route_usage"]["dirt"] for r in results)
    route_muddy = sum(r["route_usage"]["muddy"] for r in results)
    total_actions = route_paved + route_dirt + route_muddy
    
    print("\n" + "=" * 60)
    print("📊 STATISTICS SUMMARY")
    print("=" * 60)
    print(f"📈 Episodes run: {num_episodes}")
    print(f"⏱️  Time taken: {elapsed:.2f} seconds")
    print(f"⚡ Average time per episode: {elapsed/num_episodes:.2f} sec")
    print("\n💰 REWARDS:")
    print(f"   Average reward: {sum(total_rewards)/num_episodes:.3f}")
    print(f"   Best reward: {max(total_rewards):.3f}")
    print(f"   Worst reward: {min(total_rewards):.3f}")
    
    print("\n📦 DELIVERIES:")
    print(f"   Total deliveries: {sum(total_deliveries)}")
    print(f"   Average per episode: {sum(total_deliveries)/num_episodes:.2f}")
    print(f"   Total spoiled crops: {sum(total_spoiled)}")
    
    print("\n🛣️ ROUTE USAGE:")
    print(f"   Paved: {route_paved} ({route_paved/total_actions*100:.1f}%)")
    print(f"   Dirt: {route_dirt} ({route_dirt/total_actions*100:.1f}%)")
    print(f"   Muddy: {route_muddy} ({route_muddy/total_actions*100:.1f}%)")
    
    print("\n📊 EPISODE COMPLETION:")
    episodes_with_all_delivered = sum(1 for r in results if r["deliveries"] == 3)
    print(f"   Full deliveries (3/3): {episodes_with_all_delivered}/{num_episodes} ({episodes_with_all_delivered/num_episodes*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run 1000 episodes
    results = run_batch(1000)
    
    print("\n💾 To save results, run:")
    print("   python -c \"import json; from test_1000_episodes import run_batch; results = run_batch(1000); open('results.json', 'w').write(json.dumps(results))\"")