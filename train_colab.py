"""
CropOps Agent - Round 2 Training Script
Run this in Google Colab.

What this does:
  1. Runs 500 episodes against the local environment
  2. Uses a simple Q-table (tabular RL) — no GPU needed
  3. Saves reward curve plot as reward_curve.png
  4. Prints before/after comparison (Episode 1 vs Episode 500)

Usage in Colab:
  !pip install requests matplotlib numpy
  # Then just run this file
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Import environment directly (no server needed for training) ──
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from server.cropops_env_environment import CropOpsEnvironment, ALL_ACTIONS


# ── Q-table agent ─────────────────────────────────────────────
class QAgent:
    def __init__(self, actions, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.actions       = actions
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = defaultdict(lambda: np.zeros(len(actions)))

    def state_key(self, obs):
        """Convert observation to a compact hashable state key."""
        agent = tuple(obs["agent_position"])
        carried_id = obs["carried_crop"]["id"] if obs["carried_crop"] else 0
        pending = tuple(sorted(obs["pending_orders"]))
        disrupted = len(obs.get("active_disruptions", [])) > 0
        breakdown = obs.get("breakdown_steps_remaining", 0) > 0
        return (agent, carried_id, pending, disrupted, breakdown)

    def choose_action(self, obs):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state = self.state_key(obs)
        return self.actions[np.argmax(self.q_table[state])]

    def learn(self, obs, action, reward, next_obs, done):
        state      = self.state_key(obs)
        next_state = self.state_key(next_obs)
        action_idx = self.actions.index(action)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action_idx] += self.lr * (target - self.q_table[state][action_idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ── Training loop ─────────────────────────────────────────────
def train(n_episodes=500):
    env   = CropOpsEnvironment()
    agent = QAgent(actions=ALL_ACTIONS)

    rewards_per_episode = []
    deliveries_per_episode = []

    print(f"Training CropOps Agent for {n_episodes} episodes...")
    print("-" * 50)

    for ep in range(1, n_episodes + 1):
        obs          = env.reset()
        total_reward = 0.0
        done         = False
        steps        = 0

        while not done and steps < env.MAX_STEPS:
            action              = agent.choose_action(obs)
            next_obs, r, done, _ = env.step(action)
            agent.learn(obs, action, r, next_obs, done)
            obs          = next_obs
            total_reward += r
            steps        += 1

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        deliveries_per_episode.append(len(env.delivered_crops))

        if ep % 50 == 0 or ep == 1:
            avg_r = np.mean(rewards_per_episode[-50:])
            avg_d = np.mean(deliveries_per_episode[-50:])
            print(f"  Ep {ep:4d} | ε={agent.epsilon:.3f} | avg_reward={avg_r:6.2f} | avg_deliveries={avg_d:.2f}")

    print("-" * 50)
    print("Training complete!\n")
    return agent, rewards_per_episode, deliveries_per_episode


# ── Plot reward curve ─────────────────────────────────────────
def plot_reward_curve(rewards, save_path="reward_curve.png"):
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CropOps Agent — Training Reward Curve (Round 2)", fontsize=14, fontweight='bold')

    # Raw rewards
    axes[0].plot(rewards, alpha=0.3, color='steelblue', linewidth=0.8, label='Raw reward')
    axes[0].plot(range(window - 1, len(rewards)), smoothed, color='steelblue', linewidth=2, label=f'{window}-ep moving avg')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Total Reward per Episode")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Mark Episode 1, 100, 500
    checkpoints = [0, 99, 499]
    for idx in checkpoints:
        if idx < len(rewards):
            axes[0].axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            axes[0].text(idx + 2, min(rewards), f'Ep {idx+1}', color='red', fontsize=8)

    # Rolling average only (clean version for slides)
    axes[1].plot(range(window - 1, len(rewards)), smoothed, color='green', linewidth=2)
    axes[1].fill_between(range(window - 1, len(rewards)), smoothed, alpha=0.15, color='green')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Smoothed Reward")
    axes[1].set_title(f"Learning Curve ({window}-Episode Smoothed)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Reward curve saved to: {save_path}")
    plt.show()


# ── Before vs After comparison ────────────────────────────────
def before_after_comparison(agent):
    """Run 5 episodes with epsilon=1 (random/before) vs epsilon=0 (trained/after)."""
    env = CropOpsEnvironment()

    print("\n=== BEFORE training (random agent) ===")
    before_rewards = []
    before_deliveries = []
    for _ in range(5):
        obs    = env.reset()
        total  = 0.0
        done   = False
        steps  = 0
        while not done and steps < env.MAX_STEPS:
            action             = random.choice(ALL_ACTIONS)
            obs, r, done, _   = env.step(action)
            total += r
            steps += 1
        before_rewards.append(total)
        before_deliveries.append(len(env.delivered_crops))
    print(f"  Avg reward:     {np.mean(before_rewards):.2f}")
    print(f"  Avg deliveries: {np.mean(before_deliveries):.2f} / 3")

    print("\n=== AFTER training (trained agent) ===")
    original_eps   = agent.epsilon
    agent.epsilon  = 0.0             # pure exploitation
    after_rewards     = []
    after_deliveries  = []
    for _ in range(5):
        obs    = env.reset()
        total  = 0.0
        done   = False
        steps  = 0
        while not done and steps < env.MAX_STEPS:
            action             = agent.choose_action(obs)
            obs, r, done, _   = env.step(action)
            total += r
            steps += 1
        after_rewards.append(total)
        after_deliveries.append(len(env.delivered_crops))
    print(f"  Avg reward:     {np.mean(after_rewards):.2f}")
    print(f"  Avg deliveries: {np.mean(after_deliveries):.2f} / 3")
    agent.epsilon = original_eps

    improvement = np.mean(after_rewards) - np.mean(before_rewards)
    print(f"\n  Reward improvement: {improvement:+.2f}")
    print(f"  Delivery improvement: {np.mean(after_deliveries) - np.mean(before_deliveries):+.2f}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    agent, rewards, deliveries = train(n_episodes=500)
    plot_reward_curve(rewards, save_path="reward_curve.png")
    before_after_comparison(agent)

    print("\nDone! Upload reward_curve.png to your HuggingFace blog post.")