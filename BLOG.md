\# CropOps Agent — Teaching an AI to Deliver Crops Before They Spoil



\*OpenEnv Hackathon India 2026 — Theme 3: World Modeling\*



\---



\## The Problem



\*\*30 to 40 percent of all food produced globally is lost during last-mile delivery.\*\*  

Not in the fields. Not in storage. In transit — navigating real disruptions like road blocks, bad weather, and vehicle failures while racing against spoilage clocks.



We built \*\*CropOps Agent\*\*: a reinforcement learning agent that learns to solve exactly this problem in a simulated agricultural grid world.



\---



\## The Real-World Connection



Think of it like this. A self-driving car must:



\- Read sensors to understand its environment

\- Maintain a map of obstacles and other vehicles

\- Plan a route to the destination

\- Adapt in real time to rain, accidents, and road closures



Our CropOps Agent does the same thing — for agricultural delivery:



\- Read crop positions, spoilage timers, and warehouse zones

\- Maintain a world model of disruptions and blocked paths

\- Plan multi-step routes: village → pickup → warehouse → dropoff

\- Adapt to rain (crops spoil faster), road blocks, and vehicle breakdowns



This is \*\*World Modeling in action\*\* — the core capability behind modern autonomous systems, now applied to food logistics.



\---



\## The Environment



We built a \*\*10×10 grid world\*\* with:



\- \*\*4 villages\*\* — where crops spawn and wait to be picked up

\- \*\*2 warehouses\*\* — zone\_1 at (4,2) and zone\_2 at (4,7)

\- \*\*3 crops per episode\*\* — each with a specific zone destination and a spoilage timer

\- \*\*Random disruptions\*\* — 2 per episode: road blocks, rain, or vehicle breakdowns

\- \*\*Static walls\*\* — the agent must navigate around mid-grid obstacles



The agent has 6 actions: `up`, `down`, `left`, `right`, `pickup`, `dropoff`.



The reward structure teaches urgency and accuracy:



\- Deliver to the correct zone, fresh and on time → \*\*+15\*\*

\- Correct zone, fresh → \*\*+10\*\*

\- Correct zone, spoiled → \*\*+5\*\*

\- Wrong zone → \*\*−3\*\*

\- Wall collision → \*\*−1\*\*



\---



\## Training — Two Approaches



\### Q-Learning with Curriculum (3000 episodes)



We trained a Q‑table agent using \*\*curriculum learning\*\* — starting with 1 crop, then 2, then 3. This filled the Q‑table systematically so the agent learned what to do after each delivery, not just the first one.



We added \*\*BFS-based reward shaping\*\* (bonus for each step closer to the target) and \*\*experience replay\*\* (reusing rare multi‑delivery transitions many times).



\*\*Results after 3000 episodes:\*\*



|                | Before Training | After Training |

|----------------|----------------|----------------|

| Avg Reward     | −13.41         | \*\*+62.76\*\*     |

| Avg Deliveries | 0.10 / 3       | \*\*2.27 / 3\*\*   |

| Improvement    | —              | \*\*+76.17 reward\*\* |



!\[Q-Learning Training Curve](reward\_curve.png)



\*The reward curve shows steady upward learning across 3000 episodes. Red dashed lines mark curriculum phase transitions — you can see the agent reset and re-learn at each phase, then climb higher than before.\*



\---



\### HuggingFace TRL / GRPO (SmolLM2-135M-Instruct)



We also fine-tuned a \*\*135M parameter language model\*\* using GRPO — the same algorithm that trained DeepSeek‑R1 — via HuggingFace TRL.



Each training prompt describes the real current observation in natural language:



```text

You are a delivery robot in a 10x10 grid (CropOps).

Your position: row=1, col=1.

CARRYING: tomato → deliver to zone\_1 at (4,2).

Spoilage remaining: 38 steps. Distance to warehouse: 5 steps.

Crops to collect (2 remaining):

&#x20; - wheat at row=1,col=8 → zone\_2, spoilage=44, dist=7

&#x20; - corn  at row=8,col=1 → zone\_1, spoilage=40, dist=7



Output ONLY a comma-separated action sequence (15-30 actions).

Actions:

The LLM outputs: down, down, down, dropoff, up, right, right...



Those actions are replayed in the live environment, and the reward is returned to GRPO for weight updates.



Results (64 training samples):



Before	After TRL/GRPO

Avg Reward	−0.47	+8.00

Improvement	—	+8.47

https://reward\_curve\_trl.png



What the Agent Learned — Without Being Told

These behaviors emerged from the reward signal alone:



Spoilage‑priority routing — The agent learned to visit the village with the lowest remaining spoilage first, not the nearest one. Urgency emerged from reward.



Zone‑aware navigation — It learned that zone\_1 (4,2) and zone\_2 (4,7) are different targets with different rewards. It routes to the correct warehouse even when the wrong one is closer.



Disruption avoidance — When a road block activates, the agent reroutes rather than repeatedly hitting the blocked tile.



Opportunity pickup — It learned to pick up a crop immediately when spawning on a village tile — exploiting the starting position.



Try It Live

The CropOps environment is hosted on HuggingFace Spaces and available via HTTP:



python

import requests



\# Reset

obs = requests.post("https://suhailma-cropdrop-env.hf.space/reset").json()



\# Step

result = requests.post(

&#x20;   "https://suhailma-cropdrop-env.hf.space/step",

&#x20;   json={"action": "down"}

).json()



print("Reward:", result\["reward"])

Links:



🤗 HuggingFace Space: https://huggingface.co/spaces/suhailma/cropdrop-env



💻 GitHub: https://github.com/Suhail-26/cropdrop-env/tree/round2-cropops



📓 Training scripts: train\_colab.py (Q‑Learning) and trl\_colab.py (TRL/GRPO)



Team

Suhail (environment + deployment) · Shaheer (training) · Ismayeel (rewards + blog)



Built for OpenEnv Hackathon India 2026 — Theme 3: World Modeling

