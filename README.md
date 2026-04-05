---
title: CropDrop Logistics
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# 🌾 CropDrop Logistics

**Agricultural last‑mile delivery environment for AI agents**

[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue)](https://huggingface.co/spaces/suhailma/cropdrop-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

CropDrop Logistics is a realistic simulation where an AI agent manages time‑sensitive crop deliveries from farms to distribution centers. Crops spoil over time, routes have different speeds and congestion, and each delivery must go to the correct zone. The environment is built with [OpenEnv](https://github.com/openenv/openenv) and exposes a FastAPI server for easy integration.

### 🌍 Real‑world problem

Food spoilage during transportation causes **30–40% of global food loss**. This environment trains AI agents to:
- Prioritize deliveries based on spoilage urgency
- Choose optimal routes under dynamic congestion
- Balance speed vs. accuracy to maximize freshness

---

## 🎮 Environment Details

### Action space (`CropdropAction`)

| Field | Type | Description |
|-------|------|-------------|
| `crop_id` | `int` | Which crop to deliver (1, 2, or 3) |
| `route` | `"paved"`, `"dirt"`, `"muddy"` | Road type (fast, medium, slow) |
| `destination_zone` | `"zone_1"`, `"zone_2"` | Delivery zone |

### Observation space (`CropdropObservation`)

| Field | Type | Description |
|-------|------|-------------|
| `current_time` | `int` | Steps elapsed in the episode |
| `crops` | `List[Dict]` | Each crop: id, type, spoilage remaining, intended zone |
| `routes_status` | `Dict` | Congestion and estimated time for each route |
| `pending_orders` | `List` | Orders waiting to be delivered |
| `reward` | `float` | Immediate reward from the last step |
| `done` | `bool` | Whether the episode is finished |

### Reward function

| Outcome | Reward |
|---------|--------|
| Correct zone, fresh crop | `0.7 + freshness_bonus` (0–0.3) |
| Correct zone, spoiled | `0.5` |
| Wrong zone | `0.2` |
| Spoiled during delivery | `0.0` |

---

## 🎯 Tasks (3 difficulty levels)

| Task | Difficulty | Description | Grader criteria |
|------|------------|-------------|----------------|
| **Single Priority Delivery** | Easy | Deliver 1 specific crop to correct zone before spoilage | 1.0 = correct & fresh, 0.5 = correct but spoiled, 0.0 = wrong/spoiled |
| **Multi‑Crop Prioritization** | Medium | Deliver 3 crops, prioritise fast‑spoiling crops | Base score + bonus for delivering tomato first |
| **Route Optimization** | Hard | Optimise routes with dynamic congestion | 50% correct deliveries + 30% time efficiency + 20% route choice |

---

## 🚀 Quick Start

### Local development

```bash
# Clone the repository
git clone https://github.com/Suhail-26/cropdrop-env.git
cd cropdrop-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate the environment
openenv validate

# Run baseline inference
python inference.py

# Start the API server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
Docker
bash
docker build -t cropdrop-env .
docker run -p 7860:7860 cropdrop-env
Hugging Face Space
The environment is live at:
https://huggingface.co/spaces/suhailma/cropdrop-env

📡 API Endpoints
Endpoint	Method	Description
/reset	POST	Start a new episode (empty body)
/step	POST	Execute an action (send CropdropAction JSON)
/state	GET	Get current episode metadata
/tasks	GET	List all tasks with schemas
/grader/{task_name}	POST	Get grader score for a completed episode
/health	GET	Health check for Hugging Face
/docs	GET	Interactive Swagger documentation
📊 Baseline Scores
Task	Score
Easy	0.02
Medium	0.01
Hard	0.88
These scores come from a random baseline agent. A trained agent would achieve higher values.

📁 Project Structure
text
cropdrop-env/
├── server/
│   ├── __init__.py
│   ├── app.py                     # FastAPI server
│   ├── cropdrop_env_environment.py # Core environment logic
│   └── Dockerfile                 # Container definition
├── models.py                      # Action and Observation models
├── openenv.yaml                   # Environment manifest
├── client.py                      # OpenEnv client
├── graders.py                     # Easy, Medium, Hard graders
├── inference.py                   # Baseline inference script
├── test_1000_episodes.py          # Performance test
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
├── uv.lock                        # Locked dependencies
└── README.md                      # This file
🧪 Testing
bash
# Run 1000 random episodes to collect statistics
python test_1000_episodes.py

# Visualise an episode in the terminal
python visualize.py

🤝 Acknowledgments
Built with OpenEnv
Designed for the OpenEnv Hackathon

📄 License
MIT License – free to use and modify.

🔗 Links
GitHub repository: https://github.com/Suhail-26/cropdrop-env

Hugging Face Space: https://huggingface.co/spaces/suhailma/cropdrop-env
