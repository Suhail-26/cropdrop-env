"""
CropOps Agent - Round 2 FastAPI Server
Replaces Round 1 CropDrop server.

Same API shape (/reset, /step, /state, /tasks, /grader/{task_name})
so the OpenEnv client needs minimal changes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import json

from server.cropops_env_environment import CropOpsEnvironment
from models import CropOpsAction, CropOpsObservation
from graders import EasyGrader, MediumGrader, HardGrader

app = FastAPI(
    title="CropOps Agent Environment",
    description="Round 2 — Grid-based agricultural delivery with disruptions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session; fine for hackathon)
env = CropOpsEnvironment()
episode_trajectory: Dict[str, Any] = {}


# ── /reset ────────────────────────────────────────────────────
@app.post("/reset")
def reset():
    """Start a new episode. Returns the initial observation."""
    global episode_trajectory
    obs = env.reset()
    episode_trajectory = {
        "delivered_crops": [],
        "total_steps": 0,
        "disruptions_encountered": 0,
    }
    return obs


# ── /step ─────────────────────────────────────────────────────
@app.post("/step")
def step(action: CropOpsAction):
    """
    Execute one action. Returns observation, reward, done, info.
    Action must be one of: up, down, left, right, pickup, dropoff
    """
    obs, reward, done, info = env.step(action.action)

    # Update trajectory for graders
    episode_trajectory["total_steps"] = obs["current_time"]
    episode_trajectory["disruptions_encountered"] = len(info.get("disruptions", []))
    episode_trajectory["delivered_crops"] = env.delivered_crops

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


# ── /state ────────────────────────────────────────────────────
@app.get("/state")
def get_state():
    """Get current environment state and episode metadata."""
    return {
        "current_time": env.current_time,
        "agent_position": env.agent_pos,
        "done": env.done,
        "total_reward": env.total_reward,
        "crops_delivered": len(env.delivered_crops),
        "crops_pending": len([c for c in env.crops if not c["delivered"]]),
        "disruptions_so_far": len(env.disruptions_log),
    }


# ── /tasks ────────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    """List all tasks with action/observation schemas."""
    return {
        "tasks": [
            {
                "name": "single_priority_delivery",
                "difficulty": "easy",
                "description": "Pick up 1 crop and deliver it to the correct warehouse zone before it spoils.",
                "max_steps": 20,
                "action_schema": {
                    "action": "one of: up, down, left, right, pickup, dropoff"
                },
            },
            {
                "name": "multi_crop_prioritization",
                "difficulty": "medium",
                "description": "Deliver all 3 crops to correct zones. Prioritize the fast-spoiling tomato first.",
                "max_steps": 40,
                "action_schema": {
                    "action": "one of: up, down, left, right, pickup, dropoff"
                },
            },
            {
                "name": "route_optimization",
                "difficulty": "hard",
                "description": "Deliver all crops efficiently despite road blocks, rain, and vehicle breakdowns.",
                "max_steps": 50,
                "action_schema": {
                    "action": "one of: up, down, left, right, pickup, dropoff"
                },
            },
        ]
    }


# ── /grader/{task_name} ───────────────────────────────────────
@app.post("/grader/{task_name}")
def grade(task_name: str):
    """Grade the completed episode for the given task."""
    graders = {
        "single_priority_delivery": EasyGrader(),
        "multi_crop_prioritization": MediumGrader(),
        "route_optimization": HardGrader(),
    }
    if task_name not in graders:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")

    score = graders[task_name].grade(episode_trajectory)
    return {
        "task": task_name,
        "score": score,
        "delivered_crops": len(episode_trajectory.get("delivered_crops", [])),
        "total_steps": episode_trajectory.get("total_steps", 0),
        "disruptions_encountered": episode_trajectory.get("disruptions_encountered", 0),
    }


# ── /render ───────────────────────────────────────────────────
@app.get("/render")
def render():
    """ASCII render of the current grid state (for debugging)."""
    return {"grid": env.render()}


# ── /health ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "project": "CropOps Agent"}