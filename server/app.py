"""
FastAPI server for CropDrop Logistics environment.
Exposes REST endpoints for OpenEnv evaluation platform.
"""

import sys
import os

# Make sure the root package is importable when running from /server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import CropdropAction, CropdropObservation
from graders import GRADER_REGISTRY
from server.cropdrop_env_environment import CropdropEnvironment

app = FastAPI(
    title="CropDrop Logistics",
    description="Agricultural last-mile delivery environment for AI agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared environment instance
env = CropdropEnvironment()


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Environment endpoints ─────────────────────────────────────────────────────

@app.post("/reset")
async def reset():
    """Start a new episode."""
    obs = env.reset()
    return obs


@app.post("/step")
async def step(action: CropdropAction):
    """Execute one action and return next observation, reward, done, info."""
    result = env.step(action)
    return result


@app.get("/state")
async def state():
    """Return current episode metadata."""
    return env.get_state()


# ─── Tasks endpoint ────────────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks():
    """List all tasks with their schemas and grader info."""
    return {
        "tasks": [
            {
                "name": "single_priority_delivery",
                "difficulty": "easy",
                "description": "Deliver one crop to correct zone before spoilage",
                "grader": "EasyGrader",
                "has_grader": True,
            },
            {
                "name": "multi_crop_prioritization",
                "difficulty": "medium",
                "description": "Deliver 3 crops, prioritize fast-spoiling crops",
                "grader": "MediumGrader",
                "has_grader": True,
            },
            {
                "name": "route_optimization",
                "difficulty": "hard",
                "description": "Optimize routes with dynamic congestions",
                "grader": "HardGrader",
                "has_grader": True,
            },
        ]
    }


# ─── Grader endpoint ───────────────────────────────────────────────────────────

@app.post("/grader/{task_name}")
async def grade_task(task_name: str, trajectory: dict = None):
    """
    Grade a completed episode trajectory for the given task.
    Accepts trajectory data in the request body.
    Falls back to env.get_trajectory() if no body is provided.
    """
    grader = GRADER_REGISTRY.get(task_name)
    if grader is None:
        raise HTTPException(
            status_code=404,
            detail=f"No grader found for task '{task_name}'. "
                   f"Available tasks: {list(GRADER_REGISTRY.keys())}",
        )

    # Use provided trajectory, or pull it from the live environment
    if trajectory is None or not trajectory:
        trajectory = env.get_trajectory()

    score = grader.grade(trajectory)
    return {
        "task": task_name,
        "score": round(score, 4),
        "grader": type(grader).__name__,
    }


# ─── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
