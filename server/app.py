# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with:\n"
        "  pip install -r requirements.txt\n"
    ) from e

try:
    from models import CropdropAction, CropdropObservation
    from server.cropdrop_env_environment import CropdropEnvironment
except ModuleNotFoundError:
    from ..models import CropdropAction, CropdropObservation
    from .cropdrop_env_environment import CropdropEnvironment


# Create a single shared environment instance so the grader endpoint
# can access its trajectory.
env_instance = CropdropEnvironment()

# create_app requires a *callable* (class or factory), not an instance.
# We wrap env_instance in a lambda so every call returns the SAME object,
# preserving the shared-state behavior we want.
app = create_app(
    env=lambda: env_instance,
    action_cls=CropdropAction,
    observation_cls=CropdropObservation,
    env_name="cropdrop_env",
)


# Health endpoint (required for Docker health check)
@app.get("/health")
def health():
    return {"status": "healthy"}


# Root endpoint
@app.get("/")
def root():
    return {"message": "CropDrop Environment is running"}


# Grader endpoint - uses the shared env_instance to access trajectory
@app.post("/grader/{task_name}")
async def grader(task_name: str):
    score = env_instance.get_grader_score(task_name)
    return {"task": task_name, "score": score}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
