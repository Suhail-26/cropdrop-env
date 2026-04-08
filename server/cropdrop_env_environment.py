"""
Core CropDrop environment logic.
Manages state, step execution, and trajectory tracking.
"""

import random
from typing import List, Dict, Any


# ─── Constants ─────────────────────────────────────────────────────────────────

CROP_TYPES = ["tomato", "lettuce", "carrot"]
ZONES = ["zone_1", "zone_2"]
ROUTES = {
    "paved": {"base_time": 2, "congestion_factor": 0.2},
    "dirt":  {"base_time": 3, "congestion_factor": 0.1},
    "muddy": {"base_time": 5, "congestion_factor": 0.05},
}
SPOILAGE_RATES = {
    "tomato":  3,   # spoils fastest
    "lettuce": 5,
    "carrot":  8,   # spoils slowest
}
INITIAL_SPOILAGE = {
    "tomato":  6,
    "lettuce": 10,
    "carrot":  16,
}


class CropdropEnvironment:
    """CropDrop Logistics simulation environment."""

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._trajectory: Dict[str, Any] = {}
        self.reset()

    # ─── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        crops = []
        for i, crop_type in enumerate(CROP_TYPES, start=1):
            crops.append(
                {
                    "id": i,
                    "type": crop_type,
                    "spoilage_remaining": INITIAL_SPOILAGE[crop_type],
                    "intended_zone": random.choice(ZONES),
                }
            )

        self._state = {
            "current_time": 0,
            "crops": crops,
            "routes_status": self._generate_routes(),
            "pending_orders": [c["id"] for c in crops],
            "reward": 0.0,
            "done": False,
            "last_action_result": None,
        }

        self._trajectory = {
            "delivered_crops": [],
            "total_time": 0,
            "muddy_route_count": 0,
            "steps": [],
        }

        return dict(self._state)

    # ─── Step ──────────────────────────────────────────────────────────────────

    def step(self, action) -> Dict[str, Any]:
        if self._state.get("done", False):
            return {
                "observation": self._state,
                "reward": 0.0,
                "done": True,
                "info": {"message": "Episode already finished. Call /reset."},
            }

        # Support both Pydantic model and plain dict
        if hasattr(action, "crop_id"):
            crop_id = action.crop_id
            route = action.route
            destination_zone = action.destination_zone
        else:
            crop_id = action.get("crop_id", 1)
            route = action.get("route", "paved")
            destination_zone = action.get("destination_zone", "zone_1")

        # Find the crop
        crop = next(
            (c for c in self._state["crops"] if c["id"] == crop_id), None
        )

        reward = 0.0
        action_result = "invalid_crop"

        if crop is None:
            action_result = f"crop_id {crop_id} not found or already delivered"
        else:
            # Travel time
            route_info = ROUTES.get(route, ROUTES["paved"])
            congestion = self._state["routes_status"].get(route, {}).get(
                "congestion", 0
            )
            travel_time = int(
                route_info["base_time"] * (1 + congestion * route_info["congestion_factor"])
            )

            # Tick spoilage for all remaining crops
            for c in self._state["crops"]:
                c["spoilage_remaining"] = max(
                    0, c["spoilage_remaining"] - travel_time
                )

            self._state["current_time"] += travel_time

            # Track muddy usage
            if route == "muddy":
                self._trajectory["muddy_route_count"] += 1

            # Score the delivery
            correct_zone = crop["intended_zone"] == destination_zone
            still_fresh = crop["spoilage_remaining"] > 0

            if correct_zone and still_fresh:
                freshness_bonus = round(crop["spoilage_remaining"] / INITIAL_SPOILAGE[crop["type"]] * 0.3, 4)
                reward = round(0.7 + freshness_bonus, 4)
                action_result = "correct_zone_fresh"
            elif correct_zone and not still_fresh:
                reward = 0.5
                action_result = "correct_zone_spoiled"
            elif not correct_zone and still_fresh:
                reward = 0.2
                action_result = "wrong_zone_fresh"
            else:
                reward = 0.0
                action_result = "wrong_zone_spoiled"

            # Record delivered crop
            delivered_record = {
                "id": crop["id"],
                "type": crop["type"],
                "intended_zone": crop["intended_zone"],
                "delivered_zone": destination_zone,
                "spoilage_remaining": crop["spoilage_remaining"],
                "route_used": route,
                "travel_time": travel_time,
                "reward": reward,
            }
            self._trajectory["delivered_crops"].append(delivered_record)

            # Remove crop from active list
            self._state["crops"] = [
                c for c in self._state["crops"] if c["id"] != crop_id
            ]
            if crop_id in self._state["pending_orders"]:
                self._state["pending_orders"].remove(crop_id)

        # Update trajectory totals
        self._trajectory["total_time"] = self._state["current_time"]
        self._trajectory["steps"].append(
            {
                "time": self._state["current_time"],
                "action": {
                    "crop_id": crop_id,
                    "route": route,
                    "destination_zone": destination_zone,
                },
                "reward": reward,
                "result": action_result,
            }
        )

        # Refresh route congestion
        self._state["routes_status"] = self._generate_routes()
        self._state["reward"] = reward
        self._state["last_action_result"] = action_result

        # Episode ends when all crops delivered or time limit hit
        done = len(self._state["crops"]) == 0 or self._state["current_time"] >= 50
        self._state["done"] = done

        return {
            "observation": dict(self._state),
            "reward": reward,
            "done": done,
            "info": {"action_result": action_result},
        }

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def get_trajectory(self) -> Dict[str, Any]:
        return dict(self._trajectory)

    def _generate_routes(self) -> Dict[str, Any]:
        routes_status = {}
        for route_name, info in ROUTES.items():
            congestion = round(random.uniform(0, 1), 2)
            estimated_time = int(
                info["base_time"] * (1 + congestion * info["congestion_factor"])
            )
            routes_status[route_name] = {
                "congestion": congestion,
                "estimated_time": estimated_time,
            }
        return routes_status
