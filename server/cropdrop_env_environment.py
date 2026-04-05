"""
CropDrop Environment - Agricultural last-mile delivery simulation
"""

import random
from models import CropdropAction, CropdropObservation

class CropdropEnvironment:
    """Crop delivery logistics environment with spoilage timing"""

    def __init__(self):
        """Initialize the environment"""
        self.reset()

    def reset(self):
        """Initialize new episode (sync version)"""
        self.current_time = 0
        self.crops = self._generate_crops()
        self.delivered_crops = []
        self.trajectory = {
            "delivered_crops": [],
            "total_time": 0,
            "muddy_route_count": 0
        }
        self._last_action_result = None
        self.routes = {
            "paved": {"base_time": 2, "congestion": 0},
            "dirt": {"base_time": 4, "congestion": 0},
            "muddy": {"base_time": 7, "congestion": 0}
        }
        return self._get_observation()

    async def reset_async(self):
        """Async wrapper for reset (required by OpenEnv server)"""
        return self.reset()

    def step(self, action):
        """Execute one action (sync version)"""
        # Find the crop being delivered
        crop = None
        for c in self.crops:
            if c["id"] == action.crop_id:
                crop = c
                break

        if not crop:
            obs = self._get_observation()
            return obs, 0.0, False, {"error": "Invalid crop"}

        if crop["spoilage_remaining"] <= 0:
            obs = self._get_observation()
            return obs, 0.0, False, {"error": "Crop already spoiled"}

        route_info = self.routes[action.route]
        travel_time = route_info["base_time"] + route_info["congestion"]

        reward = 0.0
        freshness = 0.0

        if travel_time >= crop["spoilage_remaining"]:
            crop["spoilage_remaining"] = 0
            self._last_action_result = "spoiled"
        else:
            crop["spoilage_remaining"] -= travel_time
            if crop["intended_zone"] == action.destination_zone:
                freshness = crop["spoilage_remaining"] / crop["initial_spoilage"]
                reward = 0.7 + (freshness * 0.3)
                crop_copy = crop.copy()
                crop_copy["delivered_zone"] = action.destination_zone
                self.delivered_crops.append(crop_copy)
                self.trajectory["delivered_crops"].append(crop_copy)
                self.crops.remove(crop)
                self._last_action_result = "success"
            else:
                reward = 0.2
                self._last_action_result = "wrong_zone"

        self.current_time += travel_time
        self._update_congestion()

        if action.route == "muddy":
            self.trajectory["muddy_route_count"] += 1
        self.trajectory["total_time"] = self.current_time

        done = len(self.crops) == 0
        observation = self._get_observation(reward=reward, freshness=freshness)
        info = {"delivered": len(self.delivered_crops)}
        return observation, reward, done, info

    async def step_async(self, action):
        """Async wrapper for step (required by OpenEnv server)"""
        return self.step(action)

    def state(self):
        """Return episode metadata"""
        return {
            "current_time": self.current_time,
            "crops_remaining": len(self.crops),
            "delivered": len(self.delivered_crops),
            "spoiled": len([c for c in self.crops if c["spoilage_remaining"] <= 0])
        }

    def close(self):
        """Clean up resources (required by OpenEnv server)"""
        pass

    def _get_observation(self, reward=0.0, freshness=0.0):
        """Create observation object"""
        return CropdropObservation(
            current_time=self.current_time,
            crops=self.crops.copy(),
            routes_status={
                k: {"congestion": v["congestion"], "estimated_time": v["base_time"] + v["congestion"]}
                for k, v in self.routes.items()
            },
            pending_orders=[],
            last_action_result=self._last_action_result,
            delivered_zone=self.trajectory["delivered_crops"][-1].get("delivered_zone") if self.trajectory["delivered_crops"] else None,
            reward_breakdown={"reward": reward, "freshness_bonus": freshness} if reward > 0 else None
        )

    def _generate_crops(self):
        """Generate random crops with varying spoilage times and zones"""
        crop_types = [
            {"type": "tomato", "color": "red", "spoilage": 10, "zone": "zone_1"},
            {"type": "corn", "color": "yellow", "spoilage": 15, "zone": "zone_2"},
            {"type": "potato", "color": "brown", "spoilage": 20, "zone": "zone_1"},
            {"type": "carrot", "color": "orange", "spoilage": 12, "zone": "zone_2"},
            {"type": "lettuce", "color": "green", "spoilage": 8, "zone": "zone_1"},
        ]
        selected = random.sample(crop_types, 3)
        crops = []
        for i, crop in enumerate(selected, 1):
            spoilage = crop["spoilage"] * random.uniform(0.8, 1.2)
            crops.append({
                "id": i,
                "type": crop["type"],
                "color": crop["color"],
                "spoilage_remaining": int(spoilage),
                "initial_spoilage": int(spoilage),
                "intended_zone": crop["zone"],
            })
        return crops

    def _update_congestion(self):
        """Update route congestion dynamically"""
        for route in self.routes.values():
            route["congestion"] = max(0, route["congestion"] + random.randint(-1, 2))

    def get_grader_score(self, task_name: str):
        """Return grader score for a given task"""
        from graders import EasyGrader, MediumGrader, HardGrader
        if task_name == "single_priority_delivery":
            return EasyGrader().grade(self.trajectory)
        elif task_name == "multi_crop_prioritization":
            return MediumGrader().grade(self.trajectory)
        elif task_name == "route_optimization":
            return HardGrader().grade(self.trajectory)
        else:
            return 0.0