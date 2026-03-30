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
        """Initialize new episode"""
        self.current_time = 0
        self.crops = self._generate_crops()
        self.delivered_crops = []
        
        # Route configurations
        self.routes = {
            "paved": {"base_time": 2, "congestion": 0},
            "dirt": {"base_time": 4, "congestion": 0},
            "muddy": {"base_time": 7, "congestion": 0}
        }
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one action"""
        
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
        
        # Calculate delivery
        route_info = self.routes[action.route]
        travel_time = route_info["base_time"] + route_info["congestion"]
        
        # Check if crop spoils during delivery
        if travel_time >= crop["spoilage_remaining"]:
            # Crop spoiled during delivery
            reward = 0.0
            crop["spoilage_remaining"] = 0
        else:
            # Successful delivery
            crop["spoilage_remaining"] -= travel_time
            
            # Check if correct destination
            if crop["intended_zone"] == action.destination_zone:
                # Freshness bonus
                freshness = crop["spoilage_remaining"] / crop["initial_spoilage"]
                reward = 0.7 + (freshness * 0.3)
                # Track delivered zone for grader
                crop_copy = crop.copy()
                crop_copy["delivered_zone"] = action.destination_zone
                self.delivered_crops.append(crop_copy)
                self.crops.remove(crop)
            else:
                # Wrong zone
                reward = 0.2
        
        # Update time and congestion
        self.current_time += travel_time
        self._update_congestion()
        
        # Check if done
        done = len(self.crops) == 0
        
        observation = self._get_observation()
        
        return observation, reward, done, {"delivered": len(self.delivered_crops)}
    
    def state(self):
        """Return episode metadata"""
        return {
            "current_time": self.current_time,
            "crops_remaining": len(self.crops),
            "delivered": len(self.delivered_crops),
            "spoiled": len([c for c in self.crops if c["spoilage_remaining"] <= 0])
        }
    
    def _get_observation(self):
        """Create observation object"""
        return CropdropObservation(
            current_time=self.current_time,
            crops=self.crops.copy(),
            routes_status={
                k: {"congestion": v["congestion"], "estimated_time": v["base_time"] + v["congestion"]}
                for k, v in self.routes.items()
            },
            pending_orders=[],
            last_action_result=None
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
        
        # Randomly select 3 crops
        selected = random.sample(crop_types, 3)
        
        crops = []
        for i, crop in enumerate(selected, 1):
            # Add random variation to spoilage (±20%)
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