"""
CropOps Agent - Round 2 Environment
Upgraded from Round 1 CropDrop:
  - Real 10x10 spatial grid (villages, warehouses, roads, blocked)
  - 6 movement actions (up/down/left/right + pickup/dropoff)
  - Random disruption events (road block, rain, breakdown)
  - Denser reward function
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional

# ── Tile types ──────────────────────────────────────────────
ROAD      = 0
VILLAGE   = 1
WAREHOUSE = 2
BLOCKED   = 3

# ── Actions ─────────────────────────────────────────────────
ACTION_UP       = "up"
ACTION_DOWN     = "down"
ACTION_LEFT     = "left"
ACTION_RIGHT    = "right"
ACTION_PICKUP   = "pickup"
ACTION_DROPOFF  = "dropoff"
ALL_ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PICKUP, ACTION_DROPOFF]

# ── Disruption types ─────────────────────────────────────────
DISRUPTION_ROAD_BLOCK  = "road_block"
DISRUPTION_RAIN        = "rain"
DISRUPTION_BREAKDOWN   = "vehicle_breakdown"
ALL_DISRUPTIONS = [DISRUPTION_ROAD_BLOCK, DISRUPTION_RAIN, DISRUPTION_BREAKDOWN]


class CropOpsEnvironment:
    """
    10x10 grid world for agricultural last-mile delivery.

    Grid layout (fixed):
      - 4 villages (crop pickup points)
      - 2 warehouses (zone_1, zone_2 drop-off points)
      - Road tiles connecting everything
      - A few blocked tiles (walls / impassable terrain)

    Agent starts at center, must pick up crops from villages
    and deliver them to the correct warehouse zone.
    """

    GRID_SIZE = 10
    MAX_STEPS = 50
    MAX_CARRIED = 1          # agent can carry one crop at a time
    DISRUPTIONS_PER_EPISODE = 3

    # Fixed positions on the 10x10 grid
    VILLAGE_POSITIONS = {
        1: (1, 1),   # top-left cluster
        2: (1, 8),   # top-right cluster
        3: (8, 1),   # bottom-left cluster
        4: (8, 8),   # bottom-right cluster
    }
    WAREHOUSE_POSITIONS = {
        "zone_1": (4, 2),
        "zone_2": (4, 7),
    }
    AGENT_START = (5, 5)

    # Blocked tiles (static obstacles)
    STATIC_BLOCKED = [
        (3, 4), (3, 5), (3, 6),   # horizontal wall in middle
        (6, 4), (6, 5), (6, 6),   # second horizontal wall
    ]

    def __init__(self):
        self.grid = None
        self.agent_pos = None
        self.crops = []
        self.carried_crop = None
        self.delivered_crops = []
        self.current_time = 0
        self.total_reward = 0.0
        self.done = False
        self.disruptions_log = []
        self.active_blocked = set()   # tiles blocked by disruption events
        self.breakdown_steps = 0      # steps agent is immobilised
        self.rain_active = False      # rain slows spoilage tick?  (cosmetic flag)
        self.disruption_schedule = []
        self._build_grid()

    # ── Grid construction ────────────────────────────────────
    def _build_grid(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r, c in self.STATIC_BLOCKED:
            self.grid[r][c] = BLOCKED
        for village_id, (r, c) in self.VILLAGE_POSITIONS.items():
            self.grid[r][c] = VILLAGE
        for zone, (r, c) in self.WAREHOUSE_POSITIONS.items():
            self.grid[r][c] = WAREHOUSE

    # ── Reset ────────────────────────────────────────────────
    def reset(self) -> Dict:
        self._build_grid()
        self.agent_pos = list(self.AGENT_START)
        self.carried_crop = None
        self.delivered_crops = []
        self.current_time = 0
        self.total_reward = 0.0
        self.done = False
        self.disruptions_log = []
        self.active_blocked = set()
        self.breakdown_steps = 0
        self.rain_active = False

        # Generate 3 crops at random villages
        crop_types = ["tomato", "wheat", "corn"]
        random.shuffle(crop_types)
        zones = list(self.WAREHOUSE_POSITIONS.keys())
        self.crops = []
        for i, (ctype, village_id) in enumerate(zip(crop_types, [1, 2, 3])):
            spoilage = random.randint(12, 20) if ctype == "tomato" else random.randint(20, 35)
            self.crops.append({
                "id": i + 1,
                "type": ctype,
                "village_id": village_id,
                "position": list(self.VILLAGE_POSITIONS[village_id]),
                "spoilage_remaining": spoilage,
                "intended_zone": random.choice(zones),
                "picked_up": False,
                "delivered": False,
            })

        # Schedule disruption events at random future steps
        self.disruption_schedule = sorted(
            random.sample(range(5, self.MAX_STEPS - 5), self.DISRUPTIONS_PER_EPISODE)
        )

        return self._get_observation("Episode started. Agent at center.")

    # ── Step ─────────────────────────────────────────────────
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            return self._get_observation("Episode already done."), 0.0, True, {}

        self.current_time += 1
        reward = 0.0
        result_msg = ""

        # Tick disruptions
        self._tick_disruptions()

        # Tick spoilage on all un-delivered crops each step
        for crop in self.crops:
            if not crop["delivered"]:
                crop["spoilage_remaining"] = max(0, crop["spoilage_remaining"] - 1)

        # ── Movement actions ──
        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            if self.breakdown_steps > 0:
                self.breakdown_steps -= 1
                reward = -1.0
                result_msg = f"Breakdown! Cannot move. {self.breakdown_steps} steps remaining."
            else:
                moved, msg = self._try_move(action)
                reward = 0.0 if moved else -1.0   # penalty for hitting wall
                result_msg = msg

        # ── Pickup ──
        elif action == ACTION_PICKUP:
            reward, result_msg = self._try_pickup()

        # ── Dropoff ──
        elif action == ACTION_DROPOFF:
            reward, result_msg = self._try_dropoff()

        else:
            reward = -0.5
            result_msg = f"Unknown action: {action}"

        self.total_reward += reward

        # Check done conditions
        all_delivered = all(c["delivered"] for c in self.crops)
        if all_delivered or self.current_time >= self.MAX_STEPS:
            self.done = True
            result_msg += " | Episode finished."

        obs = self._get_observation(result_msg)
        return obs, reward, self.done, {"disruptions": self.disruptions_log}

    # ── Movement helper ──────────────────────────────────────
    def _try_move(self, action: str) -> Tuple[bool, str]:
        r, c = self.agent_pos
        if action == ACTION_UP:    nr, nc = r - 1, c
        elif action == ACTION_DOWN:  nr, nc = r + 1, c
        elif action == ACTION_LEFT:  nr, nc = r, c - 1
        elif action == ACTION_RIGHT: nr, nc = r, c + 1

        # Bounds check
        if not (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE):
            return False, "Hit boundary wall."

        # Blocked tile check (static + dynamic disruptions)
        if self.grid[nr][nc] == BLOCKED or (nr, nc) in self.active_blocked:
            return False, f"Tile ({nr},{nc}) is blocked."

        self.agent_pos = [nr, nc]
        return True, f"Moved {action} to ({nr},{nc})."

    # ── Pickup helper ────────────────────────────────────────
    def _try_pickup(self) -> Tuple[float, str]:
        pos = tuple(self.agent_pos)
        if self.carried_crop is not None:
            return -0.5, "Already carrying a crop. Dropoff first."

        for crop in self.crops:
            if not crop["picked_up"] and not crop["delivered"]:
                if tuple(crop["position"]) == pos:
                    if crop["spoilage_remaining"] <= 0:
                        return -2.0, f"Crop {crop['id']} ({crop['type']}) is fully spoiled!"
                    crop["picked_up"] = True
                    self.carried_crop = crop
                    return +2.0, f"Picked up crop {crop['id']} ({crop['type']}) at {pos}."

        return -0.5, f"No crop available at {pos}."

    # ── Dropoff helper ───────────────────────────────────────
    def _try_dropoff(self) -> Tuple[float, str]:
        pos = tuple(self.agent_pos)
        if self.carried_crop is None:
            return -0.5, "Not carrying any crop."

        for zone, wpos in self.WAREHOUSE_POSITIONS.items():
            if tuple(wpos) == pos:
                crop = self.carried_crop
                crop["delivered"] = True
                crop["delivered_zone"] = zone
                self.delivered_crops.append(crop)
                self.carried_crop = None

                # Reward logic
                correct_zone = (zone == crop["intended_zone"])
                fresh = crop["spoilage_remaining"] > 0
                on_time = crop["spoilage_remaining"] > 5   # well within deadline

                if correct_zone and on_time:
                    reward = 10.0 + 5.0   # delivery + on-time bonus
                    msg = f"PERFECT delivery! Crop {crop['id']} → {zone} (fresh & on time). +15"
                elif correct_zone and fresh:
                    reward = 10.0
                    msg = f"Correct delivery! Crop {crop['id']} → {zone} (fresh). +10"
                elif correct_zone and not fresh:
                    reward = 5.0
                    msg = f"Correct zone but crop {crop['id']} spoiled. +5"
                else:
                    reward = -3.0
                    msg = f"WRONG ZONE! Crop {crop['id']} delivered to {zone}, needed {crop['intended_zone']}. -3"

                return reward, msg

        return -0.5, f"No warehouse at {pos}. Move to a warehouse first."

    # ── Disruption engine ────────────────────────────────────
    def _tick_disruptions(self):
        if self.disruption_schedule and self.current_time == self.disruption_schedule[0]:
            self.disruption_schedule.pop(0)
            dtype = random.choice(ALL_DISRUPTIONS)
            self._apply_disruption(dtype)

        # Clear road_block after 5 steps
        expired = [pos for pos in list(self.active_blocked)]
        # (we keep track via disruption log with expiry)
        for entry in self.disruptions_log:
            if entry.get("type") == DISRUPTION_ROAD_BLOCK:
                if self.current_time >= entry.get("expires_at", 9999):
                    pos = tuple(entry["blocked_tile"])
                    self.active_blocked.discard(pos)

        # Breakdown wears off naturally (handled in step)
        if self.rain_active and self.current_time % 8 == 0:
            self.rain_active = False

    def _apply_disruption(self, dtype: str):
        if dtype == DISRUPTION_ROAD_BLOCK:
            # Block a random road tile near the agent
            r, c = self.agent_pos
            candidates = []
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE
                            and self.grid[nr][nc] == ROAD
                            and [nr, nc] != self.agent_pos):
                        candidates.append((nr, nc))
            if candidates:
                tile = random.choice(candidates)
                self.active_blocked.add(tile)
                self.disruptions_log.append({
                    "step": self.current_time,
                    "type": DISRUPTION_ROAD_BLOCK,
                    "blocked_tile": list(tile),
                    "expires_at": self.current_time + 5,
                    "message": f"Road blocked at {tile} for 5 steps!",
                })

        elif dtype == DISRUPTION_RAIN:
            self.rain_active = True
            # Rain: extra spoilage tick on all crops this step
            for crop in self.crops:
                if not crop["delivered"]:
                    crop["spoilage_remaining"] = max(0, crop["spoilage_remaining"] - 2)
            self.disruptions_log.append({
                "step": self.current_time,
                "type": DISRUPTION_RAIN,
                "message": "Heavy rain! All crops aged by 2 extra steps.",
            })

        elif dtype == DISRUPTION_BREAKDOWN:
            self.breakdown_steps = 3
            self.disruptions_log.append({
                "step": self.current_time,
                "type": DISRUPTION_BREAKDOWN,
                "message": "Vehicle breakdown! Agent immobilised for 3 steps.",
            })

    # ── Observation builder ──────────────────────────────────
    def _get_observation(self, last_action_result: str = "") -> Dict:
        pending = [c["id"] for c in self.crops if not c["delivered"]]
        carried = None
        if self.carried_crop:
            carried = {
                "id": self.carried_crop["id"],
                "type": self.carried_crop["type"],
                "spoilage_remaining": self.carried_crop["spoilage_remaining"],
                "intended_zone": self.carried_crop["intended_zone"],
            }

        return {
            "current_time": self.current_time,
            "agent_position": list(self.agent_pos),
            "carried_crop": carried,
            "crops": [
                {
                    "id": c["id"],
                    "type": c["type"],
                    "position": c["position"],
                    "village_id": c["village_id"],
                    "spoilage_remaining": c["spoilage_remaining"],
                    "intended_zone": c["intended_zone"],
                    "picked_up": c["picked_up"],
                    "delivered": c["delivered"],
                }
                for c in self.crops
            ],
            "warehouses": {
                zone: list(pos) for zone, pos in self.WAREHOUSE_POSITIONS.items()
            },
            "active_disruptions": [
                e for e in self.disruptions_log
                if e.get("type") == DISRUPTION_ROAD_BLOCK
                and self.current_time < e.get("expires_at", 0)
            ],
            "rain_active": self.rain_active,
            "breakdown_steps_remaining": self.breakdown_steps,
            "pending_orders": pending,
            "reward": 0.0,
            "done": self.done,
            "last_action_result": last_action_result,
        }

    # ── Grid display (terminal) ──────────────────────────────
    def render(self) -> str:
        symbols = {ROAD: ".", VILLAGE: "V", WAREHOUSE: "W", BLOCKED: "#"}
        lines = []
        for r in range(self.GRID_SIZE):
            row = ""
            for c in range(self.GRID_SIZE):
                if [r, c] == self.agent_pos:
                    row += "A"
                elif (r, c) in self.active_blocked:
                    row += "X"
                else:
                    row += symbols.get(self.grid[r][c], "?")
                row += " "
            lines.append(row)
        return "\n".join(lines)