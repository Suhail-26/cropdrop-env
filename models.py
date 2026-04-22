"""
Pydantic models for CropOps Agent (Round 2).
Replaces Round 1 CropdropAction/CropdropObservation with grid-based models.

Round 1 → Round 2 changes:
  - CropdropAction (crop_id + route + destination_zone)
    → CropOpsAction (action: up/down/left/right/pickup/dropoff)
  - CropdropObservation (crops + routes_status + pending_orders)
    → CropOpsObservation (agent_position + grid_info + disruptions + carried_crop)
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ── Action model ─────────────────────────────────────────────
class ActionType(str, Enum):
    up       = "up"
    down     = "down"
    left     = "left"
    right    = "right"
    pickup   = "pickup"
    dropoff  = "dropoff"


class CropOpsAction(BaseModel):
    """Action model for the CropOps Agent environment (Round 2)."""
    action: ActionType = Field(
        ...,
        description="Movement: up/down/left/right. Interaction: pickup/dropoff"
    )

    class Config:
        use_enum_values = True


# ── Sub-models ────────────────────────────────────────────────
class CropState(BaseModel):
    id: int
    type: str
    position: List[int]              # [row, col] on the grid
    village_id: int
    spoilage_remaining: int
    intended_zone: str               # "zone_1" or "zone_2"
    picked_up: bool
    delivered: bool


class DisruptionEvent(BaseModel):
    step: int
    type: str                        # road_block | rain | vehicle_breakdown
    message: str
    blocked_tile: Optional[List[int]] = None
    expires_at: Optional[int] = None


# ── Observation model ─────────────────────────────────────────
class CropOpsObservation(BaseModel):
    """Observation returned after each environment step (Round 2)."""

    current_time: int = Field(0, description="Steps elapsed in the episode")

    # Agent spatial state
    agent_position: List[int] = Field(
        default_factory=lambda: [5, 5],
        description="[row, col] of the agent on the 10x10 grid"
    )
    carried_crop: Optional[Dict[str, Any]] = Field(
        None,
        description="Crop the agent is currently holding, or null"
    )

    # World state
    crops: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All crops: position, spoilage, intended zone, delivery status"
    )
    warehouses: Dict[str, Any] = Field(
        default_factory=dict,
        description="zone_1 and zone_2 warehouse positions on the grid"
    )

    # Disruptions
    active_disruptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active road blocks or events"
    )
    rain_active: bool = Field(False, description="Whether rain is affecting spoilage")
    breakdown_steps_remaining: int = Field(
        0, description="Steps until vehicle breakdown is resolved"
    )

    # Standard fields
    pending_orders: List[int] = Field(
        default_factory=list,
        description="Crop IDs not yet delivered"
    )
    reward: float = Field(0.0, description="Immediate reward from the last step")
    done: bool = Field(False, description="Whether the episode is finished")
    last_action_result: Optional[str] = Field(
        None, description="Human-readable result of the last action"
    )