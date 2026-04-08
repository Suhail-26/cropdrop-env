"""
Pydantic models for CropDrop environment actions and observations.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class RouteType(str, Enum):
    paved = "paved"
    dirt = "dirt"
    muddy = "muddy"


class ZoneType(str, Enum):
    zone_1 = "zone_1"
    zone_2 = "zone_2"


class CropdropAction(BaseModel):
    """Action model for the CropDrop environment."""

    crop_id: int = Field(..., description="Which crop to deliver (1, 2, or 3)")
    route: RouteType = Field(..., description="Road type: paved (fast), dirt (medium), muddy (slow)")
    destination_zone: ZoneType = Field(..., description="Delivery zone: zone_1 or zone_2")

    class Config:
        use_enum_values = True


class CropState(BaseModel):
    """State of a single crop."""

    id: int
    type: str
    spoilage_remaining: int
    intended_zone: str


class RouteStatus(BaseModel):
    """Status of a delivery route."""

    congestion: float
    estimated_time: int


class CropdropObservation(BaseModel):
    """Observation model returned after each environment step."""

    current_time: int = Field(0, description="Steps elapsed in the episode")
    crops: List[Dict[str, Any]] = Field(default_factory=list, description="Available crops")
    routes_status: Dict[str, Any] = Field(default_factory=dict, description="Route congestion info")
    pending_orders: List[int] = Field(default_factory=list, description="Crop IDs not yet delivered")
    reward: float = Field(0.0, description="Immediate reward from the last step")
    done: bool = Field(False, description="Whether the episode is finished")
    last_action_result: Optional[str] = Field(None, description="Result description of last action")
