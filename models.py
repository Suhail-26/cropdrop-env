from pydantic import BaseModel
from typing import List, Dict, Optional, Literal

# Action - What the AI can do
class CropdropAction(BaseModel):
    crop_id: int
    route: Literal["paved", "dirt", "muddy"]
    destination_zone: str

# Observation - What the AI sees
class CropdropObservation(BaseModel):
    current_time: int
    crops: List[Dict]
    routes_status: Dict[str, Dict]
    pending_orders: List[Dict]
    last_action_result: Optional[str] = None