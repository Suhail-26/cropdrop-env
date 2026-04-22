# graders.py (root level) - Re-export class-based graders
import sys
import os

_server_path = os.path.join(os.path.dirname(__file__), "server")
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

try:
    from server.graders import EasyGrader, MediumGrader, HardGrader
except ImportError:
    # Fallback to local definitions
    class EasyGrader:
        def grade(self, trajectory):
            delivered = trajectory.get("crops_delivered", 0)
            if delivered == 0:
                return 0.001
            return min(0.999, delivered / 5.0)
    
    class MediumGrader:
        def grade(self, trajectory):
            delivered = trajectory.get("crops_delivered", 0)
            if delivered == 0:
                return 0.001
            base_score = delivered / 5.0
            disruptions = trajectory.get("disruptions_handled", 0)
            bonus = min(0.3, disruptions * 0.1)
            return min(0.999, base_score + bonus)
    
    class HardGrader:
        def grade(self, trajectory):
            delivered = trajectory.get("crops_delivered", 0)
            if delivered == 0:
                return 0.001
            delivery_score = (delivered / 5.0) * 0.5
            disruptions = trajectory.get("disruptions_handled", 0)
            disruption_score = min(0.2, disruptions * 0.05)
            total_steps = trajectory.get("total_steps", 60)
            efficiency_score = max(0, 0.3 * (1 - total_steps / 60))
            final = delivery_score + disruption_score + efficiency_score
            return max(0.001, min(0.999, final))

__all__ = ["EasyGrader", "MediumGrader", "HardGrader"]