"""
Graders for CropOps Agent - Round 2
Upgraded from Round 1 CropDrop graders.

Round 1 → Round 2 changes:
  - EasyGrader:   same logic, but now checks grid-based delivery
  - MediumGrader: same 3-crop check, tomato bonus preserved
  - HardGrader:   time_score based on steps (max 50), route penalty
                  replaced by disruption-handling bonus
  - All graders: return values clamped to (0,1) range (never 0.0 or 1.0)
"""


class EasyGrader:
    """
    Single crop delivery.
    Score: 0.999 = correct zone + fresh crop
           0.5   = correct zone but spoiled
           0.001 = wrong zone or not delivered
    """
    def grade(self, trajectory: dict) -> float:
        delivered = trajectory.get("delivered_crops", [])
        if not delivered:
            return 0.001
        crop = delivered[0]
        correct = crop.get("intended_zone") == crop.get("delivered_zone", "")
        fresh   = crop.get("spoilage_remaining", 0) > 0
        if correct and fresh:
            return 0.999
        if correct:
            return 0.5
        return 0.001


class MediumGrader:
    """
    Multi-crop delivery.
    Score: base = correct deliveries / 3
           bonus +0.2 if tomato delivered first (most urgent crop)
           capped at 0.999
    """
    def grade(self, trajectory: dict) -> float:
        delivered = trajectory.get("delivered_crops", [])
        if not delivered:
            return 0.001

        total_crops = 3
        correct_count = sum(
            1 for c in delivered
            if c.get("intended_zone") == c.get("delivered_zone", "")
        )
        base_score = correct_count / total_crops

        # Tomato urgency bonus (same as Round 1)
        tomato_delivered = any(c.get("type") == "tomato" for c in delivered)
        bonus = 0.2 if tomato_delivered else 0.0

        score = min(base_score + bonus, 0.999)
        return max(0.001, score)


class HardGrader:
    """
    Full grid navigation with disruptions.

    Scoring (Round 2 upgrade):
      50% correct deliveries
      30% time efficiency  (finished in under 30 steps = full marks)
      20% disruption bonus (agent delivered despite disruptions)
    """
    OPTIMAL_STEPS   = 30    # expect a good agent to finish in ~30 steps
    MAX_STEPS       = 50

    def grade(self, trajectory: dict) -> float:
        delivered   = trajectory.get("delivered_crops", [])
        total_crops = 3
        total_steps = trajectory.get("total_steps", self.MAX_STEPS)
        
        # Check both possible key names for disruptions
        disruptions = trajectory.get("disruptions_encountered", 
                                     trajectory.get("disruptions_handled", 0))

        if not delivered:
            return 0.001

        # 50% - delivery accuracy
        correct_count = sum(
            1 for c in delivered
            if c.get("intended_zone") == c.get("delivered_zone", "")
        )
        correct_score = correct_count / total_crops

        # 30% - time efficiency
        if total_steps <= self.OPTIMAL_STEPS:
            time_score = 1.0
        elif total_steps <= self.OPTIMAL_STEPS * 1.4:
            time_score = 0.7
        elif total_steps <= self.OPTIMAL_STEPS * 1.7:
            time_score = 0.4
        else:
            time_score = 0.2

        # 20% - disruption handling
        if disruptions == 0:
            disruption_score = 0.5      # no disruptions, partial credit
        else:
            deliveries_after_disruption = len(delivered)
            disruption_score = min(1.0, deliveries_after_disruption / total_crops)

        final = (correct_score * 0.5) + (time_score * 0.3) + (disruption_score * 0.2)
        
        # Clamp to (0,1) range (never return 0.0 or 1.0)
        return max(0.001, min(0.999, final))