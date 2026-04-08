class EasyGrader:
    def grade(self, trajectory):
        if not trajectory.get("delivered_crops"):
            return 0.0
        crop = trajectory["delivered_crops"][0]
        if crop["intended_zone"] == crop.get("delivered_zone", ""):
            return 1.0 if crop.get("spoilage_remaining", 0) > 0 else 0.5
        return 0.0

class MediumGrader:
    def grade(self, trajectory):
        delivered = trajectory.get("delivered_crops", [])
        if not delivered:
            return 0.0
        correct = sum(1 for c in delivered if c["intended_zone"] == c.get("delivered_zone", ""))
        base = correct / 3.0
        tomato_first = any(c.get("type") == "tomato" for c in delivered)
        return min(base + (0.2 if tomato_first else 0), 1.0)

class HardGrader:
    def grade(self, trajectory):
        delivered = trajectory.get("delivered_crops", [])
        if not delivered:
            return 0.0
        correct = sum(1 for c in delivered if c["intended_zone"] == c.get("delivered_zone", ""))
        correct_score = correct / 3.0
        total_time = trajectory.get("total_time", 0)
        optimal = 12
        if total_time <= optimal:
            time_score = 1.0
        elif total_time <= optimal * 1.5:
            time_score = 0.7
        elif total_time <= optimal * 2:
            time_score = 0.4
        else:
            time_score = 0.2
        muddy = trajectory.get("muddy_route_count", 0)
        penalty = min(0.3, muddy * 0.1)
        final = correct_score * 0.5 + time_score * 0.3 - penalty + 0.2
        return max(0.0, min(1.0, final))
