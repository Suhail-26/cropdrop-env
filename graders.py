"""
Graders for CropDrop Environment
Easy, Medium, and Hard tasks
"""

class EasyGrader:
    """Single crop delivery - must deliver correctly before spoilage"""
    
    def grade(self, trajectory):
        if not trajectory.get("delivered_crops"):
            return 0.0
        
        crop = trajectory["delivered_crops"][0]
        if crop["intended_zone"] == crop.get("delivered_zone", ""):
            if crop.get("spoilage_remaining", 0) > 0:
                return 1.0
            return 0.5
        return 0.0


class MediumGrader:
    """Multi-crop delivery - deliver all 3 crops correctly"""
    
    def grade(self, trajectory):
        delivered = trajectory.get("delivered_crops", [])
        total_crops = 3
        
        if not delivered:
            return 0.0
        
        correct_count = 0
        for crop in delivered:
            if crop["intended_zone"] == crop.get("delivered_zone", ""):
                correct_count += 1
        
        base_score = correct_count / total_crops
        
        tomato_delivered = any(crop.get("type") == "tomato" for crop in delivered)
        bonus = 0.2 if tomato_delivered else 0
        
        return min(base_score + bonus, 1.0)


class HardGrader:
    """Route optimization - deliver all crops with optimal routing"""
    
    def grade(self, trajectory):
        delivered = trajectory.get("delivered_crops", [])
        total_crops = 3
        total_time = trajectory.get("total_time", 0)
        muddy_uses = trajectory.get("muddy_route_count", 0)
        
        if not delivered:
            return 0.0
        
        correct_count = sum(1 for crop in delivered 
                           if crop["intended_zone"] == crop.get("delivered_zone", ""))
        correct_score = correct_count / total_crops
        
        optimal_time = 12
        if total_time <= optimal_time:
            time_score = 1.0
        elif total_time <= optimal_time * 1.5:
            time_score = 0.7
        elif total_time <= optimal_time * 2:
            time_score = 0.4
        else:
            time_score = 0.2
        
        route_penalty = min(0.3, muddy_uses * 0.1)
        
        final_score = (correct_score * 0.5) + (time_score * 0.3) - route_penalty + 0.2
        
        return max(0.0, min(1.0, final_score))
