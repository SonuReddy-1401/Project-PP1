"""
Data Processor & Tactical Metrics Engine for Unified Pipeline
"""
import math
import json

def extract_positions(frame):
    """
    Helper to safely extract [x, y] position array from frame dict.
    """
    if not isinstance(frame, dict):
        return []
    return frame.get("target_positions") or frame.get("players") or []

def process_dataset(raw_data, fps=25.0, clip_start_offset_sec=0.0, attacking_direction="left_to_right"):
    """
    Processes raw dataset frames, applying physical outlier filtering (< 10 m/s ceiling),
    calculating pitch thirds occupancy, team width/depth/area metrics, and centroid trajectory.
    """
    processed_frames = []
    total_frames = len(raw_data)
    
    total_samples = 0
    thirds_count = {"defensive": 0, "middle": 0, "attacking": 0}
    
    width_list = []
    depth_list = []
    area_list = []
    
    prev_centroid = None
    safe_centroid_dist_m = 0.0
    rejected_intervals = 0
    total_intervals = 0
    
    for idx, frame in enumerate(raw_data):
        positions = extract_positions(frame)
        frame_idx = frame.get("frame_idx", idx) if isinstance(frame, dict) else idx
        
        if not positions:
            continue
            
        sum_x, sum_y = 0.0, 0.0
        min_x, max_x = 999.0, -999.0
        min_y, max_y = 999.0, -999.0
        
        for p in positions:
            px = p[0] if isinstance(p, (list, tuple)) else p.get("x", 0.0)
            py = p[1] if isinstance(p, (list, tuple)) else p.get("y", 0.0)
            
            sum_x += px
            sum_y += py
            if px < min_x: min_x = px
            if px > max_x: max_x = px
            if py < min_y: min_y = py
            if py > max_y: max_y = py
            
            total_samples += 1
            
        count = len(positions)
        cx = sum_x / count
        cy = sum_y / count
        
        # Pitch Thirds classification based on attacking direction
        # For left_to_right: cx < -17.5 = Defensive, cx > 17.5 = Attacking
        # For right_to_left: cx > 17.5 = Defensive, cx < -17.5 = Attacking (FLIPPED)
        if attacking_direction == "right_to_left":
            if cx > 17.5:
                thirds_count["defensive"] += 1
            elif cx < -17.5:
                thirds_count["attacking"] += 1
            else:
                thirds_count["middle"] += 1
        else:
            if cx < -17.5:
                thirds_count["defensive"] += 1
            elif cx > 17.5:
                thirds_count["attacking"] += 1
            else:
                thirds_count["middle"] += 1
            
        w = max(1.0, max_y - min_y)
        d = max(1.0, max_x - min_x)
        
        width_list.append(w)
        depth_list.append(d)
        area_list.append(w * d)
        
        # Centroid Trajectory & Speed Ceiling Filter (< 10 m/s = 0.4m per frame at 25fps)
        if prev_centroid is not None:
            total_intervals += 1
            step_m = math.sqrt((cx - prev_centroid[0])**2 + (cy - prev_centroid[1])**2)
            if step_m < 0.4: # Under physical speed ceiling threshold
                safe_centroid_dist_m += step_m
            else:
                rejected_intervals += 1
                
        prev_centroid = (cx, cy)
        
        processed_frames.append({
            "frame_idx": frame_idx,
            "timestamp_sec": idx / fps,
            "target_positions": positions,
            "centroid": [round(cx, 2), round(cy, 2)],
            "width": round(w, 2),
            "depth": round(d, 2),
            "area": round(w * d, 1)
        })

    # Summary Statistics
    avg_width = round(sum(width_list) / max(1, len(width_list)), 1)
    avg_depth = round(sum(depth_list) / max(1, len(depth_list)), 1)
    avg_area = round(sum(area_list) / max(1, len(area_list)), 1)
    
    total_thirds = max(1, sum(thirds_count.values()))
    def_pct = round((thirds_count["defensive"] / total_thirds) * 100, 1)
    mid_pct = round((thirds_count["middle"] / total_thirds) * 100, 1)
    att_pct = round((thirds_count["attacking"] / total_thirds) * 100, 1)
    
    rejection_rate = round((rejected_intervals / max(1, total_intervals)) * 100, 1)
    duration_sec = total_frames / fps
    avg_pace_kmh = round(((safe_centroid_dist_m / max(1, duration_sec)) * 3.6), 2)
    
    stats = {
        "total_frames": total_frames,
        "total_samples": total_samples,
        "duration_sec": duration_sec,
        "avg_width_m": avg_width,
        "avg_depth_m": avg_depth,
        "avg_area_m2": avg_area,
        "safe_centroid_dist_km": round(safe_centroid_dist_m / 1000.0, 2),
        "avg_pace_kmh": avg_pace_kmh,
        "rejection_rate_pct": rejection_rate,
        "thirds_pct": {
            "defensive": def_pct,
            "middle": mid_pct,
            "attacking": att_pct
        }
    }
    
    return processed_frames, stats
