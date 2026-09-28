"""
Narrative & JSON Exporter for Unified Pipeline
"""
import os
import json
import shutil
from .config import get_color_config, parse_match_clock

def generate_narratives(team_name, opponent_name, stats, match_start_clock):
    """
    Generates dynamic narrative callout summaries tailored to the specific team and match stats.
    """
    thirds = stats.get("thirds_pct", {})
    def_pct = thirds.get("defensive", 0.0)
    mid_pct = thirds.get("middle", 50.0)
    att_pct = thirds.get("attacking", 0.0)
    
    dom_third = "Middle Third"
    dom_val = mid_pct
    if def_pct > mid_pct and def_pct > att_pct:
        dom_third = "Defensive Third"
        dom_val = def_pct
    elif att_pct > mid_pct and att_pct > def_pct:
        dom_third = "Attacking Third"
        dom_val = att_pct
        
    return {
        "overview": {
            "summary": f"{team_name} controlled the match sequence primarily through the {dom_third} ({dom_val}% occupancy), maintaining an average tactical pitch width of {stats.get('avg_width_m')}m and depth of {stats.get('avg_depth_m')}m."
        },
        "heatmap": {
            "summary": f"{team_name} established spatial occupancy density across pitch coordinates with an average squad area envelope of {stats.get('avg_area_m2')} m²."
        },
        "shape": {
            "summary": f"{team_name}'s pitch width averaged {stats.get('avg_width_m')}m while depth averaged {stats.get('avg_depth_m')}m. Structural compactness and area expansion fluctuated between {stats.get('avg_area_m2')} m²."
        },
        "thirds": {
            "summary": f"{team_name} recorded {mid_pct}% occupancy in the Middle Third, {def_pct}% in the Defensive Third, and {att_pct}% in the Attacking Third."
        },
        "trajectory": {
            "summary": f"Team centroid displacement total is {stats.get('safe_centroid_dist_km')} km ({stats.get('avg_pace_kmh')} km/h average pace) using physical outlier rejection (< 10 m/s ceiling)."
        },
        "players": {
            "summary": "Individual tracking provides continuous mobility snapshots and implied pace for active squad tracking segments."
        },
        "methodology": {
            "summary": f"Distance metrics reject impossible frame jumps using 10 m/s threshold. Interval rejection rate is {stats.get('rejection_rate_pct')}%, correlating to rapid broadcast camera panning."
        }
    }

def export_pipeline_assets(dashboard_dir, processed_frames, stats, team_name, opponent_name, team_color, match_start_clock, fps):
    """
    Exports positions dataset, metadata, narratives, and color theme variables to dashboard data directory.
    """
    data_dir = os.path.join(dashboard_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    color_cfg = get_color_config(team_color)
    clip_start_offset_sec = parse_match_clock(match_start_clock)
    
    # 1. Dataset JSON
    dataset_path = os.path.join(data_dir, "positions_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(processed_frames, f, indent=2)
        
    # 2. Tracked Dataset JSON (Fallback mirror for roster analytics)
    tracked_path = os.path.join(data_dir, "positions_dataset_tracked.json")
    with open(tracked_path, "w", encoding="utf-8") as f:
        json.dump(processed_frames, f, indent=2)
        
    # 3. Metadata JSON
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata = {
        "team_name": team_name,
        "opponent_name": opponent_name,
        "team_color": color_cfg["primary"],
        "team_color_secondary": color_cfg["secondary"],
        "team_badge": color_cfg["badge"],
        "match_start_clock": match_start_clock,
        "clip_start_offset_sec": clip_start_offset_sec,
        "fps": fps,
        "total_frames": stats["total_frames"],
        "duration_sec": stats["duration_sec"],
        "stats": stats
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # 4. Narratives JSON
    narratives = generate_narratives(team_name, opponent_name, stats, match_start_clock)
    narratives_path = os.path.join(data_dir, "narratives.json")
    with open(narratives_path, "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=2)
        
    return dataset_path, metadata_path, narratives_path
