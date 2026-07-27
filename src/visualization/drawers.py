import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Any, Optional
from ..geometry.pitch_template import TacticalPitchTemplate

class PitchDrawer:
    """
    Renders player tracking markers, speed tags, trailing paths,
    and standalone 2D top-down tactical pitch overlays.
    Supports single-team filtering (e.g. target_team=1) to focus on one team.
    """
    def __init__(self, pitch_template: Optional[TacticalPitchTemplate] = None):
        self.pitch_template = pitch_template if pitch_template is not None else TacticalPitchTemplate()
        self.team_colors_bgr = {
            1: (255, 69, 0),     # Team 1 (Bright Cyan/Orange)
            2: (0, 215, 255),    # Team 2 (Yellow)
            0: (0, 255, 255),    # Referee (Neon Yellow)
            3: (147, 20, 255)    # Goalkeeper (Neon Pink)
        }

    def draw_player_ellipses(self, frame: np.ndarray, tracked_detections: sv.Detections, 
                             team_assigner: Any = None, metrics_calc: Any = None,
                             target_team: Optional[int] = 1) -> np.ndarray:
        """
        Draws ground ellipses under target team players' feet with track ID and speed (km/h) badge.
        Filters out non-target team players if target_team is specified.
        """
        annotated = frame.copy()
        if tracked_detections.tracker_id is None:
            return annotated

        for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            x1, y1, x2, y2 = bbox
            x_center = int((x1 + x2) / 2.0)
            y_bottom = int(y2)
            box_width = int(x2 - x1)
            tid = int(track_id)
            
            # Retrieve player role metadata
            role_info = {"role": "TEAM_1", "team_id": 1, "color_bgr": (255, 69, 0)}
            if team_assigner is not None:
                role_info = team_assigner.get_player_role_info(tid)
                
            team_id = role_info["team_id"]
            
            # Single Team Filter: Only render target team players
            if target_team is not None and team_id != target_team:
                continue

            color = role_info["color_bgr"]
            role = role_info["role"]
            
            # Ground ellipse under foot
            axes = (int(box_width * 0.45), int(box_width * 0.18))
            cv2.ellipse(annotated, (x_center, y_bottom), axes, 0, 0, 360, (0, 0, 0), 3)
            cv2.ellipse(annotated, (x_center, y_bottom), axes, 0, 0, 360, color, -1)
            
            # Speed & Player ID Badge
            speed_str = ""
            if metrics_calc is not None:
                speed_kmh = metrics_calc.current_speed.get(tid, 0.0)
                speed_str = f" {speed_kmh}km/h"
                
            if role == "REFEREE":
                badge_str = f"REF #{tid}"
            elif role == "GOALKEEPER":
                badge_str = f"GK #{tid}"
            else:
                badge_str = f"#{tid}{speed_str}"
                
            # Text badge background pill
            badge_w = 40 + len(badge_str) * 6
            cv2.rectangle(annotated, (x_center - 25, y_bottom + 4), (x_center - 25 + badge_w, y_bottom + 22), (0, 0, 0), -1)
            cv2.rectangle(annotated, (x_center - 25, y_bottom + 4), (x_center - 25 + badge_w, y_bottom + 22), color, 1)
            cv2.putText(annotated, badge_str, (x_center - 22, y_bottom + 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            
        return annotated

    def draw_standalone_tactical_pitch(self, 
                                        pitch_positions: Dict[int, np.ndarray],
                                        metrics_calc: Any = None,
                                        team_assigner: Any = None,
                                        homography: Any = None,
                                        target_team: Optional[int] = 1) -> np.ndarray:
        """
        Renders a dedicated 2D top-down tactical pitch frame (Output 2) with prominent Player IDs.
        Filters out non-target team players if target_team is specified.
        """
        pitch_bg = self.pitch_template.draw_pitch()
        
        for track_id, pos_meters in pitch_positions.items():
            tid = int(track_id)
            
            # Pitch Bounds Filter
            if homography is not None:
                if not homography.is_inside_pitch(pos_meters, margin=2.0):
                    continue

            canvas_pt = self.pitch_template.meter_to_canvas(pos_meters)
            
            role_info = {"role": "TEAM_1", "team_id": 1, "color_bgr": (255, 69, 0)}
            if team_assigner is not None:
                role_info = team_assigner.get_player_role_info(tid)

            team_id = role_info["team_id"]
            
            # Single Team Filter: Only render target team players
            if target_team is not None and team_id != target_team:
                continue

            color = role_info["color_bgr"]
            role = role_info["role"]
            
            # Trailing trajectory path
            if metrics_calc is not None:
                traj_meters = metrics_calc.get_trajectory_history(tid, max_points=30)
                if len(traj_meters) > 1:
                    traj_canvas = [self.pitch_template.meter_to_canvas(pt) for pt in traj_meters]
                    pts = np.array(traj_canvas, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(pitch_bg, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Prominent 2D Player Circle Marker with Bold Player ID Inside
            cv2.circle(pitch_bg, canvas_pt, 12, (0, 0, 0), -1)
            cv2.circle(pitch_bg, canvas_pt, 11, color, -1)
            cv2.circle(pitch_bg, canvas_pt, 12, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Bold Player ID Text inside dot
            id_text = str(tid)
            text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 2)[0]
            tx = canvas_pt[0] - text_size[0] // 2
            ty = canvas_pt[1] + text_size[1] // 2
            cv2.putText(pitch_bg, id_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Speed badge above player dot
            if metrics_calc is not None:
                speed_kmh = metrics_calc.current_speed.get(tid, 0.0)
                speed_str = f"{speed_kmh}km/h"
                sp_size = cv2.getTextSize(speed_str, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0]
                sx = canvas_pt[0] - sp_size[0] // 2
                sy = canvas_pt[1] - 16
                cv2.rectangle(pitch_bg, (sx - 3, sy - sp_size[1] - 2), (sx + sp_size[0] + 3, sy + 3), (0, 0, 0), -1)
                cv2.rectangle(pitch_bg, (sx - 3, sy - sp_size[1] - 2), (sx + sp_size[0] + 3, sy + 3), color, 1)
                cv2.putText(pitch_bg, speed_str, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

        return pitch_bg
