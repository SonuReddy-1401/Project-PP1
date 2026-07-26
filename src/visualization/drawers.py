import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Any, Optional
from ..geometry.pitch_template import TacticalPitchTemplate

class PitchDrawer:
    """
    Renders player tracking markers, speed tags, trailing paths,
    and side-by-side / split-screen top-down 2D tactical pitch overlays.
    Supports distinct Referee (REF), Goalkeeper (GK), and Team 1 vs Team 2 visual badges,
    while filtering out sideline staff from the tactical pitch map.
    """
    def __init__(self, pitch_template: Optional[TacticalPitchTemplate] = None):
        self.pitch_template = pitch_template if pitch_template is not None else TacticalPitchTemplate()
        # Default fallback team colors (BGR)
        self.team_colors_bgr = {
            1: (255, 69, 0),     # Team 1 Color (Blue/Orange)
            2: (0, 215, 255),    # Team 2 Color (Gold/Yellow)
            0: (0, 255, 255),    # Referee Color (Bright Neon)
            3: (147, 20, 255)    # Goalkeeper Color (Deep Pink/Neon)
        }

    def draw_player_ellipses(self, frame: np.ndarray, tracked_detections: sv.Detections, 
                             team_assigner: Any = None, metrics_calc: Any = None) -> np.ndarray:
        """
        Draws ground ellipses under players' feet with track ID, role badge, and speed (km/h).
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
            
            # Retrieve role metadata
            role_info = {"role": "TEAM_1", "team_id": 1, "color_bgr": (255, 69, 0)}
            if team_assigner is not None:
                role_info = team_assigner.get_player_role_info(tid)
                
            color = role_info["color_bgr"]
            role = role_info["role"]
            
            # Ground ellipse
            axes = (int(box_width * 0.45), int(box_width * 0.18))
            cv2.ellipse(annotated, (x_center, y_bottom), axes, 0, 0, 360, color, 2)
            cv2.ellipse(annotated, (x_center, y_bottom), axes, 0, 0, 360, color, -1)
            
            # Speed & Role Badge Text
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
                
            # Text badge background
            badge_w = 40 + len(badge_str) * 6
            cv2.rectangle(annotated, (x_center - 25, y_bottom + 4), (x_center - 25 + badge_w, y_bottom + 22), (0, 0, 0), -1)
            cv2.putText(annotated, badge_str, (x_center - 22, y_bottom + 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            
        return annotated

    def draw_tactical_pitch_overlay(self, frame: np.ndarray, 
                                   pitch_positions: Dict[int, np.ndarray],
                                   metrics_calc: Any = None,
                                   team_assigner: Any = None,
                                   homography: Any = None) -> np.ndarray:
        """
        Renders split-screen video: Broadcast view on Left, Top-Down 2D Tactical Pitch on Right.
        Filters out sideline staff from 2D tactical map.
        """
        pitch_bg = self.pitch_template.draw_pitch()
        
        # Plot player dots & trailing paths on 2D tactical pitch
        for track_id, pos_meters in pitch_positions.items():
            tid = int(track_id)
            
            # Boundary Filter: Skip sideline staff, coaches, substitutes standing outside field
            if homography is not None:
                if not homography.is_inside_pitch(pos_meters, margin=2.0):
                    continue
            else:
                x_m, y_m = pos_meters[0], pos_meters[1]
                if x_m < -2.0 or x_m > 107.0 or y_m < -2.0 or y_m > 70.0:
                    continue

            canvas_pt = self.pitch_template.meter_to_canvas(pos_meters)
            
            # Retrieve role metadata
            role_info = {"role": "TEAM_1", "team_id": 1, "color_bgr": (255, 69, 0)}
            if team_assigner is not None:
                role_info = team_assigner.get_player_role_info(tid)

            color = role_info["color_bgr"]
            role = role_info["role"]
            
            # Draw specialized tactical pitch markers
            if role == "REFEREE":
                # Diamond marker for Referee
                pts = np.array([
                    [canvas_pt[0], canvas_pt[1] - 8],
                    [canvas_pt[0] + 8, canvas_pt[1]],
                    [canvas_pt[0], canvas_pt[1] + 8],
                    [canvas_pt[0] - 8, canvas_pt[1]]
                ], np.int32)
                cv2.fillPoly(pitch_bg, [pts], (0, 255, 255))
                cv2.polylines(pitch_bg, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
                cv2.putText(pitch_bg, "R", (canvas_pt[0] - 3, canvas_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            elif role == "GOALKEEPER":
                # Square marker for Goalkeeper
                cv2.rectangle(pitch_bg, (canvas_pt[0] - 7, canvas_pt[1] - 7), (canvas_pt[0] + 7, canvas_pt[1] + 7), color, -1)
                cv2.rectangle(pitch_bg, (canvas_pt[0] - 7, canvas_pt[1] - 7), (canvas_pt[0] + 7, canvas_pt[1] + 7), (255, 255, 255), 1)
                cv2.putText(pitch_bg, "GK", (canvas_pt[0] - 6, canvas_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
            else:
                # Circle marker for Field Players
                cv2.circle(pitch_bg, canvas_pt, 7, color, -1)
                cv2.circle(pitch_bg, canvas_pt, 8, (0, 0, 0), 1)
                cv2.putText(pitch_bg, str(tid), (canvas_pt[0] - 4, canvas_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Trailing trajectory path
            if metrics_calc is not None:
                traj_meters = metrics_calc.get_trajectory_history(tid, max_points=20)
                if len(traj_meters) > 1:
                    traj_canvas = [self.pitch_template.meter_to_canvas(pt) for pt in traj_meters]
                    pts = np.array(traj_canvas, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(pitch_bg, [pts], isClosed=False, color=color, thickness=2)

        # Resize pitch_bg to match frame height for side-by-side view
        target_h = frame.shape[0]
        aspect = pitch_bg.shape[1] / pitch_bg.shape[0]
        target_w = int(target_h * aspect)
        resized_pitch = cv2.resize(pitch_bg, (target_w, target_h))
        
        # Combine broadcast frame and tactical pitch side-by-side
        split_screen = np.hstack((frame, resized_pitch))
        return split_screen
