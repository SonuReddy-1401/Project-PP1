import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Any

class KinematicMetricsCalculator:
    """
    Computes real-world kinematic metrics (distance, speed, acceleration, trajectories)
    from pitch meter coordinates.
    """
    def __init__(self, fps: float = 30.0, smoothing_window: int = 7):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.smoothing_window = smoothing_window
        
        # Track history: track_id -> List of (frame_idx, x_meters, y_meters)
        self.history: Dict[int, List[Tuple[int, float, float]]] = {}
        # Calculated stats: track_id -> dict
        self.total_distance: Dict[int, float] = {}
        self.current_speed: Dict[int, float] = {}  # in km/h
        self.max_speed: Dict[int, float] = {}      # in km/h

    def update_positions(self, frame_idx: int, pitch_positions: Dict[int, np.ndarray]):
        """
        Updates tracking history for current frame.
        pitch_positions: dict of track_id -> np.array([x_m, y_m])
        """
        for track_id, pos in pitch_positions.items():
            if track_id not in self.history:
                self.history[track_id] = []
                self.total_distance[track_id] = 0.0
                self.current_speed[track_id] = 0.0
                self.max_speed[track_id] = 0.0
                
            self.history[track_id].append((frame_idx, float(pos[0]), float(pos[1])))
            
            # Update distance and speed if we have at least 2 points
            if len(self.history[track_id]) >= 2:
                _, x1, y1 = self.history[track_id][-2]
                _, x2, y2 = self.history[track_id][-1]
                
                step_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Filter out absurd telemetry jumps (e.g. tracking swaps > 15 m/s)
                if step_dist < 15.0 * self.dt:
                    self.total_distance[track_id] += step_dist
                    
                    # Instantaneous speed (m/s -> km/h)
                    instant_speed_kmh = (step_dist / self.dt) * 3.6
                    
                    # Exponential moving average smoothing for live display
                    prev_speed = self.current_speed[track_id]
                    smoothed_speed = 0.7 * prev_speed + 0.3 * instant_speed_kmh if prev_speed > 0 else instant_speed_kmh
                    
                    self.current_speed[track_id] = round(smoothed_speed, 1)
                    if smoothed_speed > self.max_speed[track_id] and smoothed_speed < 40.0:  # Human max speed cap
                        self.max_speed[track_id] = round(smoothed_speed, 1)

    def get_trajectory_history(self, track_id: int, max_points: int = 30) -> np.ndarray:
        """
        Returns recent (X, Y) pitch meter trajectory history for visualization.
        """
        if track_id not in self.history or len(self.history[track_id]) == 0:
            return np.empty((0, 2))
            
        pts = np.array([[x, y] for _, x, y in self.history[track_id][-max_points:]])
        return pts

    def get_player_stats(self, track_id: int) -> Dict[str, Any]:
        """
        Retrieves summary statistics for a specific player track ID.
        """
        return {
            "track_id": track_id,
            "total_distance_m": round(self.total_distance.get(track_id, 0.0), 2),
            "current_speed_kmh": self.current_speed.get(track_id, 0.0),
            "max_speed_kmh": self.max_speed.get(track_id, 0.0)
        }

    def export_summary_dataframe(self) -> pd.DataFrame:
        """
        Exports summary metrics for all tracked entities as a pandas DataFrame.
        """
        rows = []
        for track_id in self.history.keys():
            rows.append(self.get_player_stats(track_id))
        if not rows:
            return pd.DataFrame(columns=["track_id", "total_distance_m", "current_speed_kmh", "max_speed_kmh"])
        return pd.DataFrame(rows)
