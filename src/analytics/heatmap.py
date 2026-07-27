import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from scipy.stats import gaussian_kde
from typing import Dict, List, Tuple, Optional, Any
from ..geometry.pitch_template import TacticalPitchTemplate

class PlayerHeatmapGenerator:
    """
    Generates 2D Pitch Spatial Density Heatmaps for key players from opposing teams.
    Maps player movement history onto FIFA 105m x 68m tactical pitch graphics.
    """
    def __init__(self, pitch_template: Optional[TacticalPitchTemplate] = None):
        self.pitch_template = pitch_template if pitch_template is not None else TacticalPitchTemplate()

    def generate_player_heatmap(
        self, 
        track_id: int, 
        pitch_positions: np.ndarray, 
        player_label: str = "Player",
        output_path: str = "data/output/player_heatmap.png"
    ) -> str:
        """
        Generates and saves a 2D density heatmap for a single player.
        pitch_positions: Nx2 array of (X_meters, Y_meters) pitch coordinates [0..105, 0..68].
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Draw base pitch canvas
        pitch_canvas = self.pitch_template.draw_pitch()
        h_canvas, w_canvas, _ = pitch_canvas.shape

        if len(pitch_positions) < 5:
            print(f"[WARNING] Not enough trajectory data for Track ID #{track_id} to generate heatmap.")
            return output_path

        # Convert meter positions to canvas pixel coordinates
        canvas_pts = self.pitch_template.meter_to_canvas(pitch_positions)
        x_pts = canvas_pts[:, 0]
        y_pts = canvas_pts[:, 1]

        # Create density grid canvas
        density_grid = np.zeros((h_canvas, w_canvas), dtype=np.float32)

        # 2D Gaussian Kernel Density Estimation / Accumulation
        for x, y in zip(x_pts, y_pts):
            if 0 <= x < w_canvas and 0 <= y < h_canvas:
                density_grid[y, x] += 1.0

        # Apply Gaussian Blur to smooth coordinate density
        kernel_size = (65, 65)
        blurred_density = cv2.GaussianBlur(density_grid, kernel_size, sigmaX=15, sigmaY=15)

        # Normalize density grid to [0..255]
        max_val = np.max(blurred_density)
        if max_val > 0:
            norm_density = (blurred_density / max_val * 255.0).astype(np.uint8)
        else:
            norm_density = np.zeros((h_canvas, w_canvas), dtype=np.uint8)

        # Apply JET / HOT colormap (Blue -> Cyan -> Green -> Yellow -> Red)
        heatmap_color = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)

        # Create alpha blend mask (where density > threshold)
        alpha = (norm_density.astype(np.float32) / 255.0) * 0.65
        alpha_3d = np.dstack([alpha, alpha, alpha])

        # Blend heatmap onto green tactical pitch canvas
        blended = (heatmap_color * alpha_3d + pitch_canvas * (1.0 - alpha_3d)).astype(np.uint8)

        # Re-draw crisp pitch line markings on top
        pitch_lines = self.pitch_template.draw_pitch()
        line_mask = (pitch_lines != (34, 139, 34)) # Non-grass pixels
        blended[line_mask] = pitch_lines[line_mask]

        # Draw Title Badge
        badge_title = f"SPATIAL HEATMAP | {player_label} #{track_id}"
        cv2.rectangle(blended, (20, 15), (20 + len(badge_title) * 11, 48), (0, 0, 0), -1)
        cv2.putText(blended, badge_title, (28, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(output_path, blended)
        print(f"[SUCCESS] Spatial Heatmap saved to: {output_path}")
        return output_path

    def auto_generate_key_player_heatmaps(
        self, 
        metrics_calc: Any, 
        team_assigner: Any,
        output_dir: str = "data/output"
    ) -> List[str]:
        """
        Identifies the most active player from Team 1 and Team 2, generating heatmaps for both.
        """
        output_paths = []
        team1_candidates = []
        team2_candidates = []

        for track_id, hist in metrics_calc.history.items():
            if len(hist) < 10:
                continue

            role_info = team_assigner.get_player_role_info(track_id)
            role = role_info["role"]

            if role == "TEAM_1":
                team1_candidates.append((track_id, len(hist)))
            elif role == "TEAM_2":
                team2_candidates.append((track_id, len(hist)))

        # Sort candidates by frame count
        team1_candidates.sort(key=lambda x: x[1], reverse=True)
        team2_candidates.sort(key=lambda x: x[1], reverse=True)

        # Generate Team 1 Key Player Heatmap
        if team1_candidates:
            key_t1_id = team1_candidates[0][0]
            pts_t1 = metrics_calc.get_trajectory_history(key_t1_id, max_points=10000)
            out_t1 = os.path.join(output_dir, f"heatmap_team1_player_{key_t1_id}.png")
            self.generate_player_heatmap(key_t1_id, pts_t1, player_label="TEAM 1", output_path=out_t1)
            output_paths.append(out_t1)

        # Generate Team 2 Key Player Heatmap
        if team2_candidates:
            key_t2_id = team2_candidates[0][0]
            pts_t2 = metrics_calc.get_trajectory_history(key_t2_id, max_points=10000)
            out_t2 = os.path.join(output_dir, f"heatmap_team2_player_{key_t2_id}.png")
            self.generate_player_heatmap(key_t2_id, pts_t2, player_label="TEAM 2", output_path=out_t2)
            output_paths.append(out_t2)

        return output_paths
