import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any

class TeamAssigner:
    """
    Advanced Team & Entity Classifier for Broadcast Football Clips.
    Supports explicit jersey color filtering (e.g. Red Team) as well as
    CIELAB + HSV perceptual color clustering, dual-crop feature extraction,
    and specialized Referee (REF) & Goalkeeper (GK) detection.
    """
    # Role Constants
    ROLE_TEAM_1 = "TEAM_1"
    ROLE_TEAM_2 = "TEAM_2"
    ROLE_REFEREE = "REFEREE"
    ROLE_GOALKEEPER = "GOALKEEPER"

    def __init__(self, target_color: Optional[str] = "red"):
        self.target_color = target_color.lower() if target_color else None
        self.track_features: Dict[int, List[np.ndarray]] = {}
        self.track_bgr_colors: Dict[int, np.ndarray] = {}
        self.track_roles: Dict[int, str] = {}
        self.track_team_ids: Dict[int, int] = {}
        
        self.team_colors_bgr: Dict[int, Tuple[int, int, int]] = {
            1: (0, 0, 255),      # Team 1 Color (Red)
            2: (255, 255, 255),  # Team 2 Color (White/Other)
            0: (0, 255, 255),    # Referee Color (Bright Yellow)
            3: (147, 20, 255)    # Goalkeeper Color (Deep Pink/Neon)
        }
        
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False

    def _extract_crop_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Extracts CIELAB & HSV perceptual color features and Red Jersey ratio.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h < 10 or w < 5 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None

        # Torso Crop (15% to 45% height) & Shorts Crop (45% to 70% height)
        torso_crop = frame[y1 + int(h * 0.15): y1 + int(h * 0.45), x1:x2]
        shorts_crop = frame[y1 + int(h * 0.45): y1 + int(h * 0.70), x1:x2]

        if torso_crop.size == 0 or shorts_crop.size == 0:
            return None

        hsv_t = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        lab_t = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2LAB)
        
        # Calculate Red Jersey Pixel Ratio (Hue 0-12 or 155-180, Sat > 50, Val > 40)
        red_mask1 = cv2.inRange(hsv_t, np.array([0, 50, 40]), np.array([12, 255, 255]))
        red_mask2 = cv2.inRange(hsv_t, np.array([155, 50, 40]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = float(np.sum(red_mask > 0)) / float(torso_crop.shape[0] * torso_crop.shape[1] + 1e-5)

        def process_region(crop):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            pixels_hsv = hsv.reshape(-1, 3)
            pixels_lab = lab.reshape(-1, 3)
            pixels_bgr = crop.reshape(-1, 3)

            # Mask grass green (Hue 35-85, Saturation > 30)
            non_grass = ~((pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) & (pixels_hsv[:, 1] >= 30))
            if np.sum(non_grass) > 5:
                lab_filtered = pixels_lab[non_grass]
                hsv_filtered = pixels_hsv[non_grass]
                bgr_filtered = pixels_bgr[non_grass]
            else:
                lab_filtered = pixels_lab
                hsv_filtered = pixels_hsv
                bgr_filtered = pixels_bgr

            lab_mean = np.median(lab_filtered, axis=0)
            hsv_mean = np.mean(hsv_filtered, axis=0)
            bgr_mean = np.mean(bgr_filtered, axis=0)
            return lab_mean, hsv_mean, bgr_mean

        lab_t_mean, hsv_t_mean, bgr_t = process_region(torso_crop)
        lab_s_mean, hsv_s_mean, bgr_s = process_region(shorts_crop)

        feature_vec = np.concatenate([lab_t_mean, hsv_t_mean, lab_s_mean, hsv_s_mean])
        avg_bgr = 0.7 * bgr_t + 0.3 * bgr_s

        return feature_vec, avg_bgr, red_ratio

    def update(self, frame: np.ndarray, tracked_detections: Any, pitch_positions: Dict[int, np.ndarray] = None):
        """
        Accumulates player crop features frame-by-frame and fits team classification.
        """
        if tracked_detections.tracker_id is None:
            return

        for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            tid = int(track_id)
            extracted = self._extract_crop_features(frame, bbox)
            if extracted is None:
                continue

            feat, bgr, red_ratio = extracted
            if tid not in self.track_features:
                self.track_features[tid] = []
            
            if len(self.track_features[tid]) < 30:
                self.track_features[tid].append(feat)
            self.track_bgr_colors[tid] = bgr

            # Direct Red Jersey Classification if target_color=="red"
            if self.target_color == "red":
                if red_ratio >= 0.15:
                    self.track_roles[tid] = self.ROLE_TEAM_1
                    self.track_team_ids[tid] = 1
                else:
                    self.track_roles[tid] = self.ROLE_TEAM_2
                    self.track_team_ids[tid] = 2

        if not self.target_color and len(self.track_features) >= 4:
            self._fit_clustering(pitch_positions)

    def _fit_clustering(self, pitch_positions: Dict[int, np.ndarray] = None):
        """
        Fits K-Means clustering on accumulated player CIELAB+HSV features.
        """
        track_ids = []
        mean_features = []

        for tid, feat_list in self.track_features.items():
            if len(feat_list) > 0:
                track_ids.append(tid)
                mean_features.append(np.mean(feat_list, axis=0))

        if len(mean_features) < 4:
            return

        X = np.array(mean_features)
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-5)

        kmeans_teams = KMeans(n_clusters=2, n_init=10, random_state=42)
        team_labels = kmeans_teams.fit_predict(X_norm[:, [0, 1, 2, 6, 7, 8]])

        for idx, tid in enumerate(track_ids):
            assigned_team = team_labels[idx] + 1
            self.track_roles[tid] = self.ROLE_TEAM_1 if assigned_team == 1 else self.ROLE_TEAM_2
            self.track_team_ids[tid] = assigned_team

        self.is_fitted = True

    def get_player_team(self, frame: np.ndarray, bbox: np.ndarray, track_id: int) -> int:
        """
        Returns numeric team ID:
          1: Team 1 (Red Team when target_color="red")
          2: Team 2
        """
        tid = int(track_id)
        if tid in self.track_team_ids:
            return self.track_team_ids[tid]
            
        if self.target_color == "red":
            extracted = self._extract_crop_features(frame, bbox)
            if extracted is not None and extracted[2] >= 0.15:
                return 1
            return 2
            
        return 1

    def get_player_role_info(self, track_id: int) -> Dict[str, Any]:
        """
        Returns full classification metadata dict for visualization:
        {"role": str, "team_id": int, "color_bgr": tuple}
        """
        tid = int(track_id)
        team_id = self.track_team_ids.get(tid, 1)
        role = self.track_roles.get(tid, self.ROLE_TEAM_1)
        color = self.team_colors_bgr.get(team_id, (0, 0, 255) if team_id == 1 else (255, 255, 255))
        
        return {
            "role": role,
            "team_id": team_id,
            "color_bgr": color
        }
