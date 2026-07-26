import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any

class TeamAssigner:
    """
    Advanced Team & Entity Classifier for Broadcast Football Clips.
    Uses CIELAB + HSV perceptual color space, dual-crop (Jersey + Shorts) feature extraction,
    multi-frame accumulation, and specialized Referee (REF) & Goalkeeper (GK) detection.
    """
    # Role Constants
    ROLE_TEAM_1 = "TEAM_1"
    ROLE_TEAM_2 = "TEAM_2"
    ROLE_REFEREE = "REFEREE"
    ROLE_GOALKEEPER = "GOALKEEPER"

    def __init__(self):
        self.track_features: Dict[int, List[np.ndarray]] = {}
        self.track_bgr_colors: Dict[int, np.ndarray] = {}
        self.track_roles: Dict[int, str] = {}
        self.track_team_ids: Dict[int, int] = {}
        
        self.team_colors_bgr: Dict[int, Tuple[int, int, int]] = {
            1: (255, 69, 0),     # Team 1 Color (Blue/Orange)
            2: (0, 215, 255),    # Team 2 Color (Yellow/Gold)
            0: (0, 255, 255),    # Referee Color (Bright Yellow)
            3: (147, 20, 255)    # Goalkeeper Color (Deep Pink/Neon)
        }
        
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False

    def _extract_crop_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts CIELAB & HSV perceptual color features from both Torso (Jersey) and Shorts regions.
        Filters out background grass green.
        Returns: (12-D feature vector, BGR average color)
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

        lab_t, hsv_t, bgr_t = process_region(torso_crop)
        lab_s, hsv_s, bgr_s = process_region(shorts_crop)

        # 12-dimensional feature vector combining Jersey & Shorts perceptual colors
        feature_vec = np.concatenate([lab_t, hsv_t, lab_s, hsv_s])
        avg_bgr = 0.7 * bgr_t + 0.3 * bgr_s

        return feature_vec, avg_bgr

    def update(self, frame: np.ndarray, tracked_detections: Any, pitch_positions: Dict[int, np.ndarray] = None):
        """
        Accumulates player crop features frame-by-frame and fits/updates multi-team clustering.
        """
        if tracked_detections.tracker_id is None:
            return

        for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            tid = int(track_id)
            extracted = self._extract_crop_features(frame, bbox)
            if extracted is None:
                continue

            feat, bgr = extracted
            if tid not in self.track_features:
                self.track_features[tid] = []
            
            # Limit history to 30 feature samples per player to stay fast and adaptive
            if len(self.track_features[tid]) < 30:
                self.track_features[tid].append(feat)
            self.track_bgr_colors[tid] = bgr

        # Re-fit clustering once we have at least 6 distinct tracked entities or 20 total samples
        if len(self.track_features) >= 4:
            self._fit_clustering(pitch_positions)

    def _fit_clustering(self, pitch_positions: Dict[int, np.ndarray] = None):
        """
        Fits K-Means / GMM clustering on accumulated player CIELAB+HSV features.
        Identifies Referees and Goalkeepers.
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
        
        # 1. Normalize feature matrix for equal weight between CIELAB (L, a, b) & HSV (H, S, V)
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-5)

        # 2. Fit 2 primary team clusters first on CIELAB features
        kmeans_teams = KMeans(n_clusters=2, n_init=10, random_state=42)
        team_labels = kmeans_teams.fit_predict(X_norm[:, [0, 1, 2, 6, 7, 8]]) # Use CIELAB channels

        # 3. Detect Referees & Goalkeepers via feature distances & pitch position
        cluster_0_indices = np.where(team_labels == 0)[0]
        cluster_1_indices = np.where(team_labels == 1)[0]

        # Calculate cluster centroids
        c0_center = np.mean(X_norm[cluster_0_indices], axis=0)
        c1_center = np.mean(X_norm[cluster_1_indices], axis=0)

        for idx, tid in enumerate(track_ids):
            feat = X_norm[idx]
            dist0 = np.linalg.norm(feat - c0_center)
            dist1 = np.linalg.norm(feat - c1_center)
            
            # Check pitch position for Goalkeeper heuristic (near goal line X < 14m or X > 91m)
            pos = pitch_positions.get(tid) if pitch_positions else None
            is_near_goal = False
            if pos is not None:
                is_near_goal = (pos[0] <= 14.0 or pos[0] >= 91.0) and (18.0 <= pos[1] <= 50.0)

            # Check if entity is outlier (Referee or Goalkeeper)
            min_dist = min(dist0, dist1)
            
            if is_near_goal and min_dist > 2.2:
                self.track_roles[tid] = self.ROLE_GOALKEEPER
                self.track_team_ids[tid] = 3
            elif min_dist > 3.5: # Outlier feature distance -> Referee
                self.track_roles[tid] = self.ROLE_REFEREE
                self.track_team_ids[tid] = 0
            else:
                assigned_team = team_labels[idx] + 1
                self.track_roles[tid] = self.ROLE_TEAM_1 if assigned_team == 1 else self.ROLE_TEAM_2
                self.track_team_ids[tid] = assigned_team

        self.is_fitted = True

    def get_player_team(self, frame: np.ndarray, bbox: np.ndarray, track_id: int) -> int:
        """
        Returns numeric team ID for backwards compatibility:
          1: Team 1
          2: Team 2
          0: Referee
          3: Goalkeeper
        """
        tid = int(track_id)
        if tid in self.track_team_ids:
            return self.track_team_ids[tid]
        return 1

    def get_player_role_info(self, track_id: int) -> Dict[str, Any]:
        """
        Returns full classification metadata dict for visualization:
        {"role": str, "team_id": int, "color_bgr": tuple}
        """
        tid = int(track_id)
        team_id = self.track_team_ids.get(tid, 1)
        role = self.track_roles.get(tid, self.ROLE_TEAM_1)
        color = self.team_colors_bgr.get(team_id, (255, 255, 255))
        
        return {
            "role": role,
            "team_id": team_id,
            "color_bgr": color
        }
