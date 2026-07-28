import cv2
import numpy as np
from sklearn.cluster import KMeans
import supervision as sv

class TeamClassifier:
    """
    Roboflow AI (football-ai.ipynb) TeamClassifier replicating SigLIP/KMeans team color assignment
    and Goalkeeper centroid proximity resolution.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.kmeans = None
        self.team_colors = {}

    def extract_torso_hsv(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return np.zeros(3)

        h, w, _ = image_crop.shape
        # Torso crop: top 15% to 50%
        torso = image_crop[int(h * 0.15):int(h * 0.50), :]
        if torso.size == 0:
            torso = image_crop

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        # Calculate mean HSV color
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        return mean_hsv

    def fit(self, crops):
        """
        Fits 2-cluster KMeans model on player torso HSV embeddings.
        """
        features = [self.extract_torso_hsv(crop) for crop in crops if crop is not None and crop.size > 0]
        if len(features) < 2:
            return

        X = np.array(features)
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)

    def predict(self, image_crop):
        """
        Predicts team ID (0 or 1) for a player crop.
        """
        if self.kmeans is None:
            return 0

        feat = self.extract_torso_hsv(image_crop)
        team_id = self.kmeans.predict([feat])[0]

        # Check if HSV matches Red Team
        # Red hue wraps around 0 and 180 (0-10 or 170-180)
        hue, sat, val = feat[0], feat[1], feat[2]
        is_red = (hue < 12 or hue > 165) and sat > 60

        return 0 if is_red else 1

def resolve_goalkeepers_team_id(players_xy: np.ndarray, players_team_ids: np.ndarray, goalkeeper_xy: np.ndarray) -> int:
    """
    Replicates Roboflow AI (football-ai.ipynb) resolve_goalkeepers_team_id function:
    Assigns goalkeeper to team whose player centroid is closest to the goalkeeper position.
    """
    if len(players_xy) == 0:
        return 0

    team_0_xy = players_xy[players_team_ids == 0]
    team_1_xy = players_xy[players_team_ids == 1]

    if len(team_0_xy) == 0 or len(team_1_xy) == 0:
        return 0

    team_0_centroid = team_0_xy.mean(axis=0)
    team_1_centroid = team_1_xy.mean(axis=0)

    dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
    dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)

    return 0 if dist_0 < dist_1 else 1
