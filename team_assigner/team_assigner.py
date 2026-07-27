import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    """
    Team Assigner matching Abdullah Tarek's football_analysis repository.
    Uses top jersey half crop HSV & RGB KMeans 2-cluster color segmentation to assign players to Team 1 or Team 2.
    Also supports explicit target color filtering (e.g., Red Team).
    """
    def __init__(self, target_color: str = "red"):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.target_color = target_color.lower() if target_color else None

    def get_clustering_model(self, image):
        # Reshape image to 2D array of pixels
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return np.array([0, 0, 0])

        # Extract top half (jersey region)
        top_half_image = image[0:int(image.shape[0] / 2), :]
        if top_half_image.size == 0:
            return np.array([0, 0, 0])

        # Get Clustering Model
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_

        # Reshape labels back to top half image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Find corner background cluster label
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        if len(player_colors) < 2:
            self.team_colors[1] = np.array([0, 0, 255])     # Red Team
            self.team_colors[2] = np.array([255, 255, 255]) # White Team
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Explicit Red Jersey Thresholding if target_color == "red"
        if self.target_color == "red":
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                h_crop = y2 - y1
                torso = crop[int(h_crop * 0.15):int(h_crop * 0.45), :]
                if torso.size > 0:
                    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                    m1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
                    m2 = cv2.inRange(hsv, np.array([155, 50, 40]), np.array([180, 255, 255]))
                    red_mask = cv2.bitwise_or(m1, m2)
                    r_ratio = np.sum(red_mask > 0) / float(torso.shape[0] * torso.shape[1] + 1e-5)
                    if r_ratio >= 0.15:
                        self.player_team_dict[player_id] = 1
                        return 1
                    else:
                        self.player_team_dict[player_id] = 2
                        return 2

        player_color = self.get_player_color(frame, bbox)
        if self.kmeans is not None:
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        else:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
