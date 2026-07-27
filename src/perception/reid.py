import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import supervision as sv

class PlayerReIDManager:
    """
    Deep Player Re-Identification (Re-ID) & Appearance Memory Gallery.
    Maintains persistent player identity across camera cuts, zooms, and off-screen exits.
    Extracts 16-D CIELAB + HSV + Spatial Aspect Ratio feature embeddings for each player crop.
    """
    def __init__(self, similarity_threshold: float = 0.68, max_gallery_size: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        
        # Local ByteTrack ID -> Global Persistent Re-ID
        self.track_to_global_map: Dict[int, int] = {}
        
        # Global ID -> List of 16-D feature embeddings
        self.gallery: Dict[int, List[np.ndarray]] = {}
        self.next_global_id = 1

    def extract_appearance_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a 16-dimensional visual appearance embedding from a player bounding box crop.
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return None

        h_crop, w_crop = crop.shape[:2]
        
        # 1. Jersey region (Upper 45% of crop)
        jersey_crop = crop[:int(h_crop * 0.45), :]
        if jersey_crop.size == 0:
            return None
            
        j_lab = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2LAB)
        j_hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        
        j_lab_mean, j_lab_std = cv2.meanStdDev(j_lab)
        j_hsv_mean, j_hsv_std = cv2.meanStdDev(j_hsv)
        
        # 2. Shorts region (Middle 25% of crop)
        shorts_crop = crop[int(h_crop * 0.45):int(h_crop * 0.70), :]
        if shorts_crop.size == 0:
            s_lab_mean = j_lab_mean
        else:
            s_lab = cv2.cvtColor(shorts_crop, cv2.COLOR_BGR2LAB)
            s_lab_mean, _ = cv2.meanStdDev(s_lab)

        # 3. Spatial Aspect Ratio features
        aspect_ratio = h_crop / float(w_crop + 1e-5)
        rel_height = h_crop / float(frame.shape[0])
        
        # Build 16-D Feature Vector
        feature_vec = np.array([
            j_lab_mean[0][0], j_lab_mean[1][0], j_lab_mean[2][0],  # Jersey LAB Mean
            j_lab_std[0][0],  j_lab_std[1][0],  j_lab_std[2][0],   # Jersey LAB Std
            j_hsv_mean[0][0], j_hsv_mean[1][0], j_hsv_mean[2][0],  # Jersey HSV Mean
            j_hsv_std[0][0],  j_hsv_std[1][0],  j_hsv_std[2][0],   # Jersey HSV Std
            s_lab_mean[1][0], s_lab_mean[2][0],                    # Shorts AB Mean
            aspect_ratio * 10.0, rel_height * 100.0                # Spatial geometry
        ], dtype=np.float32)

        # L2 Normalize Feature Vector
        norm = np.linalg.norm(feature_vec)
        if norm > 1e-5:
            feature_vec /= norm
            
        return feature_vec

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine similarity between two feature vectors.
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-5 or norm2 < 1e-5:
            return 0.0
        return float(dot / (norm1 * norm2))

    def get_global_id(self, frame: np.ndarray, track_id: int, bbox: np.ndarray) -> int:
        """
        Resolves local ByteTrack ID to persistent global Re-ID.
        Checks appearance memory gallery for matches when a player re-enters.
        """
        # If already mapped in active session
        if track_id in self.track_to_global_map:
            gid = self.track_to_global_map[track_id]
            embedding = self.extract_appearance_embedding(frame, bbox)
            if embedding is not None:
                self.update_gallery(gid, embedding)
            return gid

        embedding = self.extract_appearance_embedding(frame, bbox)
        if embedding is None:
            gid = self.next_global_id
            self.next_global_id += 1
            self.track_to_global_map[track_id] = gid
            return gid

        # Match against Gallery of existing players
        best_gid = None
        best_sim = -1.0
        
        for gid, vec_list in self.gallery.items():
            if not vec_list:
                continue
            # Compare with average gallery embedding for this player
            mean_gallery_vec = np.mean(vec_list, axis=0)
            sim = self.compute_cosine_similarity(embedding, mean_gallery_vec)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is not None and best_sim >= self.similarity_threshold:
            # Re-ID Match Found! Re-assign existing global ID
            self.track_to_global_map[track_id] = best_gid
            self.update_gallery(best_gid, embedding)
            return best_gid
        else:
            # Create new persistent identity
            gid = self.next_global_id
            self.next_global_id += 1
            self.track_to_global_map[track_id] = gid
            self.update_gallery(gid, embedding)
            return gid

    def update_gallery(self, global_id: int, embedding: np.ndarray):
        """
        Updates appearance gallery memory for a persistent player identity.
        """
        if global_id not in self.gallery:
            self.gallery[global_id] = []
        self.gallery[global_id].append(embedding)
        if len(self.gallery[global_id]) > self.max_gallery_size:
            self.gallery[global_id].pop(0)
