"""
Detection Filtering Module
Combines two filters, applied in this order for every detection:
  1. Fixed broadcast logo exclusion (screen-space, size-matched) — cheap, first pass.
  2. Geometric pitch-boundary filter (calibration-based) — removes anything
     whose foot-point projects outside the real pitch polygon.

Rationale for filter order and design is documented inline; see project report
for the precision/recall tradeoff discussion around the logo exclusion.
"""
import cv2
import numpy as np

# --- Fixed broadcast logo exclusion ---
# Measured across 5 frames (204, 1143, 1828, 6033, 6074) of broadcast_clip_1.mp4:
# logo box consistently within x:[1642,1699], y:[907,984]. Buffer added below.
LOGO_X1, LOGO_Y1, LOGO_X2, LOGO_Y2 = 1620, 890, 1720, 1000

# Logo box size range observed (~55-77px wide, ~65-95px tall) — used to avoid
# excluding real players who happen to pass through the same screen region.
LOGO_MIN_W, LOGO_MAX_W = 40, 90
LOGO_MIN_H, LOGO_MAX_H = 50, 110


def is_logo_overlay(x1, y1, x2, y2):
    """True if this box matches both the fixed logo screen position AND its
    characteristic small size — reduces false exclusion of real players who
    might pass through the same pixel region with a different box shape."""
    box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
    box_w, box_h = x2 - x1, y2 - y1
    in_zone = LOGO_X1 <= box_cx <= LOGO_X2 and LOGO_Y1 <= box_cy <= LOGO_Y2
    matches_logo_size = LOGO_MIN_W <= box_w <= LOGO_MAX_W and LOGO_MIN_H <= box_h <= LOGO_MAX_H
    return in_zone and matches_logo_size


def build_pitch_polygon(camera, pitch, corner_keys=None):
    """Projects the four real pitch corners into image space using an already-
    loaded Camera object (from sn-calibration's Camera class) and the SoccerPitch
    model. Returns an (4,2) float32 numpy array, or None if projection failed."""
    if corner_keys is None:
        corner_keys = ["TL_PITCH_CORNER", "TR_PITCH_CORNER", "BR_PITCH_CORNER", "BL_PITCH_CORNER"]

    polygon_points = []
    for key in corner_keys:
        pt2d = camera.project_point(pitch.point_dict[key])
        if np.isnan(pt2d).any():
            return None
        polygon_points.append((float(pt2d[0]), float(pt2d[1])))

    if len(polygon_points) != 4:
        return None
    return np.array(polygon_points, dtype=np.float32)


def is_on_pitch(x1, y1, x2, y2, polygon_np):
    """True if this box's foot-point (bottom-center) falls inside the pitch
    boundary polygon. Foot-point is used (not box center) because a standing
    player's feet touch the pitch plane; the torso/head do not."""
    if polygon_np is None:
        return True  # no calibration available for this frame — can't filter, so keep everything
    foot_x = (x1 + x2) / 2.0
    foot_y = y2
    result = cv2.pointPolygonTest(polygon_np, (foot_x, foot_y), False)
    return result >= 0


def filter_detections(detections, camera, pitch):
    """
    detections: list of (cls_name, conf, x1, y1, x2, y2)
    camera: loaded Camera object for this frame, or None if calibration failed
    pitch: SoccerPitch instance

    Returns (kept, rejected), each a list of (cls_name, conf, x1, y1, x2, y2, reason)
    """
    polygon_np = build_pitch_polygon(camera, pitch) if camera is not None else None

    kept, rejected = [], []
    for cls_name, conf, x1, y1, x2, y2 in detections:
        if is_logo_overlay(x1, y1, x2, y2):
            rejected.append((cls_name, conf, x1, y1, x2, y2, "logo_overlay"))
            continue
        if not is_on_pitch(x1, y1, x2, y2, polygon_np):
            rejected.append((cls_name, conf, x1, y1, x2, y2, "off_pitch"))
            continue
        kept.append((cls_name, conf, x1, y1, x2, y2, "kept"))

    return kept, rejected
