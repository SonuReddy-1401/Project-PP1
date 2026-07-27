import numpy as np

def get_center_of_bbox(bbox):
    """Returns center point (x, y) of bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """Returns width of bounding box [x1, y1, x2, y2]."""
    return bbox[2] - bbox[0]

def get_foot_position(bbox):
    """Returns bottom-center foot coordinate (x, y) of bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def measure_distance(p1, p2):
    """Returns Euclidean distance between two 2D points (x1, y1) and (x2, y2)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def measure_xy_distance(p1, p2):
    """Returns absolute (dx, dy) distance between two points."""
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
