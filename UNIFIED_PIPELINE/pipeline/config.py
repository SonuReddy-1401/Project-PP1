"""
Configuration & Color Theme Registry for Unified Pipeline
"""
import os

# Preserved Pre-Built Color Themes for Popular Kits
COLOR_THEME_REGISTRY = {
    "sky_blue": {
        "primary": "#0080FF",
        "secondary": "#38BDF8",
        "badge": "Sky Blue Outfield",
        "label": "Sky Blue"
    },
    "red": {
        "primary": "#EF4444",
        "secondary": "#F87171",
        "badge": "Crimson Red Outfield",
        "label": "Red"
    },
    "white": {
        "primary": "#F8FAFC",
        "secondary": "#CBD5E1",
        "badge": "Classic White Outfield",
        "label": "White"
    },
    "yellow": {
        "primary": "#F59E0B",
        "secondary": "#FBBF24",
        "badge": "Amber Yellow Outfield",
        "label": "Yellow"
    },
    "emerald": {
        "primary": "#10B981",
        "secondary": "#34D399",
        "badge": "Emerald Green Outfield",
        "label": "Emerald"
    },
    "navy": {
        "primary": "#1E3A8A",
        "secondary": "#3B82F6",
        "badge": "Navy Blue Outfield",
        "label": "Navy"
    }
}

def parse_match_clock(clock_str: str) -> float:
    """
    Parses timestamp string "MM:SS" or "HH:MM:SS" into total offset seconds.
    Example: "09:54" -> 594.0 seconds.
    """
    if not clock_str or not isinstance(clock_str, str):
        return 0.0
    parts = clock_str.strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        pass
    return 0.0

def get_color_config(color_input: str):
    """
    Resolves kit color configuration from predefined preset or custom hex string.
    """
    color_key = color_input.lower().replace("-", "_").replace(" ", "_")
    if color_key in COLOR_THEME_REGISTRY:
        return COLOR_THEME_REGISTRY[color_key]
    
    # Custom Hex Code
    if color_input.startswith("#"):
        return {
            "primary": color_input,
            "secondary": color_input,
            "badge": f"Custom Kit ({color_input})",
            "label": "Custom Kit"
        }
    
    # Fallback to Sky Blue
    return COLOR_THEME_REGISTRY["sky_blue"]
