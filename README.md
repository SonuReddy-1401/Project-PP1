# Unified Pro Football Tactical Analytics Pipeline & Command Center

A production-grade, end-to-end computer vision and tactical analytics system for broadcast football (soccer) video. This pipeline ingests raw TV broadcast footage or tracking datasets, performs dynamic team color discovery, frame-by-frame camera homography calibration, player detection and tracking, physical outlier rejection, and outputs quantitative tactical analytics to a real-time interactive web dashboard.

---

## 🌟 System Architecture & Pipeline Flow

```
                     [ Broadcast TV Video / Positions Dataset ]
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │     1. Ingestion & Desktop GUI       │
                     │  (Tkinter Obsidian GUI / CLI Engine) │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │    2. Computer Vision Feature Engine │
                     │   - YOLOv8 Player Detection          │
                     │   - Torso HSV Kit Classifier         │
                     │   - Circular-Hue K-Means (k=2)       │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │ 3. PnLCalib Camera Calibration Engine│
                     │   - HRNet Keypoint & Line Prediction │
                     │   - Homography H (Pixel ➔ 2D Pitch m)│
                     │   - Pitch Polygon Pitch-Bound Filter │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │ 4. Temporal Smoothing & Outlier Rej. │
                     │   - Multi-Frame IoU Player Tracker   │
                     │   - Physical Speed Ceiling (< 10 m/s)│
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │      5. Tactical Metrics Engine      │
                     │   - Squad Width (m) & Depth (m)      │
                     │   - Convex Hull Area (m²)            │
                     │   - Centroid Trajectory (X, Y)       │
                     │   - Dynamic Pitch-Thirds (L➔R / R➔L) │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │   6. Pro Match Command Center Web UI │
                     │   - Interactive HTML5 Canvas Pitch   │
                     │   - Synchronized Scrubbing Timeline  │
                     │   - Plotly Analytics Graphs          │
                     │   - Data Quality & Methodology Audit │
                     └──────────────────────────────────────┘
```

---

## ✨ Key Features & Technical Capabilities

1. **Flexible Ingestion Interface**:
   - **Desktop GUI (`main.py`)**: Dark obsidian theme launcher with real-time logging, custom kit picker, match clock offset setting, and full screen resizable layout.
   - **Headless CLI (`run_pipeline.py`)**: Fully automated headless pipeline execution for batch processing or server deployments.

2. **Automated Team Color Discovery**:
   - Uses circular-hue K-Means clustering ($k=2$) on torso HSV crops to automatically segment target outfield players from opponents without manual bounding box annotations.

3. **Dynamic Attacking Direction Logic**:
   - Supports both **Left-to-Right (`left_to_right`)** and **Right-to-Left (`right_to_left`)** attacking orientations.
   - Dynamically flips pitch-thirds spatial classification (Defensive vs. Attacking third) to ensure metric accuracy regardless of half-time side switches or camera angles.

4. **Robust Pitch Homography Calibration**:
   - Integrates **PnLCalib HRNet** keypoint and line estimation to compute frame-by-frame homography $H$, mapping 2D pixel coordinates $[u, v]$ to real-world pitch coordinates $[x, y]$ in meters (Standard pitch: $105.0\text{ m} \times 68.0\text{ m}$).

5. **Physical Outlier Rejection & Tracking**:
   - **Multi-Frame IoU Tracker**: Smooths player trajectories across frames to eliminate frame flicker.
   - **Physical Speed Ceiling Filter**: Rejects camera pan artifacts and detection jumps by filtering out movements exceeding $10.0\text{ m/s}$ ($36\text{ km/h}$).

6. **Comprehensive Tactical Metrics**:
   - **Squad Width ($m$)**: Maximum lateral distance between outermost outfield players ($\Delta Y$).
   - **Squad Depth ($m$)**: Longitudinal distance between highest and lowest outfield lines ($\Delta X$).
   - **Convex Hull Area ($m^2$)**: Total pitch surface area occupied by the outfield unit.
   - **Centroid Trajectory ($x, y$)**: Real-time positional center of mass of the team.
   - **Pitch Thirds Dominance**: Frame percentage spent in Defensive, Middle, and Attacking thirds.

7. **Pro Match Command Center Web Dashboard**:
   - Built with Vanilla JS, HTML5 Canvas, and Plotly.js.
   - Features a **collapsible sidebar**, theme toggling (Dark Glass / Light Clean), interactive pitch heatmap, 2D player animation, synchronized match clock scrubber, player mobility breakdown, and data quality diagnostics.

---

## 📁 Repository Structure

```
UNIFIED_PIPELINE/
├── main.py                     # Desktop GUI Ingestion Launcher (Tkinter)
├── run_pipeline.py              # CLI Entry Point & Automated Web Server Launcher
├── pipeline/
│   ├── __init__.py             # Package Initialization
│   ├── config.py               # Color Theme Registry & Match Clock Parsers
│   ├── cv_engine.py            # YOLOv8 + PnLCalib + HSV Color Clustering Engine
│   ├── processor.py           # Tactical Metrics Engine (Width, Depth, Thirds, Area)
│   └── exporter.py            # Dashboard Asset Exporter (JSON & Narratives)
└── dashboard/
    ├── index.html              # Pro Match Command Center UI Structure
    ├── css/
    │   └── app.css             # Glassmorphism Design System & Theme Variables
    ├── js/
    │   ├── app.js              # Application Controller & Sidebar Handler
    │   ├── components.js       # Modular Analytics Views & Plotly Graph Renderers
    │   ├── pitch-canvas.js     # Dynamic HTML5 2D Pitch & Heatmap Canvas Renderer
    │   └── timeline.js         # Synchronized Video/Data Scrubbing Timeline Engine
    └── data/
        ├── positions_dataset.json  # Exported Pitch Coordinates per Frame
        ├── metadata.json           # Match Stats, Thirds, & Calibration Diagnostics
        └── narratives.json         # Automated Tactical Coaching Insights
```

---

## ⚙️ Installation & Requirements

### 1. Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended for PnLCalib and YOLO inference)

### 2. Environment Setup
```bash

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install opencv-python ultralytics huggingface_hub pyyaml numpy
```

---

## 🚀 How to Run

### Method 1: Desktop Ingestion Center (Recommended for GUI Users)

Launch the interactive desktop interface:
```bash
python main.py
```

#### GUI Controls & Parameters:
1. **Input Path**: Browse and select a broadcast video file (`.mp4`, `.avi`, `.mov`, `.mkv`) or a raw tracking position JSON file. *(Leave blank to run the benchmark dataset)*.
2. **Target Team Name**: Input the target squad label (e.g., `Real Madrid`, `Arsenal`, `Manchester City`).
3. **Opponent Name**: Input the opponent squad label (e.g., `Barcelona`, `Chelsea`).
4. **Target Kit Color**: Select a preset color (`sky_blue`, `red`, `white`, `yellow`, `emerald`, `navy`) or input a custom hex code (e.g., `#0080FF`).
5. **Attacking Direction**:
   - `left_to_right`: Team attacks towards $+X$ ($+52.5\text{m}$).
   - `right_to_left`: Team attacks towards $-X$ ($-52.5\text{m}$).
6. **Match Clock Start**: Set clip timestamp offset (e.g., `09:54`, `45:00`, `00:00`).
7. **Dashboard HTTP Port**: HTTP local server port (default: `8090`).

Click **"START TACTICAL ANALYSIS & LAUNCH DASHBOARD"**. The log console will display live processing output and automatically open the web dashboard in your browser upon completion.

---

### Method 2: Headless Command Line Interface (CLI)

Run processing directly from the terminal:

```bash
# Example 1: Processing a raw broadcast video clip (Right-to-Left attacking)
python run_pipeline.py \
  --video "S:/CLG/PP1/EXP/NEXT/data/broadcast_clip.mp4" \
  --team_name "Napoli" \
  --opponent_name "Roma" \
  --team_color "sky_blue" \
  --attacking_dir "right_to_left" \
  --match_start "09:54" \
  --port 8090

# Example 2: Processing pre-extracted positions dataset
python run_pipeline.py \
  --dataset "S:/CLG/PP1/EXP/NEXT/PRO_DASHBOARD/data/positions_dataset_clip15min.json" \
  --team_name "Arsenal" \
  --opponent_name "Chelsea" \
  --team_color "red" \
  --attacking_dir "left_to_right" \
  --match_start "00:00" \
  --port 8090
```

#### CLI Options:
| Flag | Description | Default |
|---|---|---|
| `--video` | Path to broadcast video file (`.mp4`, `.avi`, `.mov`) | `None` |
| `--dataset` | Path to raw positions dataset JSON file | `None` |
| `--team_name` | Target team label | `"Target Team"` |
| `--opponent_name` | Opponent team label | `"Opponent"` |
| `--team_color` | Kit color theme or hex code | `"sky_blue"` |
| `--attacking_dir` | Attacking orientation (`left_to_right` or `right_to_left`) | `"left_to_right"` |
| `--match_start` | Starting timestamp offset (`MM:SS`) | `"00:00"` |
| `--fps` | Processing FPS rate | `25.0` |
| `--port` | Web dashboard HTTP port | `8090` |
| `--no_launch` | Disable automatic browser opening | `False` |

---

## 📊 Pro Match Command Center Dashboard Guide

The web dashboard is organized into 7 tactical views, accessible via the top navigation bar or collapsible sidebar:

1. **Overview Dashboard (`#view-overview`)**:
   - Match metadata header with live clock readout and team color indicators.
   - High-level KPIs: Average Width ($m$), Average Depth ($m$), Pitch Area ($m^2$), and Centroid Displacement.
   - Interactive 2D Pitch Canvas with player nodes, convex hull polygon, and real-time centroid tracking marker.
   - Synchronized timeline scrubber with Play/Pause and variable playback speeds ($0.5\times$, $1.0\times$, $2.0\times$).

2. **Pitch Thirds Dominance (`#view-thirds`)**:
   - Bar chart breakdown of centroid time spent in Defensive, Middle, and Attacking thirds.
   - Displays explicit attacking direction indicator ($\rightarrow$ or $\leftarrow$) matching user GUI setup.

3. **Shape & Compactness (`#view-shape`)**:
   - Time-series line chart tracking Width ($m$), Depth ($m$), and Compactness ratio over time.
   - Automated coaching text highlight of expansion and compression phases.

4. **Convex Hull Pitch Area (`#view-area`)**:
   - Surface area ($m^2$) occupied by outfield unit over the match sequence.

5. **Centroid Trajectory (`#view-centroid`)**:
   - 2D positional trajectory plot of the team center of mass across the pitch.

6. **Distance & Mobility (`#view-players`)**:
   - Per-player physical tracking table detailing total distance covered ($m$), average pace ($km/h$), and tracking sample count.

7. **Methodology & Data Quality Audit (`#view-audit`)**:
   - System calibration diagnostics showing exact PnLCalib solve rate ($\%$) and physical speed outlier rejection percentage ($10\text{ m/s}$ ceiling filter).

---

## 🧪 Technical Formulas & Core Logic

- **Pixel to Pitch Conversion**:
  $$P_{\text{pitch}} = H^{-1} \cdot P_{\text{image}}$$
  where $H$ is the 3x3 homography matrix computed by PnLCalib HRNet model.

- **Attacking Thirds Classification**:
  - For **Left-to-Right** ($X \in [-52.5, +52.5]$):
    $$\text{Defensive Third: } X < -17.5\text{m} \quad | \quad \text{Middle Third: } -17.5\text{m} \le X \le 17.5\text{m} \quad | \quad \text{Attacking Third: } X > 17.5\text{m}$$
  - For **Right-to-Left** ($X \in [-52.5, +52.5]$):
    $$\text{Defensive Third: } X > 17.5\text{m} \quad | \quad \text{Middle Third: } -17.5\text{m} \le X \le 17.5\text{m} \quad | \quad \text{Attacking Third: } X < -17.5\text{m}$$

- **Physical Speed Outlier Filter**:
  $$\text{Speed} = \frac{\sqrt{(X_t - X_{t-1})^2 + (Y_t - Y_{t-1})^2}}{\Delta t} > 10.0 \text{ m/s} \implies \text{Reject Jump}$$

---

## 🛡️ License & Acknowledgments

- **YOLOv8 Detection Model**: Fine-tuned on football players (`uisikdag/yolo-v8-football-players-detection`).
- **PnLCalib Camera Calibration**: HRNet keypoint and line detection model for sports field homography.
- Built for academic research and advanced football tactical analytics.
