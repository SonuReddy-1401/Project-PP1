# Roboflow Football AI (`football-ai.ipynb`) Subfolder Module

This standalone, isolated module implements the official [Roboflow Football AI (`football-ai.ipynb`)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb) architecture without modifying the parent repository code.

---

## 📂 Subfolder Layout

- `input/`: Contains input high quality 1080p HD broadcast videos (`new_match_red_team_1080p.mp4`).
- `output/`: Rendered outputs (`1_broadcast_tracking.mp4` & `2_tactical_pitch_mapping.mp4`).
- `src/`: Core Roboflow AI classes (`TeamClassifier`, `PitchTransformer`, `FootballAITracker`).
- `run_football_ai.py`: Low-RAM frame streaming pipeline runner.

---

## 🚀 How to Execute

```powershell
.\.venv\Scripts\python.exe roboflow_football_ai/run_football_ai.py --input roboflow_football_ai/input/new_match_red_team_1080p.mp4
```
