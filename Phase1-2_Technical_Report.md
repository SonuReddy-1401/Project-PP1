# Intelligent Sports Performance Analysis Through Video Analytics — Football
## Phase 1–2 Technical Report: Broadcast Camera Calibration

---

## 1. Executive Summary

This report covers the design, implementation, debugging, and evaluation of the
camera calibration subsystem for the football video analytics pipeline. Camera
calibration — recovering the mapping between broadcast image pixels and real-world
pitch coordinates — is a prerequisite for filtering out off-pitch detections (crowd,
bench, staff) from downstream player/ball tracking, since it allows any detected
object's position to be projected onto the known FIFA pitch geometry and tested for
pitch membership.

Two approaches were implemented and evaluated end-to-end on original broadcast
footage:

- **Phase 1:** The official SoccerNet Camera Calibration baseline (`sn-calibration`),
  using a DeepLabv3-based line segmentation network.
- **Phase 2:** `PnLCalib`, a more recent points-and-lines optimization approach
  (HRNet-based keypoint + line detection), adopted after the intended Phase 2 target
  (the 2023 challenge-winning "Sportlight" solution) was found to be infeasible on
  available hardware.

**Headline result:** Phase 2 (PnLCalib) achieved substantially higher and more
reliable calibration accuracy than the Phase 1 baseline, most dramatically on a
broadcast source (a different stadium/camera setup) where the Phase 1 baseline
failed almost completely.

---

## 2. Objective

Given a broadcast or tactical-camera football video, recover per-frame camera
calibration (an estimate of the camera's pose and intrinsics, or equivalently a
frame-to-pitch homography) accurate enough to:

1. Project the known FIFA pitch line/circle geometry back onto the video frame for
   visual verification.
2. Support later filtering of player/ball detections by pitch membership, replacing
   an earlier HSV-color-based pitch segmentation method that achieved only ~75%
   accuracy and did not generalize across lighting conditions.

Two test clips were used throughout:
- **Clip A ("clip1"):** ~1 minute, 25 fps, 1920×1080, tactical-camera footage with
  relatively stable camera motion.
- **Clip B ("clip2"):** ~2 minutes 30 seconds, 50 fps, 1920×1080, standard broadcast
  footage from a different stadium (Etihad Stadium) with on-screen scoreboard graphics
  and more active camera movement.

---

## 3. Phase 1 — SoccerNet Baseline (`sn-calibration`)

### 3.1 Methodology

The official SoccerNet Camera Calibration development kit (`SoccerNet/sn-calibration`)
was used with its provided pretrained weights (no training required). The pipeline
consists of three stages:

1. **Line extremity detection** (`detect_extremities.py`): a DeepLabv3 semantic
   segmentation network detects pitch line pixels, which are skeletonized and reduced
   to line-class endpoint coordinates (normalized 0–1), at an internal resolution of
   640×360.
2. **Camera parameter solving** (`baseline_cameras.py`): detected line endpoints are
   denormalized to a 960×540 working resolution, matched against a fixed 3D FIFA
   pitch model, and a homography is estimated via direct linear transform (DLT) from
   line correspondences, then decomposed into physical camera parameters (pan, tilt,
   roll, 3D position, focal length).
3. **Visualization/rendering**: camera parameters are used to reproject the full 3D
   pitch model back into image space for verification.

A clean, decoupled pipeline was built around these three stages (frame extraction →
detection → camera solving → JSON-only rendering), replacing an initial video
renderer that attempted to solve homography live, per frame, using a custom
(unvalidated) reimplementation of the line-correspondence DLT solver.

### 3.2 Engineering issues identified and resolved

Two genuine implementation bugs were found and fixed during development, both
instructive for the overall project:

**Bug 1 — Untested live homography reimplementation.** An early full-video renderer
recomputed homography estimation inline, per frame, using a hand-written
`_estimate_homography()` / `_normalization_transform()` pair rather than the repo's
own tested equivalents (`estimate_homography_from_line_correspondences()`,
`normalization_transform()` in `baseline_cameras.py`). This produced a completely
blank output video (zero visible overlay lines across 1602 frames) despite no runtime
errors. Resolution: the pipeline was redesigned so that camera parameters are always
computed once via the tested batch scripts and written to disk as JSON; the video
renderer's sole job is to read and visualize these already-verified parameters,
eliminating live, unvalidated geometry computation from the rendering path entirely.

**Bug 2 — Resolution mismatch in visualization.** The visualization script initialized
its internal `Camera` object at 640×360 and scaled projected points by
`1920/640 = 3.0×`, while the camera parameters were actually solved at 960×540 (as
confirmed by the JSON output's `principal_point: [480.0, 270.0]`, and later verified
directly in `baseline_cameras.py`'s default arguments). The correct scale factor is
`1920/960 = 2.0×`. This 50% systematic distortion caused visibly misaligned but
plausibly-shaped overlays (correct topology, wrong scale/position) before being
identified and corrected.

### 3.3 Results

**Clip A (~1 minute, 61-frame calibration sample, later confirmed at full 1501-frame
scale):**

- Approximately **75–80% of frames** produced accurate, well-aligned pitch line
  overlays.
- Camera-parameter solving succeeded for **~77% of sampled frames** (e.g., 47/61 in
  the initial sample); failures were concentrated in frames with fewer than 4 usable
  line correspondences after excluding circular features (which are not used in the
  line-based DLT solve).
- **Root cause of failures, confirmed via direct measurement:** accuracy correlated
  with the *length* of detected line segments, not merely their count. Frames
  dominated by short, tightly-clustered lines near the goal structure (crossbar,
  posts, small/big rectangle edges — typically <0.2 in normalized length) produced
  unstable homography solutions even with 10+ detected line classes, because
  short-segment line-fitting is substantially more sensitive to pixel-level detection
  noise than long-segment fitting (e.g., touchlines, halfway line, both >0.5 in
  normalized length). This was confirmed by direct per-line length measurement across
  both a failing and a succeeding frame.
- A secondary, expected failure mode occurred during brief camera pan transitions,
  consistent with motion blur and transient partial views degrading detection
  quality.

**Clip B (~2.5 minutes, different stadium/broadcast source):**

- File-level camera-parameter solving succeeded for **95.6% of frames** (7167/7500),
  suggesting detection and DLT solving completed "successfully" in the technical
  sense.
- However, **visual accuracy was very poor** — approximately 1 in 150 frames produced
  a usable overlay; the remainder showed extreme, wildly distorted lines and grossly
  mis-shaped ellipses extending well outside the pitch area.
- **Diagnosis:** raw line-extremity detections were individually plausible and
  correctly positioned relative to visible pitch features (confirmed by direct
  coordinate inspection). The instability instead occurred in the
  homography-to-camera-parameter decomposition and extrapolation step: the solved
  camera model, while locally consistent with the directly-observed lines, produced
  physically implausible projections when extrapolated to pitch regions far outside
  what was directly detected (e.g., the far touchline, opposite penalty box). This
  indicates the baseline's single-homography-fit-then-decompose approach is sensitive
  to camera intrinsics/geometry that differ from its SoccerNet training distribution,
  a known limitation of one-shot DLT-based calibration.

**Summary — Phase 1:** The baseline validated the overall calibration approach
(keypoint/line detection → homography → 3D camera parameters → reprojection) and
produced a usable ~75–80% accuracy rate on footage resembling its training
distribution, but failed to generalize reliably to a different broadcast source,
motivating evaluation of a stronger model in Phase 2.

---

## 4. Phase 2 — Stronger Model Evaluation

### 4.1 Sportlight (2023 challenge winner) — found infeasible

The intended Phase 2 upgrade was `NikolasEnt/soccernet-calibration-sportlight`, the
1st-place solution (HRNetV2-w48 backbone, 57-keypoint annotation scheme including
ellipse tangent points, RANSAC-style geometric refinement) for the SoccerNet Camera
Calibration Challenge 2023. On inspection, this repository was found unsuitable for
the available environment:

- Requires a Linux host with Docker and NVIDIA Container Toolkit (development
  environment is Windows).
- Requires an Nvidia GPU with **at least 24GB VRAM** (available hardware: RTX 3050,
  6GB).
- Ships **no pretrained checkpoint** — the repository contains training code only
  (`train.py` for both the keypoint and line models); reproducing the winning result
  would require training both models from scratch on the challenge dataset.

Given these constraints, pursuing Sportlight directly was assessed as infeasible
without substantial additional infrastructure (cloud GPU rental, Linux environment,
full training pipeline) disproportionate to the project's scope.

### 4.2 PnLCalib — selected alternative

`mguti97/PnLCalib` ("Sports Field Registration via Points and Lines Optimization",
accepted to *Computer Vision and Image Understanding*) was identified as a
practical substitute using a closely related methodology (keypoint and line
detection, geometric optimization) with the critical advantage of **publicly
available pretrained weights** (single-view keypoint and line detection models,
pretrained on SoccerNet, downloadable directly) and a ready-to-run video inference
script, requiring no training or Docker/Linux setup.

**Environment setup notes:** initial `pip install` of `requirements.txt` produced a
CPU-only PyTorch build (confirmed via `torch.cuda.is_available() == False`) despite a
CUDA-capable GPU being present. This was corrected by explicitly reinstalling PyTorch
2.6.0 from the `cu124` index, after which GPU availability was confirmed
(`NVIDIA GeForce RTX 3050 6GB Laptop GPU` detected, `CUDA available: True`).

### 4.3 Results

Two configurations were tested: the base model (keypoint + line detection only) and
the same model with the optional PnL non-linear refinement module (`--pnl_refine`)
enabled.

| Clip | Phase 1 baseline (sn-calibration) | Phase 2 base (PnLCalib) | Phase 2 + refinement |
|---|---|---|---|
| Clip A (~1 min, stable camera) | ~75–80% overall | ~90% on static/still frames; ~75% during initial camera movement; ~65% during sustained panning | Not evaluated |
| Clip B (~2.5 min, different broadcast source, active camera movement) | ~1/150 frames usable (near-total failure) | **~85% overall**, with a sub-2-second dip in accuracy during a hard right-to-left camera pan, self-correcting immediately afterward | ~81–82% overall (slightly worse than base; did not resolve the pan-transition dip; ~2× runtime of the base model) |

**Runtime:** base model inference ran at approximately 2.1–2.35 it/s on the RTX 3050
(GPU-confirmed via device query and wall-clock timing), processing the ~2.5 minute /
7500-frame clip in ~60 minutes and the ~1 minute / ~1500-frame clip in ~11 minutes.
The refinement-enabled run on the same 2.5-minute clip took approximately 2 hours,
roughly double the base runtime, for a slightly worse result.

**Key finding — refinement module did not improve results on this footage.** Despite
its role in Sportlight's original design (non-linear optimization using detected
line correspondences to refine an initial calibration estimate), enabling
`--pnl_refine` produced marginally lower accuracy and did not close the
pan-transition accuracy dip, at roughly double the computational cost. This suggests
the refinement module's benefit, as tuned/validated on its original benchmark
datasets, does not necessarily transfer to arbitrary broadcast sources without
further tuning. **Recommendation: use the base PnLCalib model (no refinement) for
this project's footage**, as it is both faster and at least as accurate.

**Data integrity note:** inference runs on Clip A twice terminated at frame 1501 of a
metadata-reported 1602 frames, with no Python-level error or traceback observed. This
was diagnosed as a discrepancy between the video container's reported frame count
(`cv2.CAP_PROP_FRAME_COUNT`) and the number of frames actually decodable via
sequential `cap.read()` calls — a well-documented characteristic of certain MP4
encodings, not a pipeline defect. Output file size was consistent across repeated
runs, supporting that outputs were complete relative to the true frame count rather
than truncated by a crash.

---

## 5. Comparative Analysis

The most significant finding of this phase is the difference in **cross-broadcast
generalization** between the two approaches. The Phase 1 baseline performed
reasonably on footage resembling its training distribution (Clip A) but degraded to
near-total failure on a different broadcast source (Clip B), despite individual line
detections remaining locally plausible — the failure was specifically in the
camera-parameter decomposition and long-range extrapolation step. PnLCalib's base
model not only matched but substantially exceeded the baseline's performance on the
*same* previously-failing footage (Clip B: ~1/150 usable → ~85% overall), indicating
meaningfully better generalization, consistent with its richer point-based geometric
constraints (versus the baseline's line-only DLT approach).

Both approaches share a common, expected weak point: **accuracy degrades during
active camera motion** (panning/tilting), though the degradation is far less severe
and shorter-lived in PnLCalib (~65% during sustained panning, self-correcting within
under two seconds) than in the baseline (extended failure zones spanning several
consecutive frames around transitions).

---

## 6. Engineering Process Notes

This phase surfaced two categories of issues worth highlighting for future project
phases:

1. **Reimplemented-vs-reused logic bugs.** The most severe failure encountered (a
   completely blank calibrated video) traced back to reimplementing tested
   third-party math (homography-from-line-correspondences) rather than importing and
   reusing the original, validated function. Future phases should default to
   importing and calling existing tested functions directly rather than porting their
   logic by hand, even for seemingly simple operations.
2. **Resolution/coordinate-convention mismatches** between pipeline stages were
   responsible for a second major but more subtle bug (systematic 50% scale
   distortion). When integrating multiple scripts or models, explicit verification of
   shared assumptions (internal working resolution, coordinate normalization
   convention, principal point placement) is warranted before debugging accuracy
   issues at a higher level.

---

## 7. Conclusions and Recommendations

- **Phase 1 (SoccerNet baseline) is validated as a working proof of concept** but is
  not recommended for production use in this project due to poor cross-broadcast
  generalization.
- **Phase 2 (PnLCalib, base configuration, no refinement) is recommended as the
  calibration model for subsequent project phases**, based on substantially higher
  and more broadcast-independent accuracy, acceptable inference speed on available
  hardware (RTX 3050), and no training or specialized infrastructure requirements.
- **Next steps:** integrate PnLCalib's per-frame camera parameters into the pitch-
  membership filtering step (projecting player/ball detection foot-points into pitch
  coordinates and testing against the known FIFA boundary), replacing the earlier
  HSV-based approach entirely. Given the observed accuracy dip during sustained
  camera panning, a lightweight temporal smoothing or last-known-good fallback
  strategy (already implemented in the rendering pipeline) should be retained in the
  production filtering logic to bridge these brief low-confidence windows.
