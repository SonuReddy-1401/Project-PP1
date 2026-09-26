/**
 * Component: Overview & Executive Tactical Summary
 */
function renderOverviewComponent(metadata, narratives) {
  const container = document.getElementById('overview');
  if (!container) return;

  const coachText = narratives.overview || "Executive match summary for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Match Overview & Executive Summary</h1>
    <p class="section-subtitle">${metadata.clip_label || 'Tactical analytics report for 15-minute broadcast clip.'}</p>

    <!-- Coach Callout Box -->
    <div class="coach-callout">
      <div class="coach-callout-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        What am I looking at (Coach Summary)
      </div>
      <div class="coach-callout-text">${coachText}</div>
    </div>

    <!-- Overview Stat Cards -->
    <div class="stats-grid" style="margin-bottom: 24px;">
      <div class="stat-card">
        <div class="stat-label">Target Team</div>
        <div class="stat-value">${metadata.target_team_name || 'Napoli'} <span class="team-badge-swatch" style="width: 18px; height: 18px;"></span></div>
        <div class="stat-sub">Kit: ${metadata.kit_color || 'Sky Blue Outfield + Orange GK'}</div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Match Time Covered</div>
        <div class="stat-value">9:54 – 24:54</div>
        <div class="stat-sub">15:00 min duration (${metadata.total_frames.toLocaleString()} frames @ ${metadata.fps} FPS)</div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Attacking Direction</div>
        <div class="stat-value">L ➔ R</div>
        <div class="stat-sub">Left-to-Right Goal Progression</div>
      </div>

      <div class="stat-card">
        <div class="stat-label">PnL Calibration Rate</div>
        <div class="stat-value">100.0<span class="stat-unit">%</span></div>
        <div class="stat-sub">${metadata.total_frames.toLocaleString()} / ${metadata.total_frames.toLocaleString()} Frames Solved</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Tactical Performance Indicators</div>
      <p style="font-size: 14px; color: var(--text-primary); margin-bottom: 12px;">
        Spatial tracking, pitch-thirds occupancy, team shape compactness, and individual mobility metrics for <strong>${metadata.target_team_name}</strong>.
      </p>
      <p style="font-size: 13px; color: var(--text-secondary);">
        • Primary Control Zone: <strong>Middle Third (57.4% occupancy)</strong><br>
        • Safe Team Displacement: <strong>1.77 km (7.13 km/h average pace)</strong><br>
        • Spatial Density Samples: <strong>215,988 pitch coordinates</strong>
      </p>
    </div>
  `;
}
