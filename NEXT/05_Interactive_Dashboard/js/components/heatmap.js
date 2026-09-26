/**
 * Component: Team Heatmap
 */
function renderHeatmapComponent(rawDataset, metadata, narratives) {
  const container = document.getElementById('heatmap');
  if (!container) return;

  const coachText = narratives.team_heatmap || "Spatial density explanation for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Team Spatial Occupancy Heatmap</h1>
    <p class="section-subtitle">Density distribution of target team pitch positions across match time.</p>

    <!-- Coach Callout Box -->
    <div class="coach-callout">
      <div class="coach-callout-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        What am I looking at (Coach Summary)
      </div>
      <div class="coach-callout-text">${coachText}</div>
    </div>

    <div class="card">
      <div class="controls-row">
        <div class="range-group">
          <span>Time Window:</span>
          <input type="range" id="heatmap-start-slider" min="0" max="900" value="0" oninput="updateHeatmapFilter()">
          <span id="heatmap-start-val" style="font-weight:700; color:var(--text-primary);">9:54</span>
          <span>to</span>
          <input type="range" id="heatmap-end-slider" min="0" max="900" value="900" oninput="updateHeatmapFilter()">
          <span id="heatmap-end-val" style="font-weight:700; color:var(--text-primary);">24:54</span>
        </div>
        <div style="font-size: 12px; color: var(--text-secondary); margin-left: auto;">
          Active Spatial Points: <strong id="heatmap-point-count" style="color: var(--text-primary);">215,988</strong>
        </div>
      </div>

      <div id="plot-heatmap" class="plot-container"></div>

      <div class="data-table-container">
        <button class="btn" onclick="toggleRawTable('heatmap-table-wrap')">Show Raw Position Data</button>
        <button class="btn" onclick="exportCSV('heatmap-table', 'team_heatmap_positions.csv')" style="margin-left: 8px;">Export CSV</button>

        <div id="heatmap-table-wrap" class="table-wrapper" style="display: none;">
          <table id="heatmap-table">
            <thead>
              <tr>
                <th>Match Time</th>
                <th>Frame Index</th>
                <th>Pitch X (m)</th>
                <th>Pitch Y (m)</th>
              </tr>
            </thead>
            <tbody id="heatmap-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  // Draw initial plot
  updateHeatmapPlotData(rawDataset, metadata);
}

function updateHeatmapPlotData(rawDataset, metadata) {
  if (!rawDataset.length) return;

  const startSec = parseFloat(document.getElementById('heatmap-start-slider').value);
  const endSec = parseFloat(document.getElementById('heatmap-end-slider').value);

  const startClock = secondsToMatchClock(startSec, metadata.clip_start_offset_sec || 594.0);
  const endClock = secondsToMatchClock(endSec, metadata.clip_start_offset_sec || 594.0);

  document.getElementById('heatmap-start-val').innerText = startClock;
  document.getElementById('heatmap-end-val').innerText = endClock;

  let xPts = [];
  let yPts = [];
  let tableRowsHTML = '';
  let rowCount = 0;

  rawDataset.forEach(entry => {
    if (!entry.calibration_ok) return;
    const t = entry.timestamp_sec;
    if (t >= startSec && t <= endSec) {
      (entry.target_positions || []).forEach(pt => {
        xPts.push(pt[0]);
        yPts.push(pt[1]);

        if (rowCount < 100) {
          const clock = frameToTimestamp(entry.frame_idx, metadata.fps || 25.0, metadata.clip_start_offset_sec || 594.0);
          tableRowsHTML += `<tr><td style="font-weight:700;">${clock}</td><td style="color:#6b6b6b;">${entry.frame_idx}</td><td>${pt[0]}</td><td>${pt[1]}</td></tr>`;
          rowCount++;
        }
      });
    }
  });

  document.getElementById('heatmap-point-count').innerText = xPts.length.toLocaleString();
  document.getElementById('heatmap-table-body').innerHTML = tableRowsHTML;

  const data = [{
    x: xPts,
    y: yPts,
    type: 'histogram2dcontour',
    colorscale: 'YlOrRd',
    revertscale: false,
    ncontours: 12,
    contours: { coloring: 'heatmap' },
    opacity: 0.65
  }];

  const layout = getPitchLayout(`Team Spatial Density (${startClock} - ${endClock} | ${xPts.length.toLocaleString()} points)`);
  Plotly.newPlot('plot-heatmap', data, layout, { responsive: true });
}

function updateHeatmapFilter() {
  if (window.gRawDataset && window.gMetadata) {
    updateHeatmapPlotData(window.gRawDataset, window.gMetadata);
  }
}
