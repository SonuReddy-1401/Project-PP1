/**
 * Component: Team Centroid Trajectory Path
 */
let gTrajMode = 'filtered'; // 'filtered' or 'raw'

function renderTrajectoryComponent(rawDataset, metadata, narratives) {
  const container = document.getElementById('trajectory');
  if (!container) return;

  const coachText = narratives.team_trajectory || "Team trajectory explanation for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Team Centroid Trajectory Path</h1>
    <p class="section-subtitle">Spatial center-of-mass movement. Demonstrates raw vs speed-outlier filtered displacement.</p>

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
        <button class="btn btn-toggle active" id="traj-mode-filtered" onclick="setTrajMode('filtered')">Outlier-Filtered (Safe 1.77 km)</button>
        <button class="btn btn-toggle" id="traj-mode-raw" onclick="setTrajMode('raw')">Raw Unfiltered (24.01 km Glitch Path)</button>

        <div class="range-group" style="margin-left: auto;">
          <span>Play/Scrub Frame:</span>
          <input type="range" id="traj-scrub-slider" min="0" max="1000" value="0" oninput="updateTrajScrub()">
          <span id="traj-scrub-val" style="font-weight: 700; color: var(--text-primary);">9:54</span>
        </div>
      </div>

      <div id="plot-trajectory" class="plot-container"></div>

      <div class="data-table-container">
        <button class="btn" onclick="toggleRawTable('traj-table-wrap')">Show Raw Trajectory Data</button>
        <button class="btn" onclick="exportCSV('traj-table', 'team_centroid_trajectory.csv')" style="margin-left: 8px;">Export CSV</button>

        <div id="traj-table-wrap" class="table-wrapper" style="display: none;">
          <table id="traj-table">
            <thead>
              <tr>
                <th>Match Time</th>
                <th>Frame Index</th>
                <th>Centroid X (m)</th>
                <th>Centroid Y (m)</th>
                <th>Implied Speed (m/s)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="traj-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  renderTrajectoryPlotData(rawDataset, metadata);
}

function renderTrajectoryPlotData(rawDataset, metadata) {
  if (!rawDataset.length) return;

  const fps = metadata.fps || 25.0;
  const clipStart = metadata.clip_start_offset_sec || 594.0;

  let cXs = [], cYs = [], times = [], speeds = [], statusList = [], clockLabels = [];
  let tableRowsHTML = '';
  let rowCount = 0;

  for (let i = 0; i < rawDataset.length; i++) {
    const entry = rawDataset[i];
    if (!entry.calibration_ok || !entry.target_positions || !entry.target_positions.length) continue;

    const pts = entry.target_positions;
    const mx = pts.map(p => p[0]).reduce((a, b) => a + b, 0) / pts.length;
    const my = pts.map(p => p[1]).reduce((a, b) => a + b, 0) / pts.length;

    let speed = 0;
    let isRejected = false;

    if (cXs.length > 0) {
      const prevX = cXs[cXs.length - 1];
      const prevY = cYs[cYs.length - 1];
      const disp = Math.sqrt((mx - prevX)**2 + (my - prevY)**2);
      speed = disp * fps;
      if (speed > 10.0) isRejected = true;
    }

    const clock = frameToTimestamp(entry.frame_idx, fps, clipStart);
    cXs.push(mx);
    cYs.push(my);
    times.push(entry.timestamp_sec);
    clockLabels.push(clock);
    speeds.push(speed.toFixed(2));
    statusList.push(isRejected ? 'REJECTED (>10m/s)' : 'OK');

    if (rowCount < 100) {
      tableRowsHTML += `<tr><td style="font-weight:700;">${clock}</td><td style="color:#6b6b6b;">${entry.frame_idx}</td><td>${mx.toFixed(2)}</td><td>${my.toFixed(2)}</td><td>${speed.toFixed(2)}</td><td style="color:${isRejected?'#f43f5e':'#22c55e'}">${isRejected?'REJECTED':'OK'}</td></tr>`;
      rowCount++;
    }
  }

  document.getElementById('traj-table-body').innerHTML = tableRowsHTML;

  let finalXs = [], finalYs = [];
  if (gTrajMode === 'filtered') {
    finalXs.push(cXs[0]);
    finalYs.push(cYs[0]);
    for (let i = 1; i < cXs.length; i++) {
      if (statusList[i] === 'OK') {
        finalXs.push(cXs[i]);
        finalYs.push(cYs[i]);
      }
    }
  } else {
    finalXs = cXs;
    finalYs = cYs;
  }

  const tracePath = {
    x: finalXs,
    y: finalYs,
    mode: 'lines',
    name: gTrajMode === 'filtered' ? 'Safe Filtered Path (1.77 km)' : 'Raw Unfiltered Path (24.01 km)',
    line: { color: gTrajMode === 'filtered' ? '#38bdf8' : '#f43f5e', width: 2 }
  };

  const traceStart = { x: [finalXs[0]], y: [finalYs[0]], mode: 'markers', name: 'Start', marker: { color: '#22c55e', size: 10 } };
  const traceEnd = { x: [finalXs[finalXs.length-1]], y: [finalYs[finalYs.length-1]], mode: 'markers', name: 'Finish', marker: { color: '#f43f5e', size: 12, symbol: 'x' } };

  const layout = getPitchLayout(`Team Centroid Trajectory Path (${gTrajMode.toUpperCase()} Mode)`);
  Plotly.newPlot('plot-trajectory', [tracePath, traceStart, traceEnd], layout, { responsive: true });
}

function setTrajMode(mode) {
  gTrajMode = mode;
  document.getElementById('traj-mode-filtered').classList.toggle('active', mode === 'filtered');
  document.getElementById('traj-mode-raw').classList.toggle('active', mode === 'raw');
  if (window.gRawDataset && window.gMetadata) {
    renderTrajectoryPlotData(window.gRawDataset, window.gMetadata);
  }
}

function updateTrajScrub() {
  if (window.gRawDataset && window.gMetadata) {
    renderTrajectoryPlotData(window.gRawDataset, window.gMetadata);
  }
}
