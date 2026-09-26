/**
 * Component: Team Shape (Width & Depth Over Time)
 */
let gShapeMode = 'smoothed'; // 'smoothed' or 'raw'

function renderShapeComponent(rawDataset, metadata, narratives) {
  const container = document.getElementById('shape');
  if (!container) return;

  const coachText = narratives.team_shape || "Team width & depth explanation for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Team Shape & Structural Compactness Over Time</h1>
    <p class="section-subtitle">Target team pitch width (Left-to-Right span) and depth (Deepest-to-Advanced span) across match time.</p>

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
        <button class="btn btn-toggle active" id="shape-mode-smoothed" onclick="setShapeMode('smoothed')">Smoothed View (1s Rolling Average)</button>
        <button class="btn btn-toggle" id="shape-mode-raw" onclick="setShapeMode('raw')">Raw Unfiltered Data</button>
      </div>

      <div id="plot-shape" class="plot-container"></div>

      <div class="data-table-container">
        <button class="btn" onclick="toggleRawTable('shape-table-wrap')">Show Raw Shape Data</button>
        <button class="btn" onclick="exportCSV('shape-table', 'team_shape_width_depth.csv')" style="margin-left: 8px;">Export CSV</button>

        <div id="shape-table-wrap" class="table-wrapper" style="display: none;">
          <table id="shape-table">
            <thead>
              <tr>
                <th>Match Time</th>
                <th>Frame Index</th>
                <th>Team Width (m)</th>
                <th>Team Depth (m)</th>
                <th>Player Count</th>
              </tr>
            </thead>
            <tbody id="shape-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  renderShapePlotData(rawDataset, metadata);
}

function renderShapePlotData(rawDataset, metadata) {
  if (!rawDataset.length) return;

  const fps = metadata.fps || 25.0;
  const clipStart = metadata.clip_start_offset_sec || 594.0;

  let rawTimes = [];
  let matchClockLabels = [];
  let rawWidths = [];
  let rawDepths = [];
  let tableRowsHTML = '';
  let rowCount = 0;

  rawDataset.forEach(entry => {
    if (!entry.calibration_ok || !entry.target_positions || entry.target_positions.length < 2) return;

    const pts = entry.target_positions;
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);

    const width = Math.max(...ys) - Math.min(...ys);
    const depth = Math.max(...xs) - Math.min(...xs);

    const clock = frameToTimestamp(entry.frame_idx, fps, clipStart);
    matchClockLabels.push(clock);
    rawTimes.push(entry.timestamp_sec);
    rawWidths.push(width);
    rawDepths.push(depth);

    if (rowCount < 100) {
      tableRowsHTML += `<tr><td style="font-weight:700;">${clock}</td><td style="color:#6b6b6b;">${entry.frame_idx}</td><td>${width.toFixed(2)}</td><td>${depth.toFixed(2)}</td><td>${pts.length}</td></tr>`;
      rowCount++;
    }
  });

  document.getElementById('shape-table-body').innerHTML = tableRowsHTML;

  // Client-side 1-second rolling average smoothing (window = 25 frames)
  let plotWidths = rawWidths;
  let plotDepths = rawDepths;

  if (gShapeMode === 'smoothed') {
    const windowSize = 25;
    plotWidths = smoothArray(rawWidths, windowSize);
    plotDepths = smoothArray(rawDepths, windowSize);
  }

  const traceWidth = {
    x: matchClockLabels,
    y: plotWidths.map(v => v.toFixed(2)),
    mode: 'lines',
    name: 'Team Width (Y-span)',
    line: { color: '#38bdf8', width: 2 }
  };

  const traceDepth = {
    x: matchClockLabels,
    y: plotDepths.map(v => v.toFixed(2)),
    mode: 'lines',
    name: 'Team Depth (X-span)',
    line: { color: '#f43f5e', width: 2 }
  };

  const layout = {
    title: { text: `Team Compactness Metrics Across Match Time (${gShapeMode.toUpperCase()} Mode)`, font: { size: 14, color: '#1A1A1A' } },
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#FAFAF8',
    xaxis: { title: 'Match Clock Time (MM:SS)', showgrid: true, gridcolor: '#E5E5E0', nticks: 10 },
    yaxis: { title: 'Distance (meters)', showgrid: true, gridcolor: '#E5E5E0' },
    annotations: [
      { x: '11:19', y: 15, text: 'Team briefly compressed (11:19 match time)', showarrow: true, arrowhead: 2, ax: 0, ay: -30, font: { size: 11, color: '#6B6B6B' } },
      { x: '12:48', y: 18, text: 'Fast transition (12:48 match time)', showarrow: true, arrowhead: 2, ax: 0, ay: -30, font: { size: 11, color: '#6B6B6B' } }
    ],
    margin: { l: 50, r: 20, t: 40, b: 50 },
    legend: { orientation: 'h', y: 1.12 }
  };

  Plotly.newPlot('plot-shape', [traceWidth, traceDepth], layout, { responsive: true });
}

function smoothArray(arr, windowSize) {
  let result = [];
  for (let i = 0; i < arr.length; i++) {
    let start = Math.max(0, i - Math.floor(windowSize / 2));
    let end = Math.min(arr.length, i + Math.ceil(windowSize / 2));
    let sub = arr.slice(start, end);
    let avg = sub.reduce((a, b) => a + b, 0) / sub.length;
    result.push(avg);
  }
  return result;
}

function setShapeMode(mode) {
  gShapeMode = mode;
  document.getElementById('shape-mode-smoothed').classList.toggle('active', mode === 'smoothed');
  document.getElementById('shape-mode-raw').classList.toggle('active', mode === 'raw');
  if (window.gRawDataset && window.gMetadata) {
    renderShapePlotData(window.gRawDataset, window.gMetadata);
  }
}
