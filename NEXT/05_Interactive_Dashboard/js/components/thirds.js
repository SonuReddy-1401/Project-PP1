/**
 * Component: Pitch Thirds Spatial Occupancy
 */
let gThirdsWindow = 'all'; // 'all', 'w1' (0-5m), 'w2' (5-10m), 'w3' (10-15m)

function renderThirdsComponent(rawDataset, metadata, narratives) {
  const container = document.getElementById('thirds');
  if (!container) return;

  const coachText = narratives.pitch_thirds || "Pitch-thirds explanation for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Pitch-Thirds Spatial Occupancy</h1>
    <p class="section-subtitle">Percentage of match time the team spent controlling Defensive, Middle, and Attacking thirds.</p>

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
        <span style="font-size: 12px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;">Match Segment Window:</span>
        <button class="btn btn-toggle active" id="thirds-btn-all" onclick="setThirdsWindow('all')">Whole Match (9:54–24:54)</button>
        <button class="btn btn-toggle" id="thirds-btn-w1" onclick="setThirdsWindow('w1')">0–5 min (9:54–14:54)</button>
        <button class="btn btn-toggle" id="thirds-btn-w2" onclick="setThirdsWindow('w2')">5–10 min (14:54–19:54)</button>
        <button class="btn btn-toggle" id="thirds-btn-w3" onclick="setThirdsWindow('w3')">10–15 min (19:54–24:54)</button>
      </div>

      <div id="plot-thirds" class="plot-container"></div>

      <div class="data-table-container">
        <button class="btn" onclick="toggleRawTable('thirds-table-wrap')">Show Raw Thirds Breakdown</button>
        <button class="btn" onclick="exportCSV('thirds-table', 'pitch_thirds_occupancy.csv')" style="margin-left: 8px;">Export CSV</button>

        <div id="thirds-table-wrap" class="table-wrapper" style="display: none;">
          <table id="thirds-table">
            <thead>
              <tr>
                <th>Pitch Zone</th>
                <th>X Coordinate Bounds (m)</th>
                <th>Frame Count</th>
                <th>Occupancy Percentage (%)</th>
              </tr>
            </thead>
            <tbody id="thirds-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  renderThirdsPlotData(rawDataset, metadata);
}

function renderThirdsPlotData(rawDataset, metadata) {
  if (!rawDataset.length) return;

  let startSec = 0;
  let endSec = 900;
  let windowLabel = "Whole Match (9:54–24:54)";

  if (gThirdsWindow === 'w1') { startSec = 0; endSec = 300; windowLabel = "0–5 min (9:54–14:54)"; }
  if (gThirdsWindow === 'w2') { startSec = 300; endSec = 600; windowLabel = "5–10 min (14:54–19:54)"; }
  if (gThirdsWindow === 'w3') { startSec = 600; endSec = 900; windowLabel = "10–15 min (19:54–24:54)"; }

  let defCnt = 0, midCnt = 0, attCnt = 0;
  let totalValid = 0;

  rawDataset.forEach(entry => {
    if (!entry.calibration_ok || !entry.target_positions || !entry.target_positions.length) return;
    const t = entry.timestamp_sec;
    if (t < startSec || t > endSec) return;

    const xs = entry.target_positions.map(p => p[0]);
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;

    if (meanX < -17.5) defCnt++;
    else if (meanX <= 17.5) midCnt++;
    else attCnt++;
    totalValid++;
  });

  const defPct = totalValid > 0 ? ((defCnt / totalValid) * 100).toFixed(1) : '0.0';
  const midPct = totalValid > 0 ? ((midCnt / totalValid) * 100).toFixed(1) : '0.0';
  const attPct = totalValid > 0 ? ((attCnt / totalValid) * 100).toFixed(1) : '0.0';

  document.getElementById('thirds-table-body').innerHTML = `
    <tr><td>Defensive Third</td><td>X &lt; -17.5m</td><td>${defCnt}</td><td>${defPct}%</td></tr>
    <tr><td>Middle Third</td><td>-17.5m &le; X &le; 17.5m</td><td>${midCnt}</td><td>${midPct}%</td></tr>
    <tr><td>Attacking Third</td><td>X &gt; 17.5m</td><td>${attCnt}</td><td>${attPct}%</td></tr>
  `;

  const data = [{
    x: ['Defensive Third<br>(X < -17.5m)', 'Middle Third<br>(-17.5m ≤ X ≤ 17.5m)', 'Attacking Third<br>(X > 17.5m)'],
    y: [defPct, midPct, attPct],
    type: 'bar',
    marker: { color: ['#f43f5e', '#38bdf8', '#22c55e'] },
    text: [defPct + '%', midPct + '%', attPct + '%'],
    textposition: 'auto'
  }];

  const isDark = isDarkMode();
  const paperBg = isDark ? '#1E293B' : '#FFFFFF';
  const plotBg = isDark ? '#0F172A' : '#FAFAF8';
  const textColor = isDark ? '#F8FAFC' : '#1A1A1A';
  const gridColor = isDark ? '#334155' : '#E5E5E0';

  const layout = {
    title: { text: `Pitch-Thirds Team Centroid Occupancy — ${windowLabel}`, font: { size: 14, color: textColor } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { tickfont: { color: textColor } },
    yaxis: { title: { text: 'Occupancy Percentage (%)', font: { color: textColor } }, range: [0, 80], gridcolor: gridColor, tickfont: { color: textColor } },
    margin: { l: 50, r: 20, t: 40, b: 50 }
  };

  Plotly.newPlot('plot-thirds', data, layout, { responsive: true });
}

function setThirdsWindow(winKey) {
  gThirdsWindow = winKey;
  document.querySelectorAll('#thirds .btn-toggle').forEach(b => b.classList.remove('active'));
  document.getElementById(`thirds-btn-${winKey}`).classList.add('active');
  if (window.gRawDataset && window.gMetadata) {
    renderThirdsPlotData(window.gRawDataset, window.gMetadata);
  }
}
