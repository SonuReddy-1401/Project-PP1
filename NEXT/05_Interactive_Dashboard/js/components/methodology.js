/**
 * Component: Data Quality & Methodology (Two-Layer Progressive Disclosure)
 */
function renderMethodologyComponent(metadata, narratives) {
  const container = document.getElementById('methodology');
  if (!container) return;

  const coachText = narratives.data_quality || "Data quality explanation for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Data Quality & Engineering Methodology</h1>
    <p class="section-subtitle">Methodological transparency, error metrics, temporal noise clustering, and pipeline caveats.</p>

    <!-- Coach Callout Box -->
    <div class="coach-callout">
      <div class="coach-callout-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        What am I looking at (Coach Summary)
      </div>
      <div class="coach-callout-text">${coachText}</div>
    </div>

    <!-- Two-Layer Progressive Disclosure Cards -->
    <div class="stats-grid" style="margin-bottom: 24px;">
      <div class="stat-card">
        <div class="stat-label">Camera Calibration</div>
        <div style="font-size: 13.5px; font-weight: 500; color: var(--text-primary); margin-bottom: 8px;">
          The system successfully calculated the camera position for <strong>100% of frames</strong> — a strong, reliable result.
        </div>
        <button class="btn" style="padding: 4px 8px; font-size: 11px;" onclick="toggleRawTable('tech-detail-calib')">Show technical detail</button>
        <div id="tech-detail-calib" class="technical-drawer" style="display: none;">
          22,500 / 22,500 frames solved via PnLCalib. Mean reprojection error: 3.42 px across stable windows.
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Player Tracking IDs</div>
        <div style="font-size: 13.5px; font-weight: 500; color: var(--text-primary); margin-bottom: 8px;">
          Frequent camera cuts break continuous tracking into short snapshots (averaging 10–18 seconds each).
        </div>
        <button class="btn" style="padding: 4px 8px; font-size: 11px;" onclick="toggleRawTable('tech-detail-track')">Show technical detail</button>
        <div id="tech-detail-track" class="technical-drawer" style="display: none;">
          847 unique ByteTrack IDs generated vs 11 squad size due to broadcast cuts and occlusion.
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Distance Filter Rejections</div>
        <div style="font-size: 13.5px; font-weight: 500; color: var(--text-primary); margin-bottom: 8px;">
          Speed spikes from fast camera pans (over 36 km/h) were safely thrown out to keep distances accurate.
        </div>
        <button class="btn" style="padding: 4px 8px; font-size: 11px;" onclick="toggleRawTable('tech-detail-dist')">Show technical detail</button>
        <div id="tech-detail-dist" class="technical-drawer" style="display: none;">
          49.5% rejection rate (11,063 / 22,338 intervals >10 m/s ceiling filtered in distance_utils.py).
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Temporal Rejection Rate vs. PnL Reprojection Error (30s Windows)</div>
      <div id="plot-methodology-windows" class="plot-container" style="height: 380px;"></div>
    </div>

    <div class="card">
      <div class="card-title">Pipeline Audit Summary</div>
      <table>
        <thead>
          <tr>
            <th>Component</th>
            <th>Status / Performance</th>
            <th>Known Limitation & Engineering Mitigation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Camera Calibration</strong></td>
            <td>100.0% Solved (PnLCalib)</td>
            <td>Momentary reprojection error spikes during rapid camera pans (e.g. Window 24 rep_err = 9.66px). Mitigated via out-of-bounds polygon filtering.</td>
          </tr>
          <tr>
            <td><strong>Player Tracking</strong></td>
            <td>847 Unique ByteTrack IDs</td>
            <td>Frequent ID churn across broadcast camera cuts. Mitigated via 200-frame persistence threshold for individual heatmaps.</td>
          </tr>
          <tr>
            <td><strong>Distance Calculation</strong></td>
            <td>49.5% Centroid Rejection Rate</td>
            <td>Raw Euclidean sum vulnerable to 96 km/h single-frame jumps. Mitigated via 10.0 m/s (36 km/h) speed ceiling filtering.</td>
          </tr>
        </tbody>
      </table>
    </div>
  `;

  renderMethodologyPlotData();
}

function renderMethodologyPlotData() {
  const windowNumbers = Array.from({length: 30}, (_, i) => `W${i+1}`);
  const rejectionRates = [57.7, 45.2, 46.1, 57.1, 41.3, 47.8, 54.7, 64.4, 44.5, 50.9, 42.8, 56.3, 51.5, 37.5, 49.7, 42.9, 65.2, 53.5, 38.6, 44.8, 54.0, 61.2, 40.4, 66.4, 41.3, 39.7, 50.4, 54.2, 50.3, 38.0];
  const repErrors = [3.26, 3.33, 3.15, 3.74, 3.11, 3.16, 3.20, 2.94, 3.51, 3.09, 3.11, 3.33, 3.54, 3.13, 3.22, 3.20, 3.17, 3.20, 3.82, 3.07, 4.67, 2.95, 3.07, 9.66, 3.41, 3.41, 3.36, 2.43, 3.20, 3.19];

  const traceRej = {
    x: windowNumbers,
    y: rejectionRates,
    type: 'bar',
    name: 'Rejection Rate (%)',
    marker: { color: '#f43f5e' }
  };

  const traceErr = {
    x: windowNumbers,
    y: repErrors,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Reprojection Error (px)',
    yaxis: 'y2',
    line: { color: '#38bdf8', width: 2 }
  };

  const layout = {
    title: { text: '30-Second Windowed Rejection Rate vs PnL Reprojection Error (rep_err)', font: { size: 13, color: '#1A1A1A' } },
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#FAFAF8',
    xaxis: { title: '30-Second Match Time Windows' },
    yaxis: { title: 'Rejection Rate (%)', range: [0, 80], gridcolor: '#E5E5E0' },
    yaxis2: { title: 'Reprojection Error (px)', overlaying: 'y', side: 'right', range: [0, 12] },
    margin: { l: 50, r: 50, t: 40, b: 50 },
    legend: { orientation: 'h', y: 1.15 }
  };

  Plotly.newPlot('plot-methodology-windows', [traceRej, traceErr], layout, { responsive: true });
}
