/**
 * Component: Individual Player Tracking Analytics
 */
let gSelectedPlayerId = null;

function renderPlayersComponent(trackedDataset, metadata, narratives) {
  const container = document.getElementById('players');
  if (!container) return;

  const coachText = narratives.individual_players || "Individual player tracking explanation for head coach.";

  container.innerHTML = `
    <h1 class="section-title">Individual Tracked Player Analytics</h1>
    <p class="section-subtitle">Per-player heatmaps and safe distance metrics for continuous tracking segments.</p>

    <!-- Coach Callout Box -->
    <div class="coach-callout">
      <div class="coach-callout-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        What am I looking at (Coach Summary)
      </div>
      <div class="coach-callout-text">${coachText}</div>
    </div>

    <!-- Permanent Explanatory Banner -->
    <div class="info-banner">
      <strong style="color: var(--text-primary);">Tracking Limitation Notice:</strong>
      The tracking system loses and re-identifies players during camera cuts and crowded moments, so each entry below is a continuous tracking segment for one player — not a full match history. Longer segments are more reliable. This is a known limitation of broadcast camera tracking.
    </div>

    <div class="card">
      <div class="controls-row">
        <div style="font-size: 13px; color: var(--text-secondary); font-weight: 600;">Select Tracking Segment:</div>
        <select id="player-select" onchange="updatePlayerView()" style="padding: 8px 14px; font-size: 13px; border-radius: 4px; border: 1px solid var(--card-border); max-width: 460px;">
        </select>
      </div>

      <div class="stats-grid" style="margin-bottom: 24px;">
        <div class="stat-card">
          <div class="stat-label">Segment Duration</div>
          <div class="stat-value" id="player-stat-frames">0</div>
          <div class="stat-sub" id="player-stat-time">0.0 sec active</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">
            Safe Distance Covered
            <span style="font-size:10px; background:#E5E5E0; color:#6B6B6B; border-radius:50%; width:14px; height:14px; display:inline-flex; align-items:center; justify-content:center; cursor:help; margin-left:4px;" title="Calculated using outlier-rejected frame-to-frame displacement (<10 m/s ceiling).">?</span>
          </div>
          <div class="stat-value" id="player-stat-dist">0.0<span class="stat-unit">m</span></div>
          <div class="stat-sub">Implied Speed &lt; 36 km/h</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">Interval Rejection Rate</div>
          <div class="stat-value" id="player-stat-rej">0.0<span class="stat-unit">%</span></div>
          <div class="stat-sub" id="player-stat-rej-count">0 / 0 intervals rejected</div>
        </div>
      </div>

      <div id="plot-player-heatmap" class="plot-container"></div>

      <div class="data-table-container">
        <button class="btn" onclick="toggleRawTable('player-table-wrap')">Show Raw Player Track Data</button>
        <button class="btn" onclick="exportCSV('player-table', 'individual_player_track.csv')" style="margin-left: 8px;">Export CSV</button>

        <div id="player-table-wrap" class="table-wrapper" style="display: none;">
          <table id="player-table">
            <thead>
              <tr>
                <th>Match Time</th>
                <th>Frame Index</th>
                <th>Pitch X (m)</th>
                <th>Pitch Y (m)</th>
              </tr>
            </thead>
            <tbody id="player-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  renderPlayersPlotData(trackedDataset, metadata);
}

function renderPlayersPlotData(trackedDataset, metadata) {
  if (!trackedDataset.length) return;

  const fps = metadata.fps || 25.0;
  const clipStart = metadata.clip_start_offset_sec || 594.0;
  const playerMap = {};

  trackedDataset.forEach(entry => {
    if (!entry.calibration_ok || !entry.target_positions || !entry.tracker_ids) return;
    entry.target_positions.forEach((pt, idx) => {
      const tId = entry.tracker_ids[idx];
      if (tId === -1) return;
      if (!playerMap[tId]) playerMap[tId] = { pts: [], frameIndices: [] };
      playerMap[tId].pts.push(pt);
      playerMap[tId].frameIndices.push(entry.frame_idx);
    });
  });

  // Filter segments with min 200 frames and sort by segment duration (longest first)
  const segments = Object.keys(playerMap)
    .filter(tId => playerMap[tId].pts.length >= 200)
    .map(tId => {
      const frames = playerMap[tId].frameIndices;
      const startClock = frameToTimestamp(frames[0], fps, clipStart);
      const endClock = frameToTimestamp(frames[frames.length - 1], fps, clipStart);
      const durationSec = (frames.length / fps).toFixed(0);
      const minVal = Math.floor(durationSec / 60);
      const secVal = durationSec % 60;
      const durationText = minVal > 0 ? `${minVal}m ${secVal}s` : `${secVal}s`;
      return {
        id: tId,
        count: frames.length,
        label: `Tracking segment: ${startClock}–${endClock} (${durationText} continuous)`
      };
    })
    .sort((a, b) => b.count - a.count);

  const selectEl = document.getElementById('player-select');
  if (selectEl) {
    selectEl.innerHTML = segments.map(s => `<option value="${s.id}">${s.label}</option>`).join('');
  }

  gSelectedPlayerId = selectEl.value || segments[0].id;
  const pData = playerMap[gSelectedPlayerId];

  if (!pData) return;

  let pDist = 0, pRej = 0, pTotalInt = 0;
  let tableRowsHTML = '';

  for (let i = 1; i < pData.pts.length; i++) {
    if (pData.frameIndices[i] === pData.frameIndices[i-1] + 1) {
      const disp = Math.sqrt((pData.pts[i][0] - pData.pts[i-1][0])**2 + (pData.pts[i][1] - pData.pts[i-1][1])**2);
      const speed = disp * fps;
      pTotalInt++;
      if (speed > 10.0) pRej++;
      else pDist += disp;
    }

    if (i < 100) {
      const clock = frameToTimestamp(pData.frameIndices[i], fps, clipStart);
      tableRowsHTML += `<tr><td style="font-weight:700;">${clock}</td><td style="color:#6b6b6b;">${pData.frameIndices[i]}</td><td>${pData.pts[i][0]}</td><td>${pData.pts[i][1]}</td></tr>`;
    }
  }

  const startClock = frameToTimestamp(pData.frameIndices[0], fps, clipStart);
  const endClock = frameToTimestamp(pData.frameIndices[pData.frameIndices.length - 1], fps, clipStart);
  const activeSec = (pData.pts.length / fps).toFixed(1);
  const rejPct = pTotalInt > 0 ? ((pRej / pTotalInt) * 100).toFixed(1) : '0.0';

  document.getElementById('player-stat-frames').innerText = `${activeSec}s`;
  document.getElementById('player-stat-time').innerText = `${pData.pts.length} frames (${startClock} – ${endClock} match time)`;
  document.getElementById('player-stat-dist').innerHTML = `${pDist.toFixed(1)}<span class="stat-unit">m</span>`;
  document.getElementById('player-stat-rej').innerHTML = `${rejPct}<span class="stat-unit">%</span>`;
  document.getElementById('player-stat-rej-count').innerText = `${pRej} / ${pTotalInt} intervals rejected`;
  document.getElementById('player-table-body').innerHTML = tableRowsHTML;

  const xPts = pData.pts.map(p => p[0]);
  const yPts = pData.pts.map(p => p[1]);

  const data = [{
    x: xPts,
    y: yPts,
    type: 'histogram2dcontour',
    colorscale: 'Plasma',
    ncontours: 10,
    opacity: 0.7
  }];

  const layout = getPitchLayout(`Player Segment Heatmap (${startClock} – ${endClock} | ${pDist.toFixed(1)}m safe dist)`);
  Plotly.newPlot('plot-player-heatmap', data, layout, { responsive: true });
}

function updatePlayerView() {
  if (window.gTrackedDataset && window.gMetadata) {
    gSelectedPlayerId = document.getElementById('player-select').value;
    renderPlayersPlotData(window.gTrackedDataset, window.gMetadata);
  }
}
