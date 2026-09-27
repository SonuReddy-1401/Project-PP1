/**
 * Tactical Components & View Rendering Engine (Fixed Data Mapping & Roster Calculations)
 */

// Helper: Get current theme state
function isDarkTheme() {
  return document.body.getAttribute('data-theme') !== 'light';
}

// Helper: Extract positions array safely from frame object
function getFramePositions(frame) {
  if (!frame) return [];
  return frame.target_positions || frame.players || [];
}

// 1. Overview View
function renderOverviewView(meta, narratives) {
  const container = document.getElementById('view-overview');
  if (!container) return;

  const coachText = narratives.overview?.summary || "SSC Napoli controlled the match primarily through the middle third (57.4% occupancy), maintaining compact tactical shape and safe centroid displacement.";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Match Overview & Executive Command</div>
      <div class="page-sub">SSC Napoli vs AS Roma — 15-Minute Broadcast Sequence Analytics</div>
    </div>

    <!-- Executive Coach Summary Callout -->
    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Tactical Summary (Coach Insight)
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <!-- Key Stat Cards Grid -->
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-lbl">Target Squad</div>
        <div class="stat-val" style="color: var(--napoli-blue);">SSC Napoli</div>
        <div class="stat-badge">Sky Blue Outfield</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Match Time Covered</div>
        <div class="stat-val">9:54 – 24:54</div>
        <div class="stat-badge">15:00 min (22,500 frames)</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Primary Control Zone</div>
        <div class="stat-val" style="color: var(--tactical-gold);">Middle 3rd</div>
        <div class="stat-badge">57.4% Pitch Occupancy</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Safe Team Movement</div>
        <div class="stat-val">1.77 <span style="font-size: 16px; color: var(--text-muted);">km</span></div>
        <div class="stat-badge">7.13 km/h Average Pace</div>
      </div>
    </div>

    <!-- Interactive Match Timeline & Animation Player Card -->
    <div class="timeline-player-card">
      <div class="player-controls-row">
        <button id="btn-play-pause" class="btn-player-action">▶</button>
        <div class="scrubber-container">
          <input type="range" id="timeline-scrubber" class="scrubber-slider" min="0" max="22499" value="0">
          <div class="scrubber-meta">
            <span>Clip Offset: 9:54 match clock</span>
            <span id="player-clock-readout" style="color: var(--napoli-blue); font-weight: 700;">9:54</span>
            <span>End: 24:54</span>
          </div>
        </div>
        <div class="speed-selector">
          <button class="speed-btn" data-speed="0.5">0.5x</button>
          <button class="speed-btn active" data-speed="1.0">1.0x</button>
          <button class="speed-btn" data-speed="2.0">2.0x</button>
          <button class="speed-btn" data-speed="5.0">5.0x</button>
        </div>
      </div>
    </div>

    <!-- Live 2D Pitch Canvas Container -->
    <div class="pitch-container-card">
      <div style="width: 100%; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <span style="font-family: var(--font-display); font-weight: 700; font-size: 14px;">Live Tactical Pitch Canvas & Team Shape Polygon</span>
        <span style="font-size: 11px; color: var(--text-muted);">25 FPS Grounded Tracking</span>
      </div>
      <canvas id="pitch-canvas"></canvas>
    </div>
  `;

  // Initialize Pitch Canvas & Timeline Player for Overview
  window.gPitchCanvas = new ProPitchCanvas('pitch-canvas');
  if (window.gTimelinePlayer) {
    window.gTimelinePlayer.init(22500, 25.0, 594.0);
    window.gTimelinePlayer.addFrameCallback((frameIdx) => {
      if (window.gRawDataset && window.gRawDataset[frameIdx]) {
        window.gPitchCanvas.updateFrame(getFramePositions(window.gRawDataset[frameIdx]));
      }
    });
    // Trigger initial frame
    if (window.gRawDataset && window.gRawDataset[0]) {
      window.gPitchCanvas.updateFrame(getFramePositions(window.gRawDataset[0]));
    }
  }
}

// 2. Heatmap View
function renderHeatmapView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-heatmap');
  if (!container) return;

  const coachText = narratives.heatmap?.summary || "Napoli maintained dense occupancy across central midfield (-15m to +15m pitch length), forcing game flow through half-spaces.";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Team Spatial Heatmap</div>
      <div class="page-sub">2D Density Distribution of Squad Pitch Coordinates Over Match Clip</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Heatmap Insights
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="pitch-container-card">
      <div style="width: 100%; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <span style="font-family: var(--font-display); font-weight: 700; font-size: 14px;">Spatial Pitch Coordinates Density Contours</span>
        <span style="font-size: 11px; color: var(--tactical-gold); font-weight: 600;">215,988 Pitch Samples</span>
      </div>
      <canvas id="heatmap-pitch-canvas"></canvas>
    </div>
  `;

  const canvas = new ProPitchCanvas('heatmap-pitch-canvas');
  canvas.drawPitch();

  // Aggregate all pitch points for heatmap
  const allPoints = [];
  if (rawDataset) {
    for (let i = 0; i < rawDataset.length; i += 5) {
      const positions = getFramePositions(rawDataset[i]);
      for (let j = 0; j < positions.length; j++) {
        allPoints.push(positions[j]);
      }
    }
  }
  canvas.drawHeatmap(allPoints);
}

// 3. Team Shape View
function renderShapeView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-shape');
  if (!container) return;

  const coachText = narratives.shape?.summary || "Napoli's pitch width averaged 46.2 meters while depth averaged 38.9 meters. Notable tactical compacting occurred at 11:19 and 12:48 match clock.";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Team Shape & Structural Compactness</div>
      <div class="page-sub">Squad Pitch Width (Left-to-Right span) and Depth (Back-to-Front span) across match clock time</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Team Shape Summary
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="chart-card">
      <div id="plot-pro-shape" style="width: 100%; height: 460px;"></div>
    </div>

    <!-- Supplementary Complementary Visualizations Grid -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
      <div class="chart-card">
        <div id="plot-pro-area" style="width: 100%; height: 420px;"></div>
      </div>
      <div class="chart-card">
        <div id="plot-pro-phase" style="width: 100%; height: 420px;"></div>
      </div>
    </div>
  `;

  renderProShapePlot(rawDataset);
  renderProAreaPlot(rawDataset);
  renderProPhasePlot(rawDataset);
}

function renderProShapePlot(rawDataset) {
  if (!rawDataset || !rawDataset.length) return;

  const times = [];
  const widths = [];
  const depths = [];

  for (let i = 0; i < rawDataset.length; i += 25) { // 1 sample per second
    const frame = rawDataset[i];
    const positions = getFramePositions(frame);
    if (!positions || !positions.length) continue;

    let minX = 999, maxX = -999, minY = 999, maxY = -999;
    for (let p of positions) {
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      const py = Array.isArray(p) ? p[1] : (p.y !== undefined ? p.y : 0);
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    const totalSec = 594.0 + (i / 25.0);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);

    widths.push(Math.round((maxY - minY) * 10) / 10);
    depths.push(Math.round((maxX - minX) * 10) / 10);
  }

  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';

  const traceW = {
    x: times,
    y: widths,
    type: 'scatter',
    mode: 'lines',
    name: 'Team Width (Y-span)',
    line: { color: '#0080FF', width: 2 }
  };

  const traceD = {
    x: times,
    y: depths,
    type: 'scatter',
    mode: 'lines',
    name: 'Team Depth (X-span)',
    line: { color: '#F43F5E', width: 2 }
  };

  const layout = {
    title: { text: 'Team Compactness Metrics Across Match Time (1s Samples)', font: { size: 14, color: textColor, family: 'Outfit, sans-serif' } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { 
        text: 'Match Clock Time (MM:SS)', 
        font: { color: textColor, size: 12 },
        standoff: 25 
      }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 11 },
      tickangle: -45,
      nticks: 15
    },
    yaxis: { 
      title: { 
        text: 'Distance (meters)', 
        font: { color: textColor, size: 12 },
        standoff: 15 
      }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 11 } 
    },
    annotations: [
      { x: '11:19', y: 15, text: 'Team compressed (11:19)', showarrow: true, arrowhead: 2, ax: 0, ay: -25, font: { size: 11, color: '#F59E0B' } },
      { x: '12:48', y: 18, text: 'Fast transition (12:48)', showarrow: true, arrowhead: 2, ax: 0, ay: -25, font: { size: 11, color: '#F59E0B' } }
    ],
    margin: { l: 60, r: 30, t: 40, b: 85 },
    legend: { orientation: 'h', y: 1.12, font: { color: textColor } }
  };

  Plotly.newPlot('plot-pro-shape', [traceW, traceD], layout, { responsive: true });
}

// Render Tactical Pitch Area Envelope (m²) + Aspect Ratio Flow
function renderProAreaPlot(rawDataset) {
  if (!rawDataset || !rawDataset.length) return;

  const times = [];
  const areas = [];
  const aspectRatios = [];

  for (let i = 0; i < rawDataset.length; i += 25) { // 1s sample
    const frame = rawDataset[i];
    const positions = getFramePositions(frame);
    if (!positions || !positions.length) continue;

    let minX = 999, maxX = -999, minY = 999, maxY = -999;
    for (let p of positions) {
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      const py = Array.isArray(p) ? p[1] : (p.y !== undefined ? p.y : 0);
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    const totalSec = 594.0 + (i / 25.0);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);

    const w = Math.max(1, maxY - minY);
    const d = Math.max(1, maxX - minX);
    areas.push(Math.round(w * d));
    aspectRatios.push(Math.round((w / d) * 100) / 100);
  }

  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';

  const traceArea = {
    x: times,
    y: areas,
    type: 'scatter',
    mode: 'lines',
    name: 'Pitch Area (m²)',
    fill: 'tozeroy',
    fillcolor: isDark ? 'rgba(16, 185, 129, 0.18)' : 'rgba(16, 185, 129, 0.15)',
    line: { color: '#10B981', width: 2 }
  };

  const layout = {
    title: { text: 'Squad Pitch Area Envelope (m²) Over Match Clock', font: { size: 13, color: textColor, family: 'Outfit, sans-serif' } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { text: 'Match Clock (MM:SS)', font: { color: textColor, size: 11 }, standoff: 15 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 },
      tickangle: -45,
      nticks: 10
    },
    yaxis: { 
      title: { text: 'Tactical Area (m²)', font: { color: textColor, size: 11 }, standoff: 10 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    margin: { l: 55, r: 25, t: 40, b: 70 },
    legend: { orientation: 'h', y: 1.12, font: { color: textColor, size: 11 } }
  };

  Plotly.newPlot('plot-pro-area', [traceArea], layout, { responsive: true });
}

// Render 2D Width vs Depth Phase Space Matrix (Tactical State Map)
function renderProPhasePlot(rawDataset) {
  if (!rawDataset || !rawDataset.length) return;

  const depths = [];
  const widths = [];
  const times = [];

  for (let i = 0; i < rawDataset.length; i += 25) { // 1s sample
    const frame = rawDataset[i];
    const positions = getFramePositions(frame);
    if (!positions || !positions.length) continue;

    let minX = 999, maxX = -999, minY = 999, maxY = -999;
    for (let p of positions) {
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      const py = Array.isArray(p) ? p[1] : (p.y !== undefined ? p.y : 0);
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    const totalSec = 594.0 + (i / 25.0);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);

    depths.push(Math.round((maxX - minX) * 10) / 10);
    widths.push(Math.round((maxY - minY) * 10) / 10);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);
  }

  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';

  const tracePhase = {
    x: depths,
    y: widths,
    mode: 'markers',
    type: 'scatter',
    text: times,
    hovertemplate: '<b>Clock %{text}</b><br>Depth: %{x}m<br>Width: %{y}m<extra></extra>',
    marker: {
      size: 7,
      color: Array.from({ length: depths.length }, (_, k) => k),
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: { text: 'Time Progress', font: { size: 10, color: textColor } },
        tickfont: { color: textColor, size: 9 },
        thickness: 10,
        len: 0.8
      }
    }
  };

  const layout = {
    title: { text: 'Tactical Phase Space (Depth vs Width Matrix)', font: { size: 13, color: textColor, family: 'Outfit, sans-serif' } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { text: 'Team Depth X-span (meters)', font: { color: textColor, size: 11 }, standoff: 15 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    yaxis: { 
      title: { text: 'Team Width Y-span (meters)', font: { color: textColor, size: 11 }, standoff: 10 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    shapes: [
      // Median Tactical Box overlay
      { type: 'rect', x0: 30, y0: 35, x1: 45, y1: 55, line: { color: '#F59E0B', width: 1.5, dash: 'dash' }, fillcolor: 'rgba(245, 158, 11, 0.08)' }
    ],
    annotations: [
      { x: 37.5, y: 56, text: 'Median Tactical Zone', showarrow: false, font: { size: 10, color: '#F59E0B' } }
    ],
    margin: { l: 55, r: 25, t: 40, b: 65 }
  };

  Plotly.newPlot('plot-pro-phase', [tracePhase], layout, { responsive: true });
}

// 4. Pitch Thirds View
function renderThirdsView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-thirds');
  if (!container) return;

  const coachText = narratives.thirds?.summary || "Napoli dominated the Middle Third with 57.4% occupancy, maintaining defensive security in their own third (30.8%).";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Pitch-Thirds Occupancy Dominance</div>
      <div class="page-sub">Percentage of frames team centroid occupied Defensive, Middle, and Attacking pitch thirds</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Pitch Thirds Analysis
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="chart-card">
      <div id="plot-pro-thirds" style="width: 100%; height: 420px;"></div>
    </div>
  `;

  renderProThirdsPlot();
}

function renderProThirdsPlot() {
  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';

  const trace = {
    x: ['Defensive Third (-52.5m to -17.5m)', 'Middle Third (-17.5m to +17.5m)', 'Attacking Third (+17.5m to +52.5m)'],
    y: [30.8, 57.4, 11.8],
    type: 'bar',
    marker: { color: ['#10B981', '#0080FF', '#F59E0B'] },
    text: ['30.8%', '57.4%', '11.8%'],
    textposition: 'auto',
    textfont: { color: '#FFFFFF', family: 'Outfit, sans-serif', weight: 'bold' }
  };

  const layout = {
    title: { text: 'Whole Match Pitch-Thirds Centroid Occupancy (%)', font: { size: 14, color: textColor, family: 'Outfit, sans-serif' } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { tickfont: { color: textColor } },
    yaxis: { title: { text: 'Occupancy (%)', font: { color: textColor } }, range: [0, 80], gridcolor: gridColor, tickfont: { color: textColor } },
    margin: { l: 50, r: 20, t: 40, b: 50 }
  };

  Plotly.newPlot('plot-pro-thirds', [trace], layout, { responsive: true });
}

// 5. Trajectory View
function renderTrajectoryView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-trajectory');
  if (!container) return;

  const coachText = narratives.trajectory?.summary || "Team centroid displacement total is 1.77 km (7.13 km/h average pace) using physical outlier rejection (< 10 m/s ceiling).";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Safe Team Centroid Trajectory</div>
      <div class="page-sub">Flow vector paths of team movement across 15-minute clip with speed outlier rejection</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Trajectory Diagnostics
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-lbl">Safe Team Displacement</div>
        <div class="stat-val" style="color: var(--napoli-blue);">1.77 km</div>
        <div class="stat-badge">Physical Speed Ceiling: 10 m/s</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Interval Rejection Rate</div>
        <div class="stat-val" style="color: var(--tactical-gold);">47.8%</div>
        <div class="stat-badge">Camera Panning Spike Correlation</div>
      </div>
    </div>
  `;
}

// 6. Roster Analytics View
function renderRosterView(trackedDataset, meta, narratives) {
  const container = document.getElementById('view-players');
  if (!container) return;

  const coachText = narratives.players?.summary || "Player tracking provides continuous mobility snapshots for continuous tracking segments (averaging 10 to 18 seconds).";

  // Build tracked segments breakdown dynamically if dataset is array of frames
  let trackerStats = {};
  if (trackedDataset && trackedDataset.length) {
    for (let f of trackedDataset) {
      if (f.tracker_ids && f.target_positions) {
        for (let i = 0; i < f.tracker_ids.length; i++) {
          const tid = f.tracker_ids[i];
          const pos = f.target_positions[i];
          if (!pos) continue;
          if (!trackerStats[tid]) {
            trackerStats[tid] = { id: tid, firstFrame: f.frame_idx, count: 0, lastPos: null, distM: 0 };
          }
          let px = Array.isArray(pos) ? pos[0] : pos.x;
          let py = Array.isArray(pos) ? pos[1] : pos.y;
          if (trackerStats[tid].lastPos) {
            let dx = px - trackerStats[tid].lastPos[0];
            let dy = py - trackerStats[tid].lastPos[1];
            let step = Math.sqrt(dx * dx + dy * dy);
            if (step < 0.4) { // safe threshold per frame at 25fps (10 m/s)
              trackerStats[tid].distM += step;
            }
          }
          trackerStats[tid].lastPos = [px, py];
          trackerStats[tid].count++;
        }
      }
    }
  }

  let tableRows = '';
  const sortedTracks = Object.values(trackerStats).sort((a, b) => b.count - a.count).slice(0, 12);

  sortedTracks.forEach(t => {
    const durSec = (t.count / 25.0).toFixed(1);
    const min = Math.floor(t.firstFrame / (25 * 60)) + 9;
    const sec = Math.floor((t.firstFrame / 25) % 60);
    const startClock = `${min}:${sec.toString().padStart(2, '0')}`;
    const speedKmh = durSec > 0 ? ((t.distM / parseFloat(durSec)) * 3.6).toFixed(1) : '0.0';

    tableRows += `
      <tr>
        <td><strong style="color: var(--napoli-blue);">Player Tracker #${t.id}</strong></td>
        <td>${startClock} (${durSec}s total duration)</td>
        <td>${t.distM.toFixed(1)} m</td>
        <td>${speedKmh} km/h avg pace</td>
        <td><span style="color: var(--tactical-emerald);">Continuous Tracked</span></td>
      </tr>
    `;
  });

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Individual Tracked Player Analytics</div>
      <div class="page-sub">Continuous tracking segments and mobility metrics for outfield players</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Player Mobility Summary
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="table-card">
      <div class="chart-header">
        <div class="chart-title">Top Continuous Player Tracking Segments</div>
      </div>
      <div class="pro-table-wrapper">
        <table class="pro-table">
          <thead>
            <tr>
              <th>Tracker ID</th>
              <th>Segment Start (Match Clock)</th>
              <th>Safe Distance Covered</th>
              <th>Implied Speed</th>
              <th>Tracking Status</th>
            </tr>
          </thead>
          <tbody>
            ${tableRows || '<tr><td colspan="5">No tracking segments loaded.</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// 7. Audit & Methodology View
function renderAuditView(meta, narratives) {
  const container = document.getElementById('view-audit');
  if (!container) return;

  const coachText = narratives.methodology?.summary || "Distance metrics reject impossible frame jumps using 10 m/s threshold. Spike analysis ties 47.8% interval rejections to PnLCalib reprojection errors during rapid camera pans.";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Data Quality & Pipeline Methodology Audit</div>
      <div class="page-sub">PnL Calibration error diagnostics, homography reprojection error, and outlier rejection models</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Methodology Summary
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-lbl">PnL Calibration Rate</div>
        <div class="stat-val" style="color: var(--tactical-emerald);">100.0%</div>
        <div class="stat-badge">22,500 / 22,500 Frames Solved</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Mean Reprojection Error</div>
        <div class="stat-val">3.37 <span style="font-size: 16px; color: var(--text-muted);">px</span></div>
        <div class="stat-badge">Pan Spike Max: 9.66 px</div>
      </div>
    </div>
  `;
}
