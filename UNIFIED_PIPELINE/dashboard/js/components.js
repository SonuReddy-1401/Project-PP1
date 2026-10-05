/**
 * Dynamic View Rendering Engine for Unified Pipeline Dashboard
 * Adapts dynamically to metadata, match clock start offset, fps, and kit color.
 */

function isDarkTheme() {
  return document.body.getAttribute('data-theme') !== 'light';
}

function getFramePositions(frame) {
  if (!frame) return [];
  return frame.target_positions || frame.players || [];
}

// 1. Overview View
function renderOverviewView(meta, narratives) {
  const container = document.getElementById('view-overview');
  if (!container) return;

  const stats = meta.stats || {};
  const teamName = meta.team_name || "Target Squad";
  const oppName = meta.opponent_name || "Opponent";
  const teamColor = meta.team_color || "#0080FF";

  const thirds = stats.thirds_pct || {};
  const defPct = thirds.defensive || 0.0;
  const midPct = thirds.middle || 50.0;
  const attPct = thirds.attacking || 0.0;
  let domThird = "Middle Third", domVal = midPct;
  if (defPct > midPct && defPct > attPct) { domThird = "Defensive Third"; domVal = defPct; }
  else if (attPct > midPct && attPct > defPct) { domThird = "Attacking Third"; domVal = attPct; }

  const widthM = stats.avg_width_m || 41.5;
  const compactM = stats.avg_compactness_m || stats.avg_depth_m || 30.1;
  const lineH = stats.avg_line_height_m || 58.2;

  const coachText = `${teamName} controlled the match sequence primarily through the ${domThird} (${domVal}% occupancy), maintaining an average pitch width of ${widthM}m, vertical compactness of ${compactM}m, and defensive line height of ${lineH}m from own goal.`;

  const durationMin = (stats.duration_sec / 60.0).toFixed(2);
  const startClock = meta.match_start_clock || "00:00";

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Match Overview & Executive Command</div>
      <div class="page-sub">${teamName} vs ${oppName} — Dynamic Tactical Analytics</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Tactical Summary (Coach Insight)
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-lbl">Target Squad</div>
        <div class="stat-val" style="color: ${teamColor};">${teamName}</div>
        <div class="stat-badge">${meta.team_badge || 'Active Squad'}</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Clip Match Time</div>
        <div class="stat-val">${startClock} Start</div>
        <div class="stat-badge">${durationMin} min (${stats.total_frames} frames)</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Middle Third Control</div>
        <div class="stat-val" style="color: var(--tactical-gold);">${stats.thirds_pct?.middle || 50.0}%</div>
        <div class="stat-badge">Pitch Occupancy</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Safe Team Movement</div>
        <div class="stat-val">${stats.safe_centroid_dist_km || 0} <span style="font-size: 16px; color: var(--text-muted);">km</span></div>
        <div class="stat-badge">${stats.avg_pace_kmh || 0} km/h Average Pace</div>
      </div>
    </div>

    <div class="timeline-player-card">
      <div class="player-controls-row">
        <button id="btn-play-pause" class="btn-player-action">▶</button>
        <div class="scrubber-container">
          <input type="range" id="timeline-scrubber" class="scrubber-slider" min="0" max="${stats.total_frames - 1}" value="0">
          <div class="scrubber-meta">
            <span>Clip Offset: ${startClock} match clock</span>
            <span id="player-clock-readout" style="color: ${teamColor}; font-weight: 700;">${startClock}</span>
            <span>Duration: ${durationMin}m</span>
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

    <div class="pitch-container-card">
      <div style="width: 100%; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <span style="font-family: var(--font-display); font-weight: 700; font-size: 14px;">Live Tactical Pitch Canvas & Team Shape Polygon</span>
        <span style="font-size: 11px; color: var(--text-muted);">${meta.fps || 25} FPS Grounded Tracking</span>
      </div>
      <canvas id="pitch-canvas"></canvas>
    </div>
  `;

  window.gPitchCanvas = new ProPitchCanvas('pitch-canvas');
  if (window.gTimelinePlayer) {
    window.gTimelinePlayer.init(meta.total_frames || 1000, meta.fps || 25.0, meta.clip_start_offset_sec || 0.0);
    window.gTimelinePlayer.addFrameCallback((frameIdx) => {
      if (window.gRawDataset && window.gRawDataset[frameIdx]) {
        window.gPitchCanvas.updateFrame(getFramePositions(window.gRawDataset[frameIdx]));
      }
    });
    if (window.gRawDataset && window.gRawDataset[0]) {
      window.gPitchCanvas.updateFrame(getFramePositions(window.gRawDataset[0]));
    }
  }
}

// 2. Heatmap View
function renderHeatmapView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-heatmap');
  if (!container) return;

  const coachText = narratives.heatmap?.summary || "Spatial heatmap demonstrates pitch occupancy intensity contours.";

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
        <span style="font-size: 11px; color: var(--tactical-gold); font-weight: 600;">${meta.stats?.total_samples || 0} Pitch Samples</span>
      </div>
      <canvas id="heatmap-pitch-canvas"></canvas>
    </div>
  `;

  const canvas = new ProPitchCanvas('heatmap-pitch-canvas');
  canvas.drawPitch();

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

  const stats = meta.stats || {};
  const coachText = narratives.shape?.summary || `Pitch width averaged ${stats.avg_width_m || 41.5} meters while defensive line height averaged ${stats.avg_depth_m || 35.0} meters from own goal.`;
  const attackingDir = meta.attacking_direction || meta.attacking_dir || 'left_to_right';
  const attackingTag = attackingDir.replace('_', ' ').toUpperCase();

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Team Shape & Structural Compactness</div>
      <div class="page-sub">Dedicated Width (Y-span), Defensive Line Height (0–105m from own goal), and Vertical Compactness (X-span) Analysis</div>
    </div>

    <div class="coach-summary-box">
      <div class="coach-title-tag">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Team Shape Summary
      </div>
      <div class="coach-text-body">${coachText}</div>
    </div>

    <div style="display: flex; flex-direction: column; gap: 24px;">
      <div class="chart-card" style="padding: 20px;">
        <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 600; color: var(--text-primary, #F1F5F9); margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span>Team Pitch Width (Y-span)</span>
          <span style="font-size: 11px; font-weight: 600; color: #38BDF8; background: rgba(56, 189, 248, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.3);">Touchline Span: 0–68m</span>
        </div>
        <div id="plot-pro-width" style="width: 100%; height: 380px;"></div>
      </div>

      <div class="chart-card" style="padding: 20px;">
        <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 600; color: var(--text-primary, #F1F5F9); margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span>Defensive Line & Pitch Block Height</span>
          <span style="font-size: 11px; font-weight: 600; color: #F59E0B; background: rgba(245, 158, 11, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(245, 158, 11, 0.3);">Attacking: ${attackingTag}</span>
        </div>
        <div id="plot-pro-line-height" style="width: 100%; height: 380px;"></div>
      </div>

      <div class="chart-card" style="padding: 20px;">
        <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 600; color: var(--text-primary, #F1F5F9); margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span>Squad Vertical Compactness (X-span)</span>
          <span style="font-size: 11px; font-weight: 600; color: #EC4899; background: rgba(236, 72, 153, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(236, 72, 153, 0.3);">Back-to-Front Depth</span>
        </div>
        <div id="plot-pro-compactness" style="width: 100%; height: 380px;"></div>
      </div>

      <div class="chart-card" style="padding: 20px;">
        <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 600; color: var(--text-primary, #F1F5F9); margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span>Squad Pitch Area Envelope (m²)</span>
          <span style="font-size: 11px; font-weight: 600; color: #A855F7; background: rgba(168, 85, 247, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(168, 85, 247, 0.3);">Spatial Footprint</span>
        </div>
        <div id="plot-pro-area" style="width: 100%; height: 380px;"></div>
      </div>

      <div class="chart-card" style="padding: 20px;">
        <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 600; color: var(--text-primary, #F1F5F9); margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span>Tactical Phase Space (Depth vs Width Matrix)</span>
          <span style="font-size: 11px; font-weight: 600; color: #10B981; background: rgba(16, 185, 129, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.3);">Structural Clustering</span>
        </div>
        <div id="plot-pro-phase" style="width: 100%; height: 420px;"></div>
      </div>
    </div>
  `;

  renderProWidthPlot(rawDataset, meta);
  renderProLineHeightPlot(rawDataset, meta);
  renderProCompactnessPlot(rawDataset, meta);
  renderProAreaPlot(rawDataset, meta);
  renderProPhasePlot(rawDataset, meta);
}

function renderProWidthPlot(rawDataset, meta) {
  if (!rawDataset || !rawDataset.length) return;

  const times = [];
  const widths = [];
  const startSec = meta.clip_start_offset_sec || 0.0;
  const fps = meta.fps || 25.0;

  for (let i = 0; i < rawDataset.length; i += Math.round(fps)) {
    const frame = rawDataset[i];
    const positions = getFramePositions(frame);
    if (!positions || !positions.length) continue;

    let minY = 999, maxY = -999;
    for (let p of positions) {
      const py = Array.isArray(p) ? p[1] : (p.y !== undefined ? p.y : 0);
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    const totalSec = startSec + (i / fps);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);
    widths.push(Math.round((maxY - minY) * 10) / 10);
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
    fill: 'tozeroy',
    fillcolor: isDark ? 'rgba(56, 189, 248, 0.12)' : 'rgba(56, 189, 248, 0.15)',
    line: { color: '#38BDF8', width: 2.5 } // High-contrast Electric Cyan
  };

  const layout = {
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { text: 'Match Clock (MM:SS)', font: { color: textColor, size: 11 }, standoff: 15 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 },
      tickangle: -45,
      nticks: 15
    },
    yaxis: { 
      title: { text: 'Width (meters)', font: { color: textColor, size: 11 }, standoff: 10 }, 
      range: [0, 68.0], // Fixed bound relative to actual physical pitch width
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    shapes: [
      {
        type: 'line',
        x0: times[0],
        x1: times[times.length - 1],
        y0: 68.0,
        y1: 68.0,
        line: { color: 'rgba(56, 189, 248, 0.5)', width: 1.5, dash: 'dash' }
      }
    ],
    annotations: [
      {
        x: times[Math.floor(times.length / 2)],
        y: 65.0,
        text: 'Full Pitch Touchline Limit (68.0m)',
        showarrow: false,
        font: { size: 9, color: '#38BDF8' }
      }
    ],
    margin: { l: 55, r: 25, t: 25, b: 60 },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { color: textColor, size: 10 } }
  };

  Plotly.newPlot('plot-pro-width', [traceW], layout, { responsive: true });
}

// 1. Defensive Line & Pitch Block Height Graph (0-105m distance from own goal line)
function renderProLineHeightPlot(rawDataset, meta) {
  if (!rawDataset || !rawDataset.length) return;

  const times = [];
  const lineHeights = [];
  const centroidHeights = [];
  const startSec = meta.clip_start_offset_sec || 0.0;
  const fps = meta.fps || 25.0;
  const attackingDir = meta.attacking_direction || meta.attacking_dir || 'left_to_right';

  for (let i = 0; i < rawDataset.length; i += Math.round(fps)) {
    const frame = rawDataset[i];
    const positions = getFramePositions(frame);
    if (!positions || !positions.length) continue;

    let minX = 999, maxX = -999, sumX = 0;
    for (let p of positions) {
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      sumX += px;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
    }
    const count = positions.length;
    const cx = sumX / count;

    // Dynamic Orientation-Aware Position from Own Goal (0m = Own Goal, 105m = Opponent Goal)
    let lh = 0, ch = 0;
    if (attackingDir === 'right_to_left') {
      lh = Math.max(0, Math.min(105.0, 52.5 - maxX));
      ch = Math.max(0, Math.min(105.0, 52.5 - cx));
    } else {
      lh = Math.max(0, Math.min(105.0, minX + 52.5));
      ch = Math.max(0, Math.min(105.0, cx + 52.5));
    }

    const totalSec = startSec + (i / fps);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);

    lineHeights.push(Math.round(lh * 10) / 10);
    centroidHeights.push(Math.round(ch * 10) / 10);
  }

  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';

  const traceLine = {
    x: times,
    y: lineHeights,
    type: 'scatter',
    mode: 'lines',
    name: 'Defensive Line Height (m from Own Goal)',
    fill: 'tozeroy',
    fillcolor: isDark ? 'rgba(245, 158, 11, 0.12)' : 'rgba(245, 158, 11, 0.15)',
    line: { color: '#F59E0B', width: 2.5 }
  };

  const traceCentroid = {
    x: times,
    y: centroidHeights,
    type: 'scatter',
    mode: 'lines',
    name: 'Squad Centroid Height (m from Own Goal)',
    line: { color: '#10B981', width: 2, dash: 'dot' }
  };

  const layout = {
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { text: 'Match Clock (MM:SS)', font: { color: textColor, size: 11 }, standoff: 15 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 },
      tickangle: -45,
      nticks: 15
    },
    yaxis: { 
      title: { text: 'Distance from Own Goal (meters)', font: { color: textColor, size: 11 }, standoff: 10 }, 
      range: [0, 105.0],
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    shapes: [
      {
        type: 'line',
        x0: times[0],
        x1: times[times.length - 1],
        y0: 105.0,
        y1: 105.0,
        line: { color: 'rgba(245, 158, 11, 0.5)', width: 1.5, dash: 'dash' }
      },
      {
        type: 'line',
        x0: times[0],
        x1: times[times.length - 1],
        y0: 52.5,
        y1: 52.5,
        line: { color: 'rgba(255, 255, 255, 0.2)', width: 1, dash: 'dot' }
      }
    ],
    annotations: [
      {
        x: times[Math.floor(times.length / 2)],
        y: 100.0,
        text: 'Opponent Goal Line (105.0m)',
        showarrow: false,
        font: { size: 9, color: '#F59E0B' }
      },
      {
        x: times[Math.floor(times.length / 2)],
        y: 54.5,
        text: 'Halfway Line (52.5m)',
        showarrow: false,
        font: { size: 9, color: '#94A3B8' }
      }
    ],
    margin: { l: 55, r: 25, t: 25, b: 60 },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { color: textColor, size: 10 } }
  };

  Plotly.newPlot('plot-pro-line-height', [traceLine, traceCentroid], layout, { responsive: true });
}

// 2. Squad Vertical Compactness Graph (0-50m back-to-front span)
function renderProCompactnessPlot(rawDataset, meta) {
  if (!rawDataset || !rawDataset.length) return;

  const times = [];
  const compactnessList = [];
  const startSec = meta.clip_start_offset_sec || 0.0;
  const fps = meta.fps || 25.0;

  for (let i = 0; i < rawDataset.length; i += Math.round(fps)) {
    const frame = rawDataset[i];
    const positions = getFramePositions(frame);
    if (!positions || !positions.length) continue;

    let minX = 999, maxX = -999;
    for (let p of positions) {
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
    }

    const totalSec = startSec + (i / fps);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);
    compactnessList.push(Math.round((maxX - minX) * 10) / 10);
  }

  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';

  const traceCompactness = {
    x: times,
    y: compactnessList,
    type: 'scatter',
    mode: 'lines',
    name: 'Squad Vertical Compactness (X-span)',
    fill: 'tozeroy',
    fillcolor: isDark ? 'rgba(236, 72, 153, 0.12)' : 'rgba(236, 72, 153, 0.15)',
    line: { color: '#EC4899', width: 2.5 } // Vibrant Pink / Coral
  };

  const layout = {
    title: { text: 'Squad Vertical Compactness (Back-to-Front X-span)', font: { size: 14, color: textColor, family: 'Outfit, sans-serif' }, y: 0.95 },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { text: 'Match Clock (MM:SS)', font: { color: textColor, size: 11 }, standoff: 15 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 },
      tickangle: -45,
      nticks: 15
    },
    yaxis: { 
      title: { text: 'Compactness Span (meters)', font: { color: textColor, size: 11 }, standoff: 10 }, 
      range: [0, 50.0], // Dynamic realistic scale for squad depth compactness
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    shapes: [
      {
        type: 'line',
        x0: times[0],
        x1: times[times.length - 1],
        y0: 25.0,
        y1: 25.0,
        line: { color: 'rgba(16, 185, 129, 0.4)', width: 1.5, dash: 'dash' }
      },
      {
        type: 'line',
        x0: times[0],
        x1: times[times.length - 1],
        y0: 40.0,
        y1: 40.0,
        line: { color: 'rgba(244, 63, 94, 0.4)', width: 1.5, dash: 'dash' }
      }
    ],
    annotations: [
      {
        x: times[Math.floor(times.length / 2)],
        y: 26.5,
        text: 'Compact Squad Benchmark (< 25m)',
        showarrow: false,
        font: { size: 9, color: '#10B981' }
      },
      {
        x: times[Math.floor(times.length / 2)],
        y: 41.5,
        text: 'Stretched Squad Warning (> 40m)',
        showarrow: false,
        font: { size: 9, color: '#F43F5E' }
      }
    ],
    margin: { l: 55, r: 25, t: 75, b: 70 },
    legend: { orientation: 'h', y: 1.25, x: 0, font: { color: textColor, size: 10 } }
  };

  Plotly.newPlot('plot-pro-compactness', [traceCompactness], layout, { responsive: true });
}

function renderProAreaPlot(rawDataset, meta) {
  if (!rawDataset || !rawDataset.length) return;

  const times = [];
  const areas = [];
  const startSec = meta.clip_start_offset_sec || 0.0;
  const fps = meta.fps || 25.0;

  for (let i = 0; i < rawDataset.length; i += Math.round(fps)) {
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

    const totalSec = startSec + (i / fps);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    times.push(`${min}:${sec.toString().padStart(2, '0')}`);

    const w = Math.max(1, maxY - minY);
    const d = Math.max(1, maxX - minX);
    areas.push(Math.round(w * d));
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
    margin: { l: 55, r: 25, t: 25, b: 60 },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { color: textColor, size: 10 } }
  };

  Plotly.newPlot('plot-pro-area', [traceArea], layout, { responsive: true });
}

function renderProPhasePlot(rawDataset, meta) {
  if (!rawDataset || !rawDataset.length) return;

  const depths = [];
  const widths = [];
  const times = [];
  const startSec = meta.clip_start_offset_sec || 0.0;
  const fps = meta.fps || 25.0;

  for (let i = 0; i < rawDataset.length; i += Math.round(fps)) {
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

    const totalSec = startSec + (i / fps);
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
    hovertemplate: '<b>Clock %{text}</b><br>Compactness: %{x}m<br>Width: %{y}m<extra></extra>',
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
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { 
      title: { text: 'Squad Vertical Compactness (X-span in meters)', font: { color: textColor, size: 11 }, standoff: 15 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    yaxis: { 
      title: { text: 'Team Pitch Width (Y-span in meters)', font: { color: textColor, size: 11 }, standoff: 10 }, 
      showgrid: true, 
      gridcolor: gridColor, 
      tickfont: { color: textColor, size: 10 } 
    },
    shapes: [
      { type: 'rect', x0: 20, y0: 35, x1: 35, y1: 55, line: { color: '#F59E0B', width: 1.5, dash: 'dash' }, fillcolor: 'rgba(245, 158, 11, 0.08)' }
    ],
    annotations: [
      { x: 27.5, y: 56, text: 'Optimal Compactness Zone', showarrow: false, font: { size: 10, color: '#F59E0B' } }
    ],
    margin: { l: 55, r: 25, t: 25, b: 60 }
  };

  Plotly.newPlot('plot-pro-phase', [tracePhase], layout, { responsive: true });
}

// 4. Pitch Thirds View
function renderThirdsView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-thirds');
  if (!container) return;

  const thirds = meta.stats?.thirds_pct || { defensive: 30.0, middle: 55.0, attacking: 15.0 };
  const atkDir = meta.attacking_direction || 'left_to_right';
  const atkLabel = atkDir === 'right_to_left' ? 'Right \u2192 Left' : 'Left \u2192 Right';
  const coachText = narratives.thirds?.summary || `Middle third dominance of ${thirds.middle}%.`;

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Pitch-Thirds Occupancy Dominance</div>
      <div class="page-sub">Attacking Direction: <strong style="color: var(--team-primary-color);">${atkLabel}</strong> — Percentage of frames team centroid occupied each pitch third</div>
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

  renderProThirdsPlot(thirds, meta);
}

function renderProThirdsPlot(thirds, meta) {
  const isDark = isDarkTheme();
  const paperBg = isDark ? '#111622' : '#FFFFFF';
  const plotBg = isDark ? '#0A0D14' : '#F8FAFC';
  const textColor = isDark ? '#F1F5F9' : '#0F172A';
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';
  const teamColor = meta.team_color || '#0080FF';
  const atkDir = meta.attacking_direction || 'left_to_right';
  const atkArrow = atkDir === 'right_to_left' ? '\u2190' : '\u2192';

  const trace = {
    x: [`Defensive Third`, `Middle Third`, `Attacking Third (${atkArrow})`],
    y: [thirds.defensive, thirds.middle, thirds.attacking],
    type: 'bar',
    marker: { color: ['#10B981', teamColor, '#F59E0B'] },
    text: [`${thirds.defensive}%`, `${thirds.middle}%`, `${thirds.attacking}%`],
    textposition: 'auto',
    textfont: { color: '#FFFFFF', family: 'Outfit, sans-serif', weight: 'bold' }
  };

  const layout = {
    title: { text: `Pitch-Thirds Centroid Occupancy (%) — Attacking ${atkDir === 'right_to_left' ? 'Right to Left' : 'Left to Right'}`, font: { size: 14, color: textColor, family: 'Outfit, sans-serif' } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    xaxis: { tickfont: { color: textColor } },
    yaxis: { title: { text: 'Occupancy (%)', font: { color: textColor } }, range: [0, 100], gridcolor: gridColor, tickfont: { color: textColor } },
    margin: { l: 50, r: 20, t: 40, b: 50 }
  };

  Plotly.newPlot('plot-pro-thirds', [trace], layout, { responsive: true });
}

// 5. Trajectory View
function renderTrajectoryView(rawDataset, meta, narratives) {
  const container = document.getElementById('view-trajectory');
  if (!container) return;

  const stats = meta.stats || {};
  const coachText = narratives.trajectory?.summary || `Centroid displacement is ${stats.safe_centroid_dist_km || 0} km.`;

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Safe Team Centroid Trajectory</div>
      <div class="page-sub">Flow vector paths of team movement across video clip with speed outlier rejection</div>
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
        <div class="stat-val" style="color: ${meta.team_color || '#0080FF'};">${stats.safe_centroid_dist_km || 0} km</div>
        <div class="stat-badge">Physical Speed Ceiling: 10 m/s</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Interval Rejection Rate</div>
        <div class="stat-val" style="color: var(--tactical-gold);">${stats.rejection_rate_pct || 0}%</div>
        <div class="stat-badge">Camera Panning Spike Correlation</div>
      </div>
    </div>
  `;
}

// 6. Roster Analytics View
function renderRosterView(trackedDataset, meta, narratives) {
  const container = document.getElementById('view-players');
  if (!container) return;

  const coachText = narratives.players?.summary || "Player tracking provides continuous mobility snapshots.";
  const teamColor = meta.team_color || '#0080FF';
  const startClock = meta.match_start_clock || '00:00';

  // Compute per-player stats from tracked dataset
  const playerMap = {};
  if (trackedDataset && trackedDataset.length) {
    for (let i = 0; i < trackedDataset.length; i++) {
      const frame = trackedDataset[i];
      const positions = getFramePositions(frame);
      for (let j = 0; j < positions.length; j++) {
        const key = `Player #${j + 1}`;
        if (!playerMap[key]) playerMap[key] = { positions: [], firstFrame: i };
        playerMap[key].positions.push(positions[j]);
      }
    }
  }

  let playerRows = '';
  const fps = meta.fps || 25.0;
  const startOffsetSec = meta.clip_start_offset_sec || 0.0;
  const playerKeys = Object.keys(playerMap).slice(0, 11);

  if (playerKeys.length === 0) {
    playerRows = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No tracked player data available</td></tr>`;
  } else {
    for (const key of playerKeys) {
      const data = playerMap[key];
      let totalDist = 0;
      for (let k = 1; k < data.positions.length; k++) {
        const prev = data.positions[k - 1];
        const curr = data.positions[k];
        const px0 = Array.isArray(prev) ? prev[0] : (prev.x || 0);
        const py0 = Array.isArray(prev) ? prev[1] : (prev.y || 0);
        const px1 = Array.isArray(curr) ? curr[0] : (curr.x || 0);
        const py1 = Array.isArray(curr) ? curr[1] : (curr.y || 0);
        const step = Math.sqrt((px1 - px0) ** 2 + (py1 - py0) ** 2);
        if (step < 0.4) totalDist += step;
      }

      const durationSec = data.positions.length / fps;
      const paceKmh = durationSec > 0 ? ((totalDist / durationSec) * 3.6).toFixed(1) : '0.0';
      const startTimeSec = startOffsetSec + (data.firstFrame / fps);
      const min = Math.floor(startTimeSec / 60);
      const sec = Math.floor(startTimeSec % 60);
      const clockStr = `${min}:${sec.toString().padStart(2, '0')}`;

      playerRows += `
        <tr>
          <td><strong style="color: ${teamColor};">${key}</strong></td>
          <td>${clockStr}</td>
          <td>${totalDist.toFixed(1)} m</td>
          <td>${paceKmh} km/h avg pace</td>
          <td><span style="color: var(--tactical-emerald);">Tracked (${data.positions.length} samples)</span></td>
        </tr>`;
    }
  }

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
              ${playerRows}
            </tbody>
        </table>
      </div>
    </div>
  `;
}

// 7. Audit View
function renderAuditView(meta, narratives) {
  const container = document.getElementById('view-audit');
  if (!container) return;

  const coachText = narratives.methodology?.summary || "Pipeline rejects impossible frame jumps using 10 m/s threshold.";

  const stats = meta.stats || {};
  const calibRate = stats.calib_rate_pct !== undefined ? stats.calib_rate_pct : 'N/A';
  const calibSolved = stats.calib_solved || 0;
  const calibTotal = (stats.calib_solved || 0) + (stats.calib_failed || 0);
  const rejectionRate = stats.rejection_rate_pct || 0;

  container.innerHTML = `
    <div class="page-header">
      <div class="page-title">Data Quality & Pipeline Methodology Audit</div>
      <div class="page-sub">PnL Calibration diagnostics, homography projection accuracy, and outlier rejection models</div>
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
        <div class="stat-val" style="color: var(--tactical-emerald);">${calibRate}%</div>
        <div class="stat-badge">${calibSolved} / ${calibTotal} Frames Solved</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Speed Outlier Rejection</div>
        <div class="stat-val">${rejectionRate} <span style="font-size: 16px; color: var(--text-muted);">%</span></div>
        <div class="stat-badge">Ceiling: 10 m/s (camera pan filter)</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Total Processed Frames</div>
        <div class="stat-val" style="color: var(--team-primary-color);">${stats.total_frames || 0}</div>
        <div class="stat-badge">${stats.total_samples || 0} total position samples</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Clip Duration</div>
        <div class="stat-val">${((stats.duration_sec || 0) / 60).toFixed(1)} <span style="font-size: 16px; color: var(--text-muted);">min</span></div>
        <div class="stat-badge">${meta.fps || 25} FPS source</div>
      </div>
    </div>
  `;
}
