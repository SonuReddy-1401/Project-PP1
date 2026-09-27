/**
 * Custom 2D FIFA Pitch Canvas Renderer (User-Specified Coordinate Transposition)
 * Maps pitch coordinates using right-sided flip (invX = x) and upside-down flip (invY = -y)
 * to match user reference frame alignment.
 */
class ProPitchCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    // FIFA pitch dimensions (meters): 105m length x 68m width
    this.pitchLength = 105.0;
    this.pitchWidth = 68.0;
    this.margin = 35; // Canvas margin padding in px

    this.width = this.canvas.width = 840;
    this.height = this.canvas.height = 544;

    this.mode = 'live'; // 'live', 'heatmap', 'shape', 'trajectory'
    this.currentPositions = [];
    this.currentCentroid = null;
  }

  setMode(mode) {
    this.mode = mode;
    this.draw();
  }

  // Pitch coordinate (-52.5 to +52.5 X, -34 to +34 Y) to Canvas pixel (X, Y)
  // Maps invX = x (right-sided) and invY = -y (upside-down flip) per user specification.
  pitchToCanvas(x, y) {
    const drawW = this.width - (this.margin * 2);
    const drawH = this.height - (this.margin * 2);

    const invX = x;  // Right-sided orientation
    const invY = -y; // Upside-down orientation flip

    const cx = this.margin + ((invX + 52.5) / 105.0) * drawW;
    const cy = this.margin + ((34.0 - invY) / 68.0) * drawH; // Y inverted for screen space
    return { x: cx, y: cy };
  }

  drawPitch() {
    const isLight = document.body.getAttribute('data-theme') === 'light';
    const grassColor = isLight ? '#E2EADF' : '#142419';
    const lineColor = isLight ? '#859B87' : '#7E9381';

    // Clear Canvas
    this.ctx.fillStyle = grassColor;
    this.ctx.fillRect(0, 0, this.width, this.height);

    const m = this.margin;
    const w = this.width - m * 2;
    const h = this.height - m * 2;

    this.ctx.strokeStyle = lineColor;
    this.ctx.lineWidth = 2;

    // Pitch Boundary Box
    this.ctx.strokeRect(m, m, w, h);

    // Halfway Line
    this.ctx.beginPath();
    this.ctx.moveTo(m + w / 2, m);
    this.ctx.lineTo(m + w / 2, m + h);
    this.ctx.stroke();

    // Center Circle (9.15m radius)
    const center = this.pitchToCanvas(0, 0);
    const radiusPx = (9.15 / 105.0) * w;
    this.ctx.beginPath();
    this.ctx.arc(center.x, center.y, radiusPx, 0, Math.PI * 2);
    this.ctx.stroke();

    // Penalty Boxes (16.5m depth, 40.32m width)
    const penDepthPx = (16.5 / 105.0) * w;
    const penWidthPx = (40.32 / 68.0) * h;
    const penYPx = m + (h - penWidthPx) / 2;

    // Left Penalty Box
    this.ctx.strokeRect(m, penYPx, penDepthPx, penWidthPx);
    // Right Penalty Box
    this.ctx.strokeRect(m + w - penDepthPx, penYPx, penDepthPx, penWidthPx);

    // Goal Boxes (5.5m depth, 18.32m width)
    const goalDepthPx = (5.5 / 105.0) * w;
    const goalWidthPx = (18.32 / 68.0) * h;
    const goalYPx = m + (h - goalWidthPx) / 2;

    // Left Goal Box
    this.ctx.strokeRect(m, goalYPx, goalDepthPx, goalWidthPx);
    // Right Goal Box
    this.ctx.strokeRect(m + w - goalDepthPx, goalYPx, goalDepthPx, goalWidthPx);
  }

  // Draw 2D Spatial Density Heatmap
  drawHeatmap(allPitchCoords) {
    if (!allPitchCoords || !allPitchCoords.length) return;

    // Create 35x22 grid density
    const gridCols = 35;
    const gridRows = 22;
    const counts = Array(gridRows).fill(0).map(() => Array(gridCols).fill(0));
    let maxVal = 1;

    for (let i = 0; i < allPitchCoords.length; i++) {
      const p = allPitchCoords[i];
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      const py = Array.isArray(p) ? p[1] : (p.y !== undefined ? p.y : 0);

      const invX = px;
      const invY = -py;
      const col = Math.floor(((invX + 52.5) / 105.0) * gridCols);
      const row = Math.floor(((invY + 34.0) / 68.0) * gridRows);
      if (col >= 0 && col < gridCols && row >= 0 && row < gridRows) {
        counts[row][col]++;
        if (counts[row][col] > maxVal) maxVal = counts[row][col];
      }
    }

    const cellW = (this.width - this.margin * 2) / gridCols;
    const cellH = (this.height - this.margin * 2) / gridRows;

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const val = counts[r][c];
        if (val > 0) {
          const norm = val / maxVal;
          const alpha = Math.min(0.85, norm * 1.2);
          // Heatmap Gradient (Sky Blue -> Amber -> Rose Red)
          let color = `rgba(0, 128, 255, ${alpha})`;
          if (norm > 0.4) color = `rgba(245, 158, 11, ${alpha})`;
          if (norm > 0.7) color = `rgba(244, 63, 94, ${alpha})`;

          const xPx = this.margin + c * cellW;
          const yPx = this.height - this.margin - (r + 1) * cellH;

          this.ctx.fillStyle = color;
          this.ctx.fillRect(xPx, yPx, cellW + 0.5, cellH + 0.5);
        }
      }
    }
  }

  // Render current frame positions, team shape polygon, and centroid
  updateFrame(playersData) {
    this.currentPositions = playersData || [];
    this.draw();
  }

  draw() {
    this.drawPitch();

    if (!this.currentPositions.length) return;

    let sumX = 0, sumY = 0;
    let minX = 999, maxX = -999, minY = 999, maxY = -999;
    const screenPoints = [];

    for (let i = 0; i < this.currentPositions.length; i++) {
      const p = this.currentPositions[i];
      const px = Array.isArray(p) ? p[0] : (p.x !== undefined ? p.x : 0);
      const py = Array.isArray(p) ? p[1] : (p.y !== undefined ? p.y : 0);

      sumX += px;
      sumY += py;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;

      screenPoints.push(this.pitchToCanvas(px, py));
    }

    const count = this.currentPositions.length;
    const centroidX = sumX / count;
    const centroidY = sumY / count;
    const centroidPt = this.pitchToCanvas(centroidX, centroidY);

    // Calculate bounding box using mapped screen coordinates of corners
    const corners = [
      this.pitchToCanvas(minX, minY),
      this.pitchToCanvas(minX, maxY),
      this.pitchToCanvas(maxX, minY),
      this.pitchToCanvas(maxX, maxY)
    ];

    let screenMinX = 9999, screenMaxX = -9999, screenMinY = 9999, screenMaxY = -9999;
    for (let c of corners) {
      if (c.x < screenMinX) screenMinX = c.x;
      if (c.x > screenMaxX) screenMaxX = c.x;
      if (c.y < screenMinY) screenMinY = c.y;
      if (c.y > screenMaxY) screenMaxY = c.y;
    }

    const boxW = screenMaxX - screenMinX;
    const boxH = screenMaxY - screenMinY;

    this.ctx.fillStyle = 'rgba(0, 128, 255, 0.15)';
    this.ctx.strokeStyle = '#0080FF';
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([4, 4]);

    this.ctx.fillRect(screenMinX, screenMinY, boxW, boxH);
    this.ctx.strokeRect(screenMinX, screenMinY, boxW, boxH);
    this.ctx.setLineDash([]);

    // Draw Player Dots
    for (let i = 0; i < screenPoints.length; i++) {
      const pt = screenPoints[i];
      this.ctx.beginPath();
      this.ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
      this.ctx.fillStyle = '#0080FF';
      this.ctx.fill();
      this.ctx.strokeStyle = '#FFFFFF';
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }

    // Draw Team Centroid Star Marker
    this.ctx.beginPath();
    this.ctx.arc(centroidPt.x, centroidPt.y, 9, 0, Math.PI * 2);
    this.ctx.fillStyle = '#F59E0B';
    this.ctx.fill();
    this.ctx.strokeStyle = '#FFFFFF';
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    // Label Centroid
    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.font = 'bold 10px Outfit, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('★ CENTROID', centroidPt.x, centroidPt.y - 12);
  }
}
