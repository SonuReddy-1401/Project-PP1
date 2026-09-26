/**
 * Shared Utilities for Interactive Match Analytics Dashboard
 */

// Global match clock offset constant (Clip 15min runs from 9:54 to 24:54 of match)
const DEFAULT_CLIP_START_OFFSET_SEC = 594.0; 

/**
 * Converts a raw frame index into a real match-clock timestamp string "MM:SS".
 * @param {number} frameIdx - Current frame index (0-indexed)
 * @param {number} fps - Video frames per second (default: 25.0)
 * @param {number} clipStartOffsetSec - Match clock offset in seconds (default: 594.0s = 9:54)
 * @returns {string} Formatted match clock time "MM:SS"
 */
function frameToTimestamp(frameIdx, fps = 25.0, clipStartOffsetSec = DEFAULT_CLIP_START_OFFSET_SEC) {
  const totalSec = clipStartOffsetSec + (frameIdx / fps);
  const min = Math.floor(totalSec / 60);
  const sec = Math.floor(totalSec % 60);
  return `${min}:${sec.toString().padStart(2, '0')}`;
}

/**
 * Converts total seconds into "MM:SS" string.
 */
function secondsToMatchClock(secTotal, clipStartOffsetSec = DEFAULT_CLIP_START_OFFSET_SEC) {
  const totalSec = clipStartOffsetSec + secTotal;
  const min = Math.floor(totalSec / 60);
  const sec = Math.floor(totalSec % 60);
  return `${min}:${sec.toString().padStart(2, '0')}`;
}

/**
 * FIFA Pitch Layout Shapes Helper (105m x 68m)
 */
function getPitchShapes() {
  return [
    // Pitch Surface Box
    { type: 'rect', x0: -52.5, y0: -34, x1: 52.5, y1: 34, fillcolor: '#E8EDE4', line: { color: '#8A9186', width: 2 }, layer: 'below' },
    // Halfway Line
    { type: 'line', x0: 0, y0: -34, x1: 0, y1: 34, line: { color: '#8A9186', width: 1.5 } },
    // Center Circle
    { type: 'circle', x0: -9.15, y0: -9.15, x1: 9.15, y1: 9.15, line: { color: '#8A9186', width: 1.5 } },
    // Left Penalty Box
    { type: 'rect', x0: -52.5, y0: -20.16, x1: -36.0, y1: 20.16, line: { color: '#8A9186', width: 1.5 } },
    // Right Penalty Box
    { type: 'rect', x0: 36.0, y0: -20.16, x1: 52.5, y1: 20.16, line: { color: '#8A9186', width: 1.5 } },
    // Left Goal Box
    { type: 'rect', x0: -52.5, y0: -9.16, x1: -47.0, y1: 9.16, line: { color: '#8A9186', width: 1.0 } },
    // Right Goal Box
    { type: 'rect', x0: 47.0, y0: -9.16, x1: 52.5, y1: 9.16, line: { color: '#8A9186', width: 1.0 } }
  ];
}

/**
 * Returns standard Plotly layout configuration for pitch maps.
 */
function getPitchLayout(titleText) {
  return {
    title: { text: titleText, font: { size: 14, color: '#1A1A1A', family: '-apple-system, sans-serif' } },
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#E8EDE4',
    shapes: getPitchShapes(),
    xaxis: { range: [-56, 56], zeroline: false, showgrid: false, ticks: '', showticklabels: false },
    yaxis: { range: [-37, 37], zeroline: false, showgrid: false, ticks: '', showticklabels: false, scaleanchor: 'x' },
    margin: { l: 20, r: 20, t: 40, b: 20 }
  };
}

/**
 * Toggle visibility of collapsible table drawers.
 */
function toggleRawTable(containerId) {
  const el = document.getElementById(containerId);
  if (el) {
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
  }
}

/**
 * Exports tabular data as a client-side CSV download.
 */
function exportCSV(tableId, filename) {
  const table = document.getElementById(tableId);
  if (!table) return;
  let rows = Array.from(table.querySelectorAll('tr'));
  let csvContent = rows.map(r => Array.from(r.querySelectorAll('th,td')).map(td => `"${td.innerText.replace(/"/g, '""')}"`).join(',')).join('\n');

  let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  let url = URL.createObjectURL(blob);
  let a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
