/**
 * Shared Utilities for Interactive Match Analytics Dashboard
 */

// Global match clock offset constant (Clip 15min runs from 9:54 to 24:54 of match)
const DEFAULT_CLIP_START_OFFSET_SEC = 594.0; 

/**
 * Check if the application is currently in Dark Mode.
 */
function isDarkMode() {
  return document.body.getAttribute('data-theme') === 'dark';
}

/**
 * Converts a raw frame index into a real match-clock timestamp string "MM:SS".
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
 * Theme-aware FIFA Pitch Layout Shapes Helper (105m x 68m)
 */
function getPitchShapes() {
  const isDark = isDarkMode();
  const pitchFill = isDark ? '#17251D' : '#E8EDE4';
  const pitchLine = isDark ? '#475569' : '#8A9186';

  return [
    // Pitch Surface Box
    { type: 'rect', x0: -52.5, y0: -34, x1: 52.5, y1: 34, fillcolor: pitchFill, line: { color: pitchLine, width: 2 }, layer: 'below' },
    // Halfway Line
    { type: 'line', x0: 0, y0: -34, x1: 0, y1: 34, line: { color: pitchLine, width: 1.5 } },
    // Center Circle
    { type: 'circle', x0: -9.15, y0: -9.15, x1: 9.15, y1: 9.15, line: { color: pitchLine, width: 1.5 } },
    // Left Penalty Box
    { type: 'rect', x0: -52.5, y0: -20.16, x1: -36.0, y1: 20.16, line: { color: pitchLine, width: 1.5 } },
    // Right Penalty Box
    { type: 'rect', x0: 36.0, y0: -20.16, x1: 52.5, y1: 20.16, line: { color: pitchLine, width: 1.5 } },
    // Left Goal Box
    { type: 'rect', x0: -52.5, y0: -9.16, x1: -47.0, y1: 9.16, line: { color: pitchLine, width: 1.0 } },
    // Right Goal Box
    { type: 'rect', x0: 47.0, y0: -9.16, x1: 52.5, y1: 9.16, line: { color: pitchLine, width: 1.0 } }
  ];
}

/**
 * Theme-aware Plotly layout configuration for pitch maps.
 */
function getPitchLayout(titleText) {
  const isDark = isDarkMode();
  const paperBg = isDark ? '#1E293B' : '#FFFFFF';
  const plotBg = isDark ? '#17251D' : '#E8EDE4';
  const textColor = isDark ? '#F8FAFC' : '#1A1A1A';

  return {
    title: { text: titleText, font: { size: 14, color: textColor, family: '-apple-system, sans-serif' } },
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
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
