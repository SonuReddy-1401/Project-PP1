/**
 * PRO_DASHBOARD Main App Initializer & State Controller
 */
window.gRawDataset = [];
window.gTrackedDataset = [];
window.gMetadata = {};
window.gNarratives = {};

// Theme Switcher Controller
function toggleTheme() {
  const currentTheme = document.body.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.body.setAttribute('data-theme', newTheme);
  localStorage.setItem('pro_theme', newTheme);

  // Update theme toggle label
  const lbl = document.getElementById('theme-toggle-lbl');
  if (lbl) {
    lbl.innerText = newTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
  }

  // Refresh active view to adjust Plotly/Canvas colors
  const activeView = document.querySelector('.view-section.active');
  if (activeView) {
    switchTab(activeView.id.replace('view-', ''));
  }
}

// Single-Page View Switcher
function switchTab(viewId) {
  // Update sidebar active link
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('onclick')?.includes(viewId)) {
      link.classList.add('active');
    }
  });

  // Update view section active class
  document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
  const targetSec = document.getElementById(`view-${viewId}`);
  if (targetSec) {
    targetSec.classList.add('active');
  }

  // Render view content dynamically
  switch (viewId) {
    case 'overview':
      renderOverviewView(window.gMetadata, window.gNarratives);
      break;
    case 'heatmap':
      renderHeatmapView(window.gRawDataset, window.gMetadata, window.gNarratives);
      break;
    case 'shape':
      renderShapeView(window.gRawDataset, window.gMetadata, window.gNarratives);
      break;
    case 'thirds':
      renderThirdsView(window.gRawDataset, window.gMetadata, window.gNarratives);
      break;
    case 'trajectory':
      renderTrajectoryView(window.gRawDataset, window.gMetadata, window.gNarratives);
      break;
    case 'players':
      renderRosterView(window.gTrackedDataset, window.gMetadata, window.gNarratives);
      break;
    case 'audit':
      renderAuditView(window.gMetadata, window.gNarratives);
      break;
  }
}

// App Bootstrapper
window.addEventListener('DOMContentLoaded', async () => {
  // Restore saved theme
  const savedTheme = localStorage.getItem('pro_theme') || 'dark';
  document.body.setAttribute('data-theme', savedTheme);
  const lbl = document.getElementById('theme-toggle-lbl');
  if (lbl) lbl.innerText = savedTheme === 'dark' ? 'Light Mode' : 'Dark Mode';

  // Instantiate Timeline Player
  window.gTimelinePlayer = new ProTimelinePlayer();

  try {
    // Load JSON datasets asynchronously
    const [metaRes, narrRes, rawRes, trackedRes] = await Promise.all([
      fetch('data/metadata.json'),
      fetch('data/narratives.json'),
      fetch('data/positions_dataset_clip15min.json'),
      fetch('data/positions_dataset_clip15min_tracked.json')
    ]);

    window.gMetadata = await metaRes.json();
    window.gNarratives = await narrRes.json();
    window.gRawDataset = await rawRes.json();
    window.gTrackedDataset = await trackedRes.json();

    // Initial view render
    switchTab('overview');
  } catch (err) {
    console.error('Failed to load JSON datasets:', err);
  }
});
