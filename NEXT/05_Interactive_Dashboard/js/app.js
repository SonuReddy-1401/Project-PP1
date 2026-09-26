/**
 * Main Application Initializer & Single-Page Router
 */
window.gRawDataset = [];
window.gTrackedDataset = [];
window.gMetadata = {};
window.gNarratives = {};

/**
 * Apply theme ('light' or 'dark') and sync UI controls + Plotly charts.
 */
function applyTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);

  const sunIcon = document.getElementById('theme-sun-icon');
  const moonIcon = document.getElementById('theme-moon-icon');
  const btnLabel = document.getElementById('theme-btn-label');

  if (theme === 'dark') {
    if (sunIcon) sunIcon.style.display = 'inline';
    if (moonIcon) moonIcon.style.display = 'none';
    if (btnLabel) btnLabel.innerText = 'Light Mode';
  } else {
    if (sunIcon) sunIcon.style.display = 'none';
    if (moonIcon) moonIcon.style.display = 'inline';
    if (btnLabel) btnLabel.innerText = 'Dark Mode';
  }

  // Re-render active section's Plotly chart cleanly if datasets are loaded
  if (window.gRawDataset && window.gRawDataset.length) {
    const activeSection = document.querySelector('.section.active');
    if (activeSection) {
      switchTab(activeSection.id);
    }
  }
}

/**
 * Toggle between light and dark modes.
 */
function toggleTheme() {
  const currentTheme = document.body.getAttribute('data-theme') || 'light';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  applyTheme(newTheme);
}

/**
 * Initialize theme preference from localStorage or OS settings.
 */
function initTheme() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    applyTheme(savedTheme);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    applyTheme('dark');
  } else {
    applyTheme('light');
  }
}

async function initDashboardApp() {
  initTheme();

  try {
    const [resMeta, resNarratives, resRaw, resTracked] = await Promise.all([
      fetch('metadata.json'),
      fetch('narratives.json'),
      fetch('positions_dataset_clip15min.json'),
      fetch('positions_dataset_clip15min_tracked.json')
    ]);

    window.gMetadata = await resMeta.json();
    window.gNarratives = await resNarratives.json();
    window.gRawDataset = await resRaw.json();
    window.gTrackedDataset = await resTracked.json();

    // Set Sidebar Brand & Swatch
    const teamName = window.gMetadata.target_team_name || "Napoli";
    const sidebarTeamEl = document.getElementById('sidebar-team-name');
    if (sidebarTeamEl) {
      sidebarTeamEl.innerText = `${teamName} Analytics`;
    }

    // Render Overview Component by default
    renderOverviewComponent(window.gMetadata, window.gNarratives);

  } catch (err) {
    console.error("[ERROR] Failed to initialize dashboard application datasets:", err);
  }
}

/**
 * Single-Page Navigation Router
 */
function switchTab(sectionId, btnElement) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item button').forEach(b => b.classList.remove('active'));

  const activeSection = document.getElementById(sectionId);
  if (activeSection) {
    activeSection.classList.add('active');
  }
  
  if (btnElement) {
    btnElement.classList.add('active');
  } else {
    // Find button matching sectionId if navigated programmatically
    const matchBtn = document.querySelector(`.nav-item button[onclick*="${sectionId}"]`);
    if (matchBtn) matchBtn.classList.add('active');
  }

  // Render modular components on tab switch
  if (sectionId === 'overview') {
    renderOverviewComponent(window.gMetadata, window.gNarratives);
  } else if (sectionId === 'heatmap') {
    renderHeatmapComponent(window.gRawDataset, window.gMetadata, window.gNarratives);
  } else if (sectionId === 'shape') {
    renderShapeComponent(window.gRawDataset, window.gMetadata, window.gNarratives);
  } else if (sectionId === 'thirds') {
    renderThirdsComponent(window.gRawDataset, window.gMetadata, window.gNarratives);
  } else if (sectionId === 'trajectory') {
    renderTrajectoryComponent(window.gRawDataset, window.gMetadata, window.gNarratives);
  } else if (sectionId === 'players') {
    renderPlayersComponent(window.gTrackedDataset, window.gMetadata, window.gNarratives);
  } else if (sectionId === 'methodology') {
    renderMethodologyComponent(window.gMetadata, window.gNarratives);
  }
}

// Launch application on DOM load
window.addEventListener('DOMContentLoaded', initDashboardApp);
