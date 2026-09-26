/**
 * Main Application Initializer & Single-Page Router
 */
window.gRawDataset = [];
window.gTrackedDataset = [];
window.gMetadata = {};
window.gNarratives = {};

async function initDashboardApp() {
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
