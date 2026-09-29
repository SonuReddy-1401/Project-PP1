/**
 * Main Dynamic Dashboard Application Loader
 * Loads data asynchronously from data/ directory and binds theme/navigation handlers.
 */
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const metaRes = await fetch('data/metadata.json');
    const meta = await metaRes.json();
    window.gMetadata = meta;

    const navRes = await fetch('data/narratives.json');
    const narratives = await navRes.json();
    window.gNarratives = narratives;

    const dataRes = await fetch('data/positions_dataset.json');
    const rawDataset = await dataRes.json();
    window.gRawDataset = rawDataset;

    // 1. Inject Dynamic Kit Color into CSS Variables
    if (meta.team_color) {
      document.body.style.setProperty('--team-primary-color', meta.team_color);
    }
    if (meta.team_color_secondary) {
      document.body.style.setProperty('--team-secondary-color', meta.team_color_secondary);
    }

    // 2. Update Header Readouts
    const titleEl = document.getElementById('header-match-title');
    if (titleEl) {
      titleEl.innerText = `${meta.team_name || 'Target Squad'} vs ${meta.opponent_name || 'Opponent'}`;
    }
    const badgeEl = document.getElementById('header-league-badge');
    if (badgeEl) {
      badgeEl.innerText = meta.team_badge || 'MATCH ANALYTICS';
    }

    // 3. Initialize View Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const viewPanels = document.querySelectorAll('.view-panel');

    function switchView(viewName) {
      navItems.forEach(item => {
        if (item.dataset.view === viewName) item.classList.add('active');
        else item.classList.remove('active');
      });

      viewPanels.forEach(panel => {
        if (panel.id === `view-${viewName}`) panel.classList.add('active');
        else panel.classList.remove('active');
      });

      // Render view content dynamically
      if (viewName === 'overview') renderOverviewView(meta, narratives);
      else if (viewName === 'heatmap') renderHeatmapView(rawDataset, meta, narratives);
      else if (viewName === 'shape') renderShapeView(rawDataset, meta, narratives);
      else if (viewName === 'thirds') renderThirdsView(rawDataset, meta, narratives);
      else if (viewName === 'trajectory') renderTrajectoryView(rawDataset, meta, narratives);
      else if (viewName === 'players') renderRosterView(rawDataset, meta, narratives);
      else if (viewName === 'audit') renderAuditView(meta, narratives);
    }

    navItems.forEach(item => {
      item.onclick = () => switchView(item.dataset.view);
    });

    // 4. Initialize Theme Toggle
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.onclick = () => {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.body.setAttribute('data-theme', newTheme);

        // Re-render active view to update chart themes
        const activeNav = document.querySelector('.nav-item.active');
        if (activeNav) switchView(activeNav.dataset.view);
      };
    }

    // 5. Initialize Sidebar Toggle
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('app-sidebar');
    if (sidebarToggle && sidebar) {
      sidebarToggle.onclick = () => {
        sidebar.classList.toggle('collapsed');
      };
    }

    // Default View Initialization
    switchView('overview');

  } catch (err) {
    console.error('Failed to initialize Pro Dashboard:', err);
  }
});
