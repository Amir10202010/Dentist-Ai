/**
 * Application shell entry.
 *
 * One bundle for the authenticated app. Page controllers are loaded via
 * dynamic `import()`, so a user on the dashboard never downloads the WebGL
 * mesh viewer — Rollup splits each screen into its own chunk.
 */

import '../styles/tokens.css';
import '../styles/base.css';
import '../styles/components.css';
import '../styles/app.css';

import { api } from '../lib/api';
import { all, maybe, on } from '../lib/dom';
import { notifyError } from '../lib/toast';
import { cycleTheme, initTheme } from '../lib/theme';
import { initCommandPalette } from '../features/command-palette';

function initSidebar(): void {
  const sidebar = maybe('[data-sidebar]');
  const toggle = maybe<HTMLButtonElement>('[data-sidebar-toggle]');
  const scrim = maybe('[data-sidebar-scrim]');
  if (!sidebar || !toggle) return;

  const setOpen = (open: boolean): void => {
    sidebar.dataset['open'] = String(open);
    toggle.setAttribute('aria-expanded', String(open));
    if (scrim) scrim.hidden = !open;
    // Prevent the page behind the drawer from scrolling on touch.
    document.body.style.overflow = open ? 'hidden' : '';
  };

  on(toggle, 'click', () => setOpen(sidebar.dataset['open'] !== 'true'));
  if (scrim) on(scrim, 'click', () => setOpen(false));

  on(document.documentElement, 'keydown', (event) => {
    if (event.key === 'Escape' && sidebar.dataset['open'] === 'true') setOpen(false);
  });

  // Restore the desktop layout if the viewport grows while the drawer is open.
  const wide = window.matchMedia('(min-width: 900px)');
  wide.addEventListener('change', (event) => {
    if (event.matches) setOpen(false);
  });
}

function initThemeToggle(): void {
  for (const button of all<HTMLButtonElement>('[data-theme-toggle]')) {
    on(button, 'click', () => {
      const next = cycleTheme();
      button.setAttribute('aria-label', next === 'dark' ? 'Светлая тема' : 'Тёмная тема');
    });
  }
}

function initLogout(): void {
  for (const button of all<HTMLButtonElement>('[data-logout]')) {
    on(button, 'click', () => {
      void api.auth
        .logout()
        .catch(notifyError)
        .finally(() => location.assign('/login'));
    });
  }
}

async function initPage(): Promise<void> {
  const page = document.body.dataset['page'];

  switch (page) {
    case 'dashboard': {
      const { initDashboard } = await import('../features/dashboard');
      await initDashboard();
      break;
    }
    case 'studies': {
      const { initStudiesPage } = await import('../features/studies');
      initStudiesPage();
      break;
    }
    case 'study-detail': {
      const publicId = document.body.dataset['studyId'];
      if (!publicId) return;
      const { initStudyDetail } = await import('../features/study-detail');
      initStudyDetail(publicId);
      break;
    }
    case 'volumes': {
      const { initVolumesPage } = await import('../features/volumes');
      initVolumesPage();
      break;
    }
    case 'volume-detail': {
      const publicId = document.body.dataset['volumeId'];
      if (!publicId) return;
      const { initVolumeDetail } = await import('../features/volume-detail');
      await initVolumeDetail(publicId);
      break;
    }
    case 'patients': {
      const { initPatientsPage } = await import('../features/patients');
      initPatientsPage();
      break;
    }
    case 'patient-detail': {
      const patientId = Number(document.body.dataset['patientId']);
      if (!patientId) return;
      const { initPatientDetail } = await import('../features/patient-detail');
      initPatientDetail(patientId);
      break;
    }
    case 'scan-detail': {
      const publicId = document.body.dataset['scanId'];
      if (!publicId) return;
      const { initScanDetail } = await import('../features/scan-detail');
      initScanDetail(publicId);
      break;
    }
    case 'settings': {
      const { initSettingsPage } = await import('../features/settings');
      initSettingsPage();
      break;
    }
    default:
      break;
  }
}

function boot(): void {
  initTheme();
  initSidebar();
  initThemeToggle();
  initLogout();
  initCommandPalette();
  void initPage().catch(notifyError);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
