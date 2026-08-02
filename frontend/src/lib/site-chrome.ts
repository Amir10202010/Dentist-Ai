/**
 * Shared chrome for the public site: mobile nav, theme toggle, sticky header.
 *
 * Extracted so the landing bundle and the rest of the marketing site run the
 * same code rather than two copies that drift. Everything here is progressive
 * enhancement — every marketing page is fully legible with JavaScript off.
 */

import { all, on } from './dom';
import { cycleTheme, initTheme } from './theme';

function initMobileNav(): void {
  for (const toggle of all<HTMLButtonElement>('[data-nav-toggle]')) {
    const menu = document.getElementById(toggle.getAttribute('aria-controls') ?? '');
    if (!menu) continue;

    const setOpen = (open: boolean): void => {
      toggle.setAttribute('aria-expanded', String(open));
      menu.dataset['open'] = String(open);
      document.body.style.overflow = open ? 'hidden' : '';
    };

    on(toggle, 'click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));
    on(document.documentElement, 'keydown', (event) => {
      if (event.key === 'Escape') setOpen(false);
    });
    for (const link of all('a', menu)) {
      on(link, 'click', () => setOpen(false));
    }
  }
}

function initThemeToggle(): void {
  for (const button of all<HTMLButtonElement>('[data-theme-toggle]')) {
    on(button, 'click', () => cycleTheme());
  }
}

/** Give the sticky header a border once the page has scrolled past the top. */
function initHeaderState(): void {
  const header = document.querySelector<HTMLElement>('[data-site-header]');
  if (!header) return;

  const sentinel = document.createElement('div');
  sentinel.setAttribute('aria-hidden', 'true');
  header.before(sentinel);

  // An IntersectionObserver on a sentinel avoids a scroll listener firing on
  // every frame.
  new IntersectionObserver(
    ([entry]) => {
      header.dataset['scrolled'] = String(entry ? !entry.isIntersecting : false);
    },
    { threshold: 0 },
  ).observe(sentinel);
}

export function initSiteChrome(): void {
  initTheme();
  initMobileNav();
  initThemeToggle();
  initHeaderState();
}

/** Runs `boot` once the document is parsed, now or on DOMContentLoaded. */
export function onReady(boot: () => void): void {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
}
