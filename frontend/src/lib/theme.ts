/**
 * Theme switching.
 *
 * Three states, not two: `system` is the default and keeps tracking the OS
 * preference, while an explicit choice is remembered. The initial value is
 * applied by a tiny inline script in `base.html` before first paint, so there
 * is no flash of the wrong theme.
 */

import { on } from './dom';

export type ThemeChoice = 'light' | 'dark' | 'system';

const STORAGE_KEY = 'dentist-ai:theme';

function isThemeChoice(value: string | null): value is ThemeChoice {
  return value === 'light' || value === 'dark' || value === 'system';
}

export function getTheme(): ThemeChoice {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return isThemeChoice(stored) ? stored : 'system';
  } catch {
    // Safari in private mode throws on localStorage access.
    return 'system';
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === 'system') {
    root.removeAttribute('data-theme');
  } else {
    root.dataset['theme'] = choice;
  }
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // Persistence is a nicety; the applied theme still works this session.
  }
  document.dispatchEvent(new CustomEvent<ThemeChoice>('themechange', { detail: choice }));
}

/** Resolve `system` to what is actually on screen right now. */
export function resolvedTheme(): 'light' | 'dark' {
  const choice = getTheme();
  if (choice !== 'system') return choice;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function cycleTheme(): ThemeChoice {
  const next: ThemeChoice = resolvedTheme() === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  return next;
}

export function initTheme(): void {
  applyTheme(getTheme());

  // Follow the OS while the user has not expressed a preference.
  const query = window.matchMedia('(prefers-color-scheme: dark)');
  on(query as unknown as EventTarget, 'change' as never, () => {
    if (getTheme() === 'system') applyTheme('system');
  });
}
