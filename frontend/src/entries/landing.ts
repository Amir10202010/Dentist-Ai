/**
 * Landing page entry.
 *
 * Only the showcase tabs need scripting, and even they degrade cleanly: every
 * panel is in the DOM and only the first is hidden, so a failed bundle costs
 * the tab switching and nothing else.
 */

import '../styles/tokens.css';
import '../styles/base.css';
import '../styles/components.css';
import '../styles/marketing.css';
import '../styles/landing.css';

import { all, on } from '../lib/dom';
import { initSiteChrome, onReady } from '../lib/site-chrome';

function initTabs(): void {
  for (const group of all('[data-tabs]')) {
    const tabs = all<HTMLButtonElement>('[role="tab"]', group);
    if (tabs.length === 0) continue;

    const panelFor = (tab: HTMLElement): HTMLElement | null =>
      document.getElementById(tab.getAttribute('aria-controls') ?? '');

    const select = (next: HTMLButtonElement, { focus = true } = {}): void => {
      for (const tab of tabs) {
        const active = tab === next;
        tab.setAttribute('aria-selected', String(active));
        // Roving tabindex: only the selected tab is a tab stop, so Tab moves
        // past the whole group rather than through every option in it.
        tab.tabIndex = active ? 0 : -1;
        const panel = panelFor(tab);
        if (panel) panel.hidden = !active;
      }
      if (focus) next.focus();
    };

    for (const tab of tabs) {
      on(tab, 'click', () => select(tab, { focus: false }));
    }

    on(group, 'keydown', (event) => {
      const current = tabs.indexOf(document.activeElement as HTMLButtonElement);
      if (current === -1) return;

      const moves: Record<string, number> = {
        ArrowRight: 1,
        ArrowLeft: -1,
        Home: -current,
        End: tabs.length - 1 - current,
      };
      const delta = moves[event.key];
      if (delta === undefined) return;

      event.preventDefault();
      // Wraps, which is what the ARIA authoring practices specify for a
      // horizontal tablist.
      const next = tabs[(current + delta + tabs.length) % tabs.length];
      if (next) select(next);
    });
  }
}

onReady(() => {
  initSiteChrome();
  initTabs();
});
