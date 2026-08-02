/**
 * ⌘K command palette.
 *
 * Navigation, actions and patient search in one keyboard surface. Built on a
 * native `<dialog>` so focus handling and Esc come from the platform.
 */

import { api } from '../lib/api';
import { debounce, el, on, replaceChildren } from '../lib/dom';
import { cycleTheme } from '../lib/theme';

interface Command {
  readonly id: string;
  readonly label: string;
  readonly hint?: string;
  readonly keywords: string;
  readonly run: () => void;
}

const STATIC_COMMANDS: readonly Command[] = [
  {
    id: 'nav-dashboard',
    label: 'Обзор',
    hint: 'Перейти',
    keywords: 'обзор дашборд dashboard главная',
    run: () => location.assign('/app'),
  },
  {
    id: 'nav-studies',
    label: 'Снимки',
    hint: 'Перейти',
    keywords: 'снимки studies рентген',
    run: () => location.assign('/app/studies'),
  },
  {
    id: 'nav-patients',
    label: 'Пациенты',
    hint: 'Перейти',
    keywords: 'пациенты patients карты',
    run: () => location.assign('/app/patients'),
  },
  {
    id: 'nav-settings',
    label: 'Настройки',
    hint: 'Перейти',
    keywords: 'настройки settings профиль пароль',
    run: () => location.assign('/app/settings'),
  },
  {
    id: 'action-theme',
    label: 'Переключить тему',
    hint: 'Действие',
    keywords: 'тема dark light темная светлая',
    run: () => cycleTheme(),
  },
  {
    id: 'action-logout',
    label: 'Выйти из аккаунта',
    hint: 'Действие',
    keywords: 'выйти logout выход',
    run: () => {
      void api.auth.logout().finally(() => location.assign('/login'));
    },
  },
];

function score(command: Command, query: string): number {
  if (!query) return 1;
  const haystack = `${command.label} ${command.keywords}`.toLowerCase();
  const needle = query.toLowerCase();
  if (haystack.startsWith(needle)) return 3;
  if (haystack.includes(needle)) return 2;
  // Subsequence match, so "прс" still finds "Пациенты — расширенный поиск".
  let index = 0;
  for (const char of haystack) {
    if (char === needle[index]) index += 1;
    if (index === needle.length) return 1;
  }
  return 0;
}

export function initCommandPalette(): void {
  const dialog = el('dialog', { class: 'palette dialog' }) as HTMLDialogElement;
  const input = el('input', {
    class: 'palette-input',
    type: 'search',
    placeholder: 'Команда, страница или пациент…',
    autocomplete: 'off',
    spellcheck: false,
    aria: { label: 'Поиск команд' },
  });
  const list = el('ul', { class: 'palette-list', role: 'listbox' });
  const footer = el(
    'div',
    { class: 'palette-footer' },
    el('kbd', {}, '↑↓'),
    el('span', {}, 'выбрать'),
    el('kbd', {}, '↵'),
    el('span', {}, 'открыть'),
    el('kbd', {}, 'esc'),
    el('span', {}, 'закрыть'),
  );

  dialog.append(
    el('div', { class: 'palette-header' }, input),
    list,
    footer,
  );
  document.body.append(dialog);

  let results: Command[] = [...STATIC_COMMANDS];
  let activeIndex = 0;
  let searchToken = 0;

  function render(): void {
    if (results.length === 0) {
      replaceChildren(list, el('li', { class: 'palette-empty' }, 'Ничего не найдено'));
      return;
    }

    replaceChildren(
      list,
      ...results.map((command, index) =>
        el(
          'li',
          {
            class: 'palette-item',
            role: 'option',
            dataset: { index: String(index) },
            aria: { selected: String(index === activeIndex) },
          },
          el('span', { class: 'palette-label' }, command.label),
          command.hint ? el('span', { class: 'palette-hint' }, command.hint) : null,
        ),
      ),
    );

    list
      .querySelector<HTMLElement>('[aria-selected="true"]')
      ?.scrollIntoView({ block: 'nearest' });
  }

  const search = debounce((query: string): void => {
    const token = ++searchToken;
    const matches = STATIC_COMMANDS.map((command) => ({
      command,
      rank: score(command, query),
    }))
      .filter((entry) => entry.rank > 0)
      .sort((a, b) => b.rank - a.rank)
      .map((entry) => entry.command);

    results = matches;
    activeIndex = 0;
    render();

    if (query.length < 2) return;

    // Patient results arrive asynchronously and are appended only if the
    // query has not moved on.
    void api.patients
      .list({ q: query, limit: 5 })
      .then((page) => {
        if (token !== searchToken) return;
        results = [
          ...matches,
          ...page.items.map<Command>((patient) => ({
            id: `patient-${patient.id}`,
            label: patient.fullName,
            hint: 'Пациент',
            keywords: patient.fullName,
            run: () => location.assign(`/app/studies?patient=${patient.id}`),
          })),
        ];
        render();
      })
      .catch(() => {
        /* Palette search failing is not worth a toast. */
      });
  }, 160);

  function open(): void {
    input.value = '';
    results = [...STATIC_COMMANDS];
    activeIndex = 0;
    render();
    dialog.showModal();
    input.focus();
  }

  function runActive(): void {
    const command = results[activeIndex];
    if (!command) return;
    dialog.close();
    command.run();
  }

  on(input, 'input', () => search(input.value.trim()));

  on(dialog, 'keydown', (event) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        activeIndex = (activeIndex + 1) % Math.max(results.length, 1);
        render();
        break;
      case 'ArrowUp':
        event.preventDefault();
        activeIndex = (activeIndex - 1 + results.length) % Math.max(results.length, 1);
        render();
        break;
      case 'Enter':
        event.preventDefault();
        runActive();
        break;
      default:
        break;
    }
  });

  on(list, 'click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const item = target.closest<HTMLElement>('.palette-item');
    const index = item?.dataset['index'];
    if (index !== undefined) {
      activeIndex = Number(index);
      runActive();
    }
  });

  // Clicking the backdrop closes; `<dialog>` reports those clicks as landing
  // on the dialog element itself.
  on(dialog, 'click', (event) => {
    if (event.target === dialog) dialog.close();
  });

  on(document.documentElement, 'keydown', (event) => {
    const isPaletteShortcut = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k';
    if (isPaletteShortcut) {
      event.preventDefault();
      dialog.open ? dialog.close() : open();
      return;
    }

    // "/" focuses search, but not while the user is typing in a field.
    if (event.key === '/' && !isEditable(event.target)) {
      event.preventDefault();
      open();
    }
  });

  for (const trigger of document.querySelectorAll<HTMLElement>('[data-open-palette]')) {
    on(trigger, 'click', open);
  }
}

function isEditable(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  );
}
