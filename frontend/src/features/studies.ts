/**
 * Studies list: search, pagination, upload, empty and error states.
 *
 * Every request is cancellable, so typing quickly cannot land results out of
 * order.
 */

import { api } from '../lib/api';
import { debounce, el, maybe, must, on, replaceChildren } from '../lib/dom';
import { formatRelative } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { Page, StudyListItem } from '../lib/types';
import { mountUpload } from './upload';

const PAGE_SIZE = 24;

interface ListState {
  query: string;
  offset: number;
  loading: boolean;
}

export function initStudiesPage(): void {
  const grid = must('[data-studies-grid]');
  const searchInput = maybe<HTMLInputElement>('[data-studies-search]');
  const loadMore = maybe<HTMLButtonElement>('[data-load-more]');
  const summary = maybe('[data-studies-summary]');

  const state: ListState = { query: '', offset: 0, loading: false };
  let inFlight: AbortController | null = null;

  function card(study: StudyListItem): HTMLElement {
    /*
     * One badge, not two. Showing "4 требуют внимания" and
     * "Критично" side by side, both in the same red — two chips encoding the
     * same fact, which reads as noise and steals the eye from the count that
     * actually differs between cards.
     */
    const attention =
      study.attentionCount > 0
        ? el(
            'span',
            {
              class: 'badge badge--severity',
              dataset: { severity: study.topSeverity ?? 'medium' },
            },
            el('span', { class: 'dot' }),
            study.topSeverityLabel
              ? `${study.topSeverityLabel} · ${study.attentionCount}`
              : `Требуют внимания · ${study.attentionCount}`,
          )
        : el(
            'span',
            { class: 'badge badge--success' },
            icon('check', { class: 'icon--sm' }),
            'Без патологий',
          );

    return el(
      'a',
      {
        class: 'study-card card card--interactive',
        href: `/app/studies/${study.publicId}`,
      },
      el(
        'div',
        { class: 'study-card-media' },
        el('img', {
          src: study.thumbnailUrl,
          alt: '',
          loading: 'lazy',
          decoding: 'async',
          // Reserving the box prevents the grid from reflowing as thumbnails
          // stream in.
          width: 320,
          height: 200,
        }),
      ),
      el(
        'div',
        { class: 'study-card-body' },
        el(
          'p',
          { class: 'study-card-title' },
          study.patientName ?? study.originalFilename,
        ),
        el(
          'p',
          { class: 'study-card-meta' },
          `${study.findingCount} находок · ${formatRelative(study.createdAt)}`,
        ),
        el('div', { class: 'cluster' }, attention),
      ),
    );
  }

  function skeletons(count: number): readonly HTMLElement[] {
    return Array.from({ length: count }, () =>
      el(
        'div',
        { class: 'study-card card' },
        el('div', { class: 'study-card-media skeleton' }),
        el(
          'div',
          { class: 'study-card-body stack' },
          el('div', { class: 'skeleton skeleton--line' }),
          el('div', { class: 'skeleton skeleton--text', style: { width: '60%' } }),
        ),
      ),
    );
  }

  function emptyState(): HTMLElement {
    const searching = state.query.length > 0;
    return el(
      'div',
      { class: 'state' },
      el(
        'div',
        { class: 'state-icon' },
        icon(searching ? 'search' : 'scan', { class: 'icon--lg' }),
      ),
      el(
        'p',
        { class: 'state-title' },
        searching ? 'Ничего не найдено' : 'Снимков пока нет',
      ),
      el(
        'p',
        { class: 'state-body' },
        searching
          ? `По запросу «${state.query}» ничего нет. Попробуйте имя пациента или ID снимка.`
          : 'Перетащите рентгеновский снимок в область загрузки — анализ займёт пару секунд.',
      ),
      searching
        ? el(
            'button',
            {
              class: 'btn',
              type: 'button',
              onclick: () => {
                if (searchInput) searchInput.value = '';
                state.query = '';
                void load({ reset: true });
              },
            },
            'Сбросить поиск',
          )
        : null,
    );
  }

  function errorState(retry: () => void): HTMLElement {
    return el(
      'div',
      { class: 'state state--error' },
      el('div', { class: 'state-icon' }, icon('alert', { class: 'icon--lg' })),
      el('p', { class: 'state-title' }, 'Не удалось загрузить снимки'),
      el('p', { class: 'state-body' }, 'Проверьте подключение и попробуйте ещё раз.'),
      el(
        'button',
        { class: 'btn btn--sm', type: 'button', onclick: retry },
        icon('refresh', { class: 'icon--sm' }),
        'Повторить',
      ),
    );
  }

  async function load({ reset }: { reset: boolean }): Promise<void> {
    if (state.loading) inFlight?.abort();
    state.loading = true;
    if (reset) state.offset = 0;

    if (reset) replaceChildren(grid, ...skeletons(6));
    if (loadMore) loadMore.hidden = true;

    inFlight = new AbortController();
    let page: Page<StudyListItem>;
    try {
      page = await api.studies.list(
        { q: state.query, limit: PAGE_SIZE, offset: state.offset },
        inFlight.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      state.loading = false;
      replaceChildren(grid, errorState(() => void load({ reset: true })));
      notifyError(error);
      return;
    }

    state.loading = false;

    if (reset) grid.replaceChildren();
    if (page.items.length === 0 && state.offset === 0) {
      replaceChildren(grid, emptyState());
    } else {
      grid.append(...page.items.map(card));
    }

    if (summary) {
      summary.textContent =
        page.meta.total === 0 ? '' : `Показано ${grid.childElementCount} из ${page.meta.total}`;
    }

    if (loadMore) {
      loadMore.hidden = !page.meta.hasMore;
      state.offset += page.items.length;
    }
  }

  if (searchInput) {
    on(
      searchInput,
      'input',
      debounce(() => {
        state.query = searchInput.value.trim();
        void load({ reset: true });
      }, 250),
    );
  }

  if (loadMore) {
    on(loadMore, 'click', () => void load({ reset: false }));
  }

  mountUpload((study) => {
    notify.info('Открываем результаты…');
    location.assign(`/app/studies/${study.publicId}`);
  });

  void load({ reset: true });
}
