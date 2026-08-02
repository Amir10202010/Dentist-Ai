/**
 * Dashboard.
 *
 * The template already carries the layout and a skeleton with the same
 * geometry as the loaded state, so this module fills values in rather than
 * building the page: the dashboard does not reflow as it loads, and a failed
 * request degrades to a readable page with an error region.
 *
 * Counters animate to a number that is already in the DOM on the first frame,
 * and every animation checks `prefers-reduced-motion` first.
 */

import { api, studyAssets } from '../lib/api';
import { all, el, maybe, on, prefersReducedMotion, replaceChildren } from '../lib/dom';
import {
  daysSince,
  formatChange,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRelative,
  formatTime,
  plural,
} from '../lib/format';
import { icon, type IconName } from '../lib/icons';
import { notifyError } from '../lib/toast';
import type {
  ActivityItem,
  Dashboard,
  Insight,
  MetricDelta,
  ReviewQueueItem,
  StudyListItem,
  TimeSeriesPoint,
} from '../lib/types';
import { renderBreakdown, renderRing, renderSparkline, renderTrend } from './charts';

const RECENT_LIMIT = 4;
const SPARKLINE_DAYS = 14;
const DEFAULT_RANGE = 30;

/** Icon names arrive from the server as plain strings; this keeps them honest. */
const KNOWN_ICONS = new Set<string>([
  'upload',
  'check-circle',
  'pencil',
  'trash',
  'download',
  'user',
  'users',
  'archive',
  'lock',
  'shield',
  'alert-circle',
  'stethoscope',
  'activity',
  'trending-up',
  'trending-down',
  'target',
  'layers',
  'sparkles',
]);

function safeIcon(name: string, fallback: IconName, options?: { class?: string }): SVGSVGElement {
  return icon(KNOWN_ICONS.has(name) ? (name as IconName) : fallback, options ?? {});
}

// ---------------------------------------------------------------------------
// Animated counters
// ---------------------------------------------------------------------------

/**
 * Generation token per node. A refresh landing mid-animation must not leave
 * two rAF loops writing to the same element — the last one to start wins, and
 * earlier ones exit on their next frame.
 */
const counterGeneration = new WeakMap<HTMLElement, number>();

function countUp(node: HTMLElement, target: number, format: (value: number) => string): void {
  const generation = (counterGeneration.get(node) ?? 0) + 1;
  counterGeneration.set(node, generation);

  if (prefersReducedMotion() || target === 0) {
    node.textContent = format(target);
    return;
  }

  // Longer for larger numbers, but bounded: a four-digit count should feel
  // weightier than a single digit without making the reader wait for it.
  const duration = Math.min(900, 380 + Math.log10(target + 1) * 260);
  const started = performance.now();

  const frame = (now: number): void => {
    if (counterGeneration.get(node) !== generation) return;
    const progress = Math.min(1, (now - started) / duration);
    // Cubic ease-out: fast enough to read the magnitude immediately, slow
    // enough at the end that the final value settles rather than snapping.
    const eased = 1 - (1 - progress) ** 3;
    node.textContent = format(target * eased);
    if (progress < 1) requestAnimationFrame(frame);
  };

  node.textContent = format(0);
  requestAnimationFrame(frame);
}

function setCounter(key: string, value: number, format: (value: number) => string): void {
  const node = maybe(`[data-stat="${key}"]`);
  if (node) countUp(node, value, format);
}

function setText(selector: string, ...children: (Node | string | false | null)[]): void {
  const node = maybe(selector);
  if (node) replaceChildren(node, ...children);
}

const asInteger = (value: number): string => formatNumber(Math.round(value));
const asPercent = (value: number): string => `${Math.round(value)}%`;

// ---------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------

function emptyState(name: IconName, title: string, body: string): HTMLElement {
  return el(
    'div',
    { class: 'state state--inline' },
    el('div', { class: 'state-icon' }, icon(name, { class: 'icon--lg' })),
    el('p', { class: 'state-title' }, title),
    el('p', { class: 'state-body' }, body),
  );
}

/**
 * A delta chip: direction glyph, signed percentage, period.
 *
 * A change of zero gets no arrow and no sign — "+0%" with an upward arrow
 * reads as growth that did not happen. Growth from a zero baseline has no
 * percentage at all, so the absolute count is shown instead.
 */
function deltaChip(delta: MetricDelta, period = 'к прошлой неделе'): HTMLElement {
  if (delta.change === null) {
    const label =
      delta.current > 0
        ? `+${formatNumber(delta.current)} ${plural(delta.current, 'новый', 'новых', 'новых')}`
        : 'нет данных за прошлый период';
    return el('span', { class: 'delta', dataset: { direction: 'flat' } }, label);
  }

  const direction = delta.change > 0 ? 'up' : delta.change < 0 ? 'down' : 'flat';
  return el(
    'span',
    { class: 'delta', dataset: { direction } },
    direction !== 'flat' &&
      icon(direction === 'up' ? 'trending-up' : 'trending-down', { class: 'icon--sm' }),
    direction === 'flat' ? 'без изменений' : formatChange(delta.change),
    el('span', { class: 'delta-period' }, period),
  );
}

/** A horizontal fill used inside a KPI card, where a full chart is too much. */
function miniMeter(share: number, tone: string): HTMLElement {
  const fill = el('div', {
    class: 'meter-fill',
    style: { '--meter-color': tone, 'inline-size': '0%' },
  });
  const width = `${Math.round(Math.min(1, Math.max(0, share)) * 100)}%`;
  if (prefersReducedMotion()) {
    fill.style.setProperty('inline-size', width);
  } else {
    requestAnimationFrame(() => fill.style.setProperty('inline-size', width));
  }
  return el('div', { class: 'meter meter--mini', role: 'presentation' }, fill);
}

function setVisual(key: string, node: Node): void {
  const slot = maybe(`[data-visual="${key}"]`);
  if (slot) replaceChildren(slot, node);
}

// ---------------------------------------------------------------------------
// Header pulse
// ---------------------------------------------------------------------------

function renderPulse(data: Dashboard): void {
  const container = maybe('[data-pulse]');
  if (!container) return;

  const { pipeline } = data;
  const pills: HTMLElement[] = [
    el(
      'span',
      { class: 'pulse-pill' },
      icon('scan', { class: 'icon--sm' }),
      `${formatNumber(pipeline.completedToday)} ${plural(pipeline.completedToday, 'снимок', 'снимка', 'снимков')} сегодня`,
    ),
  ];

  const inFlight = pipeline.processing + pipeline.pending;
  if (inFlight > 0) {
    pills.push(
      el(
        'span',
        { class: 'pulse-pill', dataset: { tone: 'info' } },
        el('span', { class: 'pulse-dot' }),
        `${formatNumber(inFlight)} в обработке`,
      ),
    );
  }
  if (pipeline.failedRecent > 0) {
    pills.push(
      el(
        'span',
        { class: 'pulse-pill', dataset: { tone: 'critical' } },
        icon('alert-circle', { class: 'icon--sm' }),
        `${formatNumber(pipeline.failedRecent)} с ошибкой`,
      ),
    );
  }
  pills.push(
    el('span', { class: 'pulse-pill pulse-pill--quiet' }, `Обновлено ${formatTime(data.generatedAt)}`),
  );

  replaceChildren(container, ...pills);
  container.hidden = false;
}

// ---------------------------------------------------------------------------
// KPIs
// ---------------------------------------------------------------------------

function renderKpis(data: Dashboard): void {
  const { reviewStats } = data;

  setCounter('studies-week', data.studiesThisWeek, asInteger);
  setVisual(
    'studies-spark',
    renderSparkline(data.studiesOverTime.slice(-SPARKLINE_DAYS), { tone: 'var(--accent)' }),
  );
  setText(
    '[data-foot="studies"]',
    deltaChip(data.studiesDelta),
    el('span', { class: 'kpi-foot-note' }, `${formatNumber(data.totalStudies)} всего`),
  );

  setCounter('pending', data.pendingFindings, asInteger);
  // Of the clinic's pathology findings, how many still await a decision. The
  // count alone says how many; the share says whether that is a backlog.
  const pendingShare =
    data.findingsNeedingAttention > 0 ? data.pendingFindings / data.findingsNeedingAttention : 0;
  setVisual('pending-meter', miniMeter(pendingShare, 'var(--severity-critical)'));
  const waited = data.oldestPendingAt === null ? 0 : daysSince(data.oldestPendingAt);
  setText(
    '[data-foot="pending"]',
    data.pendingFindings === 0
      ? el('span', { class: 'delta', dataset: { direction: 'good' } }, 'очередь пуста')
      : el(
          'span',
          { class: 'kpi-foot-note' },
          `в ${formatNumber(data.reviewQueueTotal)} ${plural(data.reviewQueueTotal, 'снимке', 'снимках', 'снимках')}`,
        ),
    data.pendingFindings > 0 &&
      waited > 0 &&
      el(
        'span',
        { class: 'kpi-foot-note' },
        `ждёт ${waited} ${plural(waited, 'день', 'дня', 'дней')}`,
      ),
  );

  setCounter('reviewed-share', data.reviewedShare * 100, asPercent);
  setVisual('reviewed-ring', renderRing(data.reviewedShare, { tone: 'var(--success)' }));
  const decided = reviewStats.confirmed + reviewStats.rejected;
  const totalFindings = decided + reviewStats.unreviewed;
  setText(
    '[data-foot="reviewed"]',
    el(
      'span',
      { class: 'kpi-foot-note' },
      `${formatNumber(decided)} из ${formatNumber(totalFindings)} ${plural(totalFindings, 'находки', 'находок', 'находок')}`,
    ),
  );

  setCounter('patients-total', data.totalPatients, asInteger);
  setVisual(
    'patients-meter',
    miniMeter(
      data.totalPatients > 0 ? data.newPatientsThisWeek / data.totalPatients : 0,
      'var(--category-orthodontic)',
    ),
  );
  setText(
    '[data-foot="patients"]',
    deltaChip(data.patientsDelta),
    el(
      'span',
      { class: 'kpi-foot-note' },
      `+${formatNumber(data.newPatientsThisWeek)} за неделю`,
    ),
  );
}

// ---------------------------------------------------------------------------
// Insights
// ---------------------------------------------------------------------------

function insightNode(item: Insight, expanded: boolean): HTMLElement {
  const panelId = `insight-${item.key}`;

  const detail = el(
    'div',
    { class: 'insight-detail', id: panelId, hidden: !expanded },
    el('p', { class: 'insight-body' }, item.body),
    item.actionLabel !== null &&
      item.actionHref !== null &&
      el(
        'a',
        { class: 'btn btn--sm', href: item.actionHref },
        item.actionLabel,
        icon('chevron-right', { class: 'icon--sm' }),
      ),
  );

  const toggle = el(
    'button',
    { type: 'button', class: 'insight-toggle', aria: { expanded } },
    el('span', { class: 'insight-icon' }, safeIcon(item.icon, 'sparkles', { class: 'icon--sm' })),
    el(
      'span',
      { class: 'insight-heading' },
      el('span', { class: 'insight-title' }, item.title),
      item.metric !== null && el('span', { class: 'insight-metric' }, item.metric),
    ),
    el('span', { class: 'insight-chevron' }, icon('chevron-down', { class: 'icon--sm' })),
  );
  toggle.setAttribute('aria-controls', panelId);

  on(toggle, 'click', () => {
    const next = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(next));
    detail.hidden = !next;
  });

  return el('li', { class: 'insight', dataset: { tone: item.tone } }, toggle, detail);
}

function renderInsights(container: HTMLElement, items: readonly Insight[]): void {
  if (items.length === 0) {
    replaceChildren(
      container,
      emptyState(
        'sparkles',
        'Пока нечего отметить',
        'Выводы появятся, когда в клинике накопятся снимки и решения врачей.',
      ),
    );
    return;
  }
  // The first is the most urgent — the server sorts by tone — so it opens by
  // default. The rest stay one click away rather than filling the column.
  replaceChildren(container, ...items.map((item, index) => insightNode(item, index === 0)));
}

// ---------------------------------------------------------------------------
// Review queue
// ---------------------------------------------------------------------------

function queueNode(item: ReviewQueueItem): HTMLElement {
  const waited = daysSince(item.createdAt);
  const meta = [
    item.topFindingLabel,
    `${Math.round(item.topConfidence * 100)}%`,
    waited > 0
      ? `ждёт ${waited} ${plural(waited, 'день', 'дня', 'дней')}`
      : formatRelative(item.createdAt),
  ].join(' · ');

  return el(
    'li',
    {},
    el(
      'a',
      { class: 'queue-item', href: studyAssets.page(item.publicId) },
      el('img', {
        class: 'queue-thumb',
        src: studyAssets.thumbnail(item.publicId),
        alt: '',
        loading: 'lazy',
        decoding: 'async',
        width: 44,
        height: 44,
      }),
      el(
        'span',
        { class: 'queue-text' },
        el('span', { class: 'queue-title' }, item.patientName ?? item.originalFilename),
        el('span', { class: 'queue-meta' }, meta),
      ),
      el(
        'span',
        { class: 'queue-side' },
        el(
          'span',
          { class: 'badge badge--severity', dataset: { severity: item.topSeverity } },
          el('span', { class: 'dot' }),
          item.topSeverityLabel,
        ),
        el(
          'span',
          { class: 'queue-count', title: 'Находок ждут решения' },
          formatNumber(item.pendingCount),
        ),
      ),
      icon('chevron-right', { class: 'icon--sm icon--muted' }),
    ),
  );
}

function renderQueue(container: HTMLElement, data: Dashboard): void {
  const subtitle = maybe('[data-queue-subtitle]');
  if (subtitle) {
    subtitle.textContent =
      data.reviewQueueTotal === 0
        ? 'Все патологии разобраны'
        : `${formatNumber(data.reviewQueueTotal)} ${plural(data.reviewQueueTotal, 'снимок ждёт', 'снимка ждут', 'снимков ждут')} решения врача`;
  }

  if (data.reviewQueue.length === 0) {
    replaceChildren(
      container,
      emptyState(
        'check-circle',
        data.totalStudies === 0 ? 'Снимков пока нет' : 'Очередь пуста',
        data.totalStudies === 0
          ? 'Загрузите первую рентгенограмму — находки, требующие решения, появятся здесь.'
          : 'Каждая находка категории «Патологии» получила решение врача.',
      ),
    );
    return;
  }

  replaceChildren(container, ...data.reviewQueue.map(queueNode));

  if (data.reviewQueueTotal > data.reviewQueue.length) {
    const rest = data.reviewQueueTotal - data.reviewQueue.length;
    container.append(
      el(
        'li',
        { class: 'queue-more' },
        el(
          'a',
          { class: 'btn btn--sm btn--ghost btn--block', href: '/app/studies' },
          `Ещё ${formatNumber(rest)} ${plural(rest, 'снимок', 'снимка', 'снимков')}`,
          icon('chevron-right', { class: 'icon--sm' }),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Activity
// ---------------------------------------------------------------------------

/** Actions whose subject is a study that still exists and can be opened. */
const LINKABLE_ACTIONS = new Set(['study.uploaded', 'study.updated', 'finding.reviewed']);

function activityNode(item: ActivityItem): HTMLElement {
  const linkable =
    item.resourceType === 'study' &&
    item.resourceId !== null &&
    LINKABLE_ACTIONS.has(item.action);

  const body = el(
    'span',
    { class: 'timeline-body' },
    el('span', { class: 'timeline-summary' }, item.summary),
    el(
      'span',
      { class: 'timeline-meta' },
      item.actorName ?? 'Система',
      el('span', { class: 'timeline-sep' }, '·'),
      formatRelative(item.createdAt),
    ),
  );

  return el(
    'li',
    { class: 'timeline-item', dataset: { tone: item.tone } },
    el('span', { class: 'timeline-marker' }, safeIcon(item.icon, 'activity', { class: 'icon--sm' })),
    linkable && item.resourceId !== null
      ? el('a', { class: 'timeline-link', href: studyAssets.page(item.resourceId) }, body)
      : body,
  );
}

function renderActivity(container: HTMLElement, items: readonly ActivityItem[]): void {
  if (items.length === 0) {
    replaceChildren(
      container,
      emptyState(
        'clock',
        'Пока тихо',
        'Загрузки, проверки находок и изменения карт появятся здесь.',
      ),
    );
    return;
  }
  replaceChildren(container, ...items.map(activityNode));
}

// ---------------------------------------------------------------------------
// Recent studies
// ---------------------------------------------------------------------------

function recentTile(study: StudyListItem): HTMLElement {
  return el(
    'li',
    {},
    el(
      'a',
      { class: 'recent-tile', href: studyAssets.page(study.publicId) },
      el(
        'span',
        { class: 'recent-tile-media' },
        el('img', {
          src: study.thumbnailUrl,
          alt: '',
          loading: 'lazy',
          decoding: 'async',
          width: 160,
          height: 100,
        }),
        study.attentionCount > 0 &&
          el(
            'span',
            {
              class: 'badge badge--severity recent-tile-badge',
              dataset: { severity: study.topSeverity ?? 'high' },
            },
            formatNumber(study.attentionCount),
          ),
      ),
      el('span', { class: 'recent-tile-title' }, study.patientName ?? study.originalFilename),
      el(
        'span',
        { class: 'recent-tile-meta' },
        `${formatNumber(study.findingCount)} ${plural(study.findingCount, 'находка', 'находки', 'находок')} · ${formatRelative(study.createdAt)}`,
      ),
    ),
  );
}

function renderRecent(container: HTMLElement, studies: readonly StudyListItem[]): void {
  if (studies.length === 0) {
    replaceChildren(
      container,
      el(
        'li',
        { class: 'recent-empty' },
        emptyState(
          'inbox',
          'Снимков пока нет',
          'Как только вы загрузите первую рентгенограмму, она появится здесь.',
        ),
      ),
    );
    return;
  }
  replaceChildren(container, ...studies.map(recentTile));
}

// ---------------------------------------------------------------------------
// Findings panel
// ---------------------------------------------------------------------------

function renderTopFindings(container: HTMLElement, data: Dashboard): void {
  const items = data.topFindings.filter((item) => item.count > 0);
  if (items.length === 0) {
    replaceChildren(
      container,
      emptyState(
        'stethoscope',
        'Пока нет находок',
        'Загрузите снимок — здесь появится сводка по самым частым находкам.',
      ),
    );
    return;
  }

  const max = Math.max(...items.map((item) => item.count));
  replaceChildren(
    container,
    ...items.map((item) =>
      el(
        'li',
        { class: 'finding-summary' },
        el('span', {
          class: 'dot',
          style: { color: `var(--severity-${item.severity ?? 'info'})` },
        }),
        el('span', { class: 'finding-summary-label' }, item.label),
        // A bar behind the count turns a column of numbers into a ranking that
        // can be read without comparing digits.
        el(
          'span',
          { class: 'finding-summary-bar' },
          el('span', {
            class: 'finding-summary-fill',
            style: {
              'inline-size': `${Math.max(6, (item.count / max) * 100)}%`,
              '--meter-color': `var(--severity-${item.severity ?? 'info'})`,
            },
          }),
        ),
        el('span', { class: 'finding-summary-count' }, formatNumber(item.count)),
      ),
    ),
  );
}

function renderFindings(data: Dashboard): void {
  const categories = maybe('[data-chart="categories"]');
  if (categories) renderBreakdown(categories, data.categoryBreakdown, { colorBy: 'category' });
  const top = maybe('[data-top-findings]');
  if (top) renderTopFindings(top, data);
}

function selectFindingsTab(key: string): void {
  for (const tab of all<HTMLButtonElement>('[data-findings-tab]')) {
    tab.setAttribute('aria-selected', String(tab.dataset['findingsTab'] === key));
  }
  for (const [name, selector] of [
    ['categories', '[data-chart="categories"]'],
    ['top', '[data-top-findings]'],
  ] as const) {
    const panel = maybe(selector);
    if (panel) panel.hidden = name !== key;
  }

  const subtitle = maybe('[data-findings-subtitle]');
  if (subtitle) {
    subtitle.textContent =
      key === 'categories' ? 'Доля от всех находок клиники' : 'Самые частые, в порядке значимости';
  }
}

// ---------------------------------------------------------------------------
// Trend
// ---------------------------------------------------------------------------

/**
 * Draw the last `days` of the series the server already sent.
 *
 * The payload carries 90 days precisely so the range control is a slice rather
 * than a request: switching period is instant, and a clinic flicking between
 * views does not generate three round-trips.
 */
function selectRange(points: readonly TimeSeriesPoint[], days: number): void {
  for (const button of all<HTMLButtonElement>('[data-range]')) {
    button.setAttribute('aria-pressed', String(Number(button.dataset['range']) === days));
  }

  const container = maybe('[data-chart="trend"]');
  if (!container) return;

  const sliced = points.slice(-days);
  renderTrend(container, sliced);

  const total = maybe('[data-trend-total]');
  if (total) {
    const sum = Math.round(sliced.reduce((carry, point) => carry + point.value, 0));
    const perWeek = sliced.length > 0 ? (sum / sliced.length) * 7 : 0;
    total.textContent =
      sum === 0
        ? `Ни одного снимка за ${days} дней`
        : `${formatNumber(sum)} ${plural(sum, 'снимок', 'снимка', 'снимков')} · в среднем ${perWeek.toFixed(1)} в неделю`;
  }
}

function activeRange(): number {
  const active = all<HTMLButtonElement>('[data-range]').find(
    (button) => button.getAttribute('aria-pressed') === 'true',
  );
  return Number(active?.dataset['range']) || DEFAULT_RANGE;
}

function activeFindingsTab(): string {
  const active = all<HTMLButtonElement>('[data-findings-tab]').find(
    (tab) => tab.getAttribute('aria-selected') === 'true',
  );
  return active?.dataset['findingsTab'] ?? 'categories';
}

/**
 * The most recent payload, so the controls can redraw without refetching.
 *
 * Held here rather than passed into the handlers because the handlers are
 * bound once, at init: binding them inside `load` added a listener per refresh,
 * and after five refreshes one click on "7 дн." redrew the chart five times.
 */
let latest: Dashboard | null = null;

function bindControls(): void {
  for (const button of all<HTMLButtonElement>('[data-range]')) {
    on(button, 'click', () => {
      if (latest) selectRange(latest.studiesOverTime, Number(button.dataset['range']) || DEFAULT_RANGE);
    });
  }

  for (const tab of all<HTMLButtonElement>('[data-findings-tab]')) {
    on(tab, 'click', () => selectFindingsTab(tab.dataset['findingsTab'] ?? 'categories'));
  }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Put every value slot back into its loading state before a refresh.
 *
 * The placeholder is a *child* span, never a class on the slot itself: writing
 * a value replaces the children, so the shimmer cannot outlive the load. An
 * earlier version toggled `.skeleton` on the slot and left the micro-stats
 * rendering as solid grey blocks, because `color: transparent` survived the
 * text being written into them.
 */
function showSkeletons(): void {
  for (const node of all('[data-stat]')) {
    if (node.querySelector('.skeleton')) continue;
    const small = node.tagName === 'DD';
    replaceChildren(
      node,
      el('span', {
        class: small ? 'skeleton skeleton--line' : 'skeleton skeleton--value',
        style: small ? { 'inline-size': '3.5rem', display: 'block' } : {},
      }),
    );
  }
}

/**
 * Fade the cards in, once, as soon as the module runs.
 *
 * Before the fetch, not after it: the template ships a skeleton with the same
 * geometry as the loaded page, and holding the cards at zero opacity until the
 * data arrived hid exactly the thing the skeleton exists to show. The reveal
 * is an entrance for the layout; the data lands into it afterwards.
 */
function reveal(): void {
  // Not `.dash-grid > *`: at one column the column wrappers are
  // `display: contents` and the cards are the grid items, so the two would
  // disagree about what to animate.
  const cards = all('.dash-kpis, .dash-card');
  if (prefersReducedMotion()) {
    for (const card of cards) card.classList.add('is-revealed');
    return;
  }
  cards.forEach((card, index) => {
    card.style.setProperty('--reveal-delay', `${Math.min(index * 45, 240)}ms`);
  });
  requestAnimationFrame(() => {
    for (const card of cards) card.classList.add('is-revealed');
  });
}

async function load(signal?: AbortSignal): Promise<void> {
  showSkeletons();
  const errorRegion = maybe('[data-dashboard-error]');
  if (errorRegion) errorRegion.hidden = true;

  let data: Dashboard;
  try {
    data = await api.dashboard(signal);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return;
    notifyError(error);
    if (errorRegion) errorRegion.hidden = false;
    return;
  }

  latest = data;
  renderPulse(data);
  renderKpis(data);

  setText(
    '[data-stat="avg-inference"]',
    formatDuration(data.averageInferenceMs),
  );
  setText(
    '[data-stat="avg-confidence"]',
    data.reviewStats.averageConfidence === null
      ? '—'
      : formatPercent(data.reviewStats.averageConfidence),
  );
  setText(
    '[data-stat="agreement"]',
    data.reviewStats.agreementRate === null ? '—' : formatPercent(data.reviewStats.agreementRate),
  );

  // Redraw at whatever period and tab the reader had chosen: a background
  // refresh must not throw them back to the defaults.
  selectRange(data.studiesOverTime, activeRange());
  renderFindings(data);
  selectFindingsTab(activeFindingsTab());

  const insights = maybe('[data-insights]');
  if (insights) renderInsights(insights, data.insights);

  const queue = maybe('[data-review-queue]');
  if (queue) renderQueue(queue, data);

  const activity = maybe('[data-activity]');
  if (activity) renderActivity(activity, data.activity);

  // Sequential: the recent list is supporting detail, and firing
  // it alongside the dashboard request only makes the primary numbers land
  // later on a slow connection.
  const recent = maybe('[data-recent-studies]');
  if (recent) {
    try {
      const page = await api.studies.list({ limit: RECENT_LIMIT, offset: 0 }, signal);
      renderRecent(recent, page.items);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      replaceChildren(
        recent,
        el(
          'li',
          { class: 'recent-empty' },
          emptyState('alert-circle', 'Список недоступен', 'Не удалось загрузить последние снимки.'),
        ),
      );
    }
  }
}

export async function initDashboard(signal?: AbortSignal): Promise<void> {
  reveal();
  bindControls();

  const retry = maybe('[data-dashboard-retry]');
  if (retry) on(retry, 'click', () => void load(signal));

  const refresh = maybe<HTMLButtonElement>('[data-dashboard-refresh]');
  if (refresh) {
    on(refresh, 'click', () => {
      refresh.dataset['spinning'] = 'true';
      void load(signal).finally(() => {
        delete refresh.dataset['spinning'];
      });
    });
  }

  await load(signal);
}
