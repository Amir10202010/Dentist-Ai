/**
 * Charts, drawn as inline SVG — no charting library, which also keeps
 * `script-src 'self'` intact.
 *
 * * **Bars, not a line.** Daily upload counts are discrete events on a sparse
 *   series; joining them with a line would claim a value for every instant
 *   between two days. The sparklines follow the same rule at a smaller size.
 * * **Rendered at pixel size, not stretched.** Measuring the container and
 *   drawing 1 unit = 1 px keeps strokes even and type upright, at the cost of
 *   a redraw on resize — which a ResizeObserver makes cheap.
 * * **A readout, not a `<title>`.** The native SVG tooltip appears after a
 *   browser-controlled delay, unstyled. The hover readout tracks the pointer
 *   immediately; `<title>` stays as the accessible fallback.
 */

import { el, prefersReducedMotion } from '../lib/dom';
import { formatNumber, parseInstant, plural } from '../lib/format';
import type { LabelledCount, Severity, TimeSeriesPoint } from '../lib/types';

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Readonly<Record<string, string | number>> = {},
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

/** Tracks the observer per container so a re-render never stacks another one. */
const observers = new WeakMap<HTMLElement, ResizeObserver>();

function onResize(container: HTMLElement, draw: () => void): void {
  observers.get(container)?.disconnect();

  let frame = 0;
  let lastWidth = container.clientWidth;
  const observer = new ResizeObserver(() => {
    // Width is the only dimension the layout derives from; ignoring height
    // changes stops the observer reacting to its own redraw.
    if (container.clientWidth === lastWidth) return;
    lastWidth = container.clientWidth;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  });
  observer.observe(container);
  observers.set(container, observer);
}

/** "Nice" axis maximum, so gridline labels are round numbers. */
function niceMax(value: number): number {
  if (value <= 4) return Math.max(1, value);
  const magnitude = 10 ** Math.floor(Math.log10(value));
  return Math.ceil(value / magnitude) * magnitude;
}

const locale = document.documentElement.lang || 'ru';
const dayFormatter = new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short' });
const weekdayFormatter = new Intl.DateTimeFormat(locale, { weekday: 'long' });

// ---------------------------------------------------------------------------
// Upload trend
// ---------------------------------------------------------------------------

/**
 * Daily upload counts as a bar chart, with a value axis, dated ticks and a
 * pointer-tracking readout.
 */
export function renderTrend(container: HTMLElement, points: readonly TimeSeriesPoint[]): void {
  if (points.length === 0) {
    container.replaceChildren(emptyNote('Пока нет данных'));
    return;
  }

  const readout = el('div', { class: 'chart-readout', hidden: true });

  const draw = (): void => {
    const width = Math.max(240, container.clientWidth);
    const height = 208;
    const pad = { top: 14, right: 6, bottom: 26, left: 34 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const max = niceMax(Math.max(1, ...points.map((point) => point.value)));
    const slot = plotW / points.length;
    // Cap the bar width so a 7-day range does not render as fat slabs, and
    // floor it so a 90-day range stays visible.
    const barW = Math.max(2, Math.min(slot - 2, 22));
    const animate = !prefersReducedMotion();

    const svg = svgEl('svg', {
      viewBox: `0 0 ${width} ${height}`,
      width,
      height,
      class: 'chart',
      role: 'img',
      'aria-label': `Загрузки по дням за ${points.length} дней, максимум ${max} за день`,
    });

    // Gridlines at 0 / mid / max. Three lines is enough to read a value off
    // the chart and few enough to stay out of the way of the bars.
    for (const step of [0, 0.5, 1]) {
      const y = pad.top + plotH - step * plotH;
      const label = svgEl('text', {
        x: pad.left - 8,
        y: y + 4,
        class: 'chart-tick',
        'text-anchor': 'end',
      });
      label.textContent = formatNumber(Math.round(step * max));
      svg.append(
        svgEl('line', {
          x1: pad.left,
          y1: y,
          x2: pad.left + plotW,
          y2: y,
          class: step === 0 ? 'chart-axis' : 'chart-grid',
        }),
        label,
      );
    }

    points.forEach((point, index) => {
      const x = pad.left + index * slot + (slot - barW) / 2;
      const barH = (point.value / max) * plotH;
      const group = svgEl('g', { class: 'chart-bar-group' });
      group.dataset['index'] = String(index);

      // A full-height transparent target: hovering a 2px-tall zero bar is
      // otherwise impossible, and the readout is most useful on quiet days.
      group.append(
        svgEl('rect', {
          x: pad.left + index * slot,
          y: pad.top,
          width: slot,
          height: plotH,
          class: 'chart-bar-hit',
        }),
      );

      const zero = point.value === 0;
      const barHeight = zero ? 2 : Math.max(3, barH);
      const bar = svgEl('rect', {
        x,
        y: pad.top + plotH - barHeight,
        width: barW,
        height: barHeight,
        rx: Math.min(3, barW / 2),
        // A hairline keeps a zero day present on the axis instead of leaving a
        // gap that reads as missing data.
        class: zero ? 'chart-bar chart-bar--zero' : 'chart-bar',
      });
      if (animate) {
        // Grow from the baseline. `transform-origin` is set in CSS to the
        // bottom of the plot so every bar rises from the same line.
        bar.style.setProperty('--bar-delay', `${Math.min(index * 12, 320)}ms`);
        bar.classList.add('chart-bar--enter');
      }
      group.append(bar);

      const title = svgEl('title');
      title.textContent = `${dayFormatter.format(parseInstant(point.date))} — ${formatNumber(point.value)}`;
      group.append(title);
      svg.append(group);
    });

    // Dated ticks at the ends only. Labelling every day collides; labelling
    // the extremes is what a reader actually needs to place the range.
    const first = points[0];
    const last = points[points.length - 1];
    for (const [point, anchor, x] of [
      [first, 'start', pad.left],
      [last, 'end', pad.left + plotW],
    ] as const) {
      if (!point) continue;
      const tick = svgEl('text', {
        x,
        y: height - 8,
        class: 'chart-tick',
        'text-anchor': anchor,
      });
      tick.textContent = dayFormatter.format(parseInstant(point.date));
      svg.append(tick);
    }

    container.replaceChildren(svg, readout);
    bindReadout(container, svg, points, readout);
  };

  draw();
  onResize(container, draw);
}

function bindReadout(
  container: HTMLElement,
  svg: SVGSVGElement,
  points: readonly TimeSeriesPoint[],
  readout: HTMLElement,
): void {
  const show = (event: PointerEvent): void => {
    const target = event.target;
    const group = target instanceof Element ? target.closest('.chart-bar-group') : null;
    if (!(group instanceof SVGGElement)) return;

    const index = Number(group.dataset['index']);
    const point = points[index];
    if (!point) return;

    const date = parseInstant(point.date);
    const count = Math.round(point.value);
    readout.replaceChildren(
      el('span', { class: 'chart-readout-value' }, formatNumber(count)),
      el(
        'span',
        { class: 'chart-readout-label' },
        `${plural(count, 'снимок', 'снимка', 'снимков')} · ${dayFormatter.format(date)}, ${weekdayFormatter.format(date)}`,
      ),
    );
    readout.hidden = false;

    // Positioned against the container, then clamped so the readout never
    // hangs off the card at either end of the series.
    const bounds = container.getBoundingClientRect();
    const bar = group.getBoundingClientRect();
    const centre = bar.left + bar.width / 2 - bounds.left;
    const half = readout.offsetWidth / 2;
    const clamped = Math.min(Math.max(centre, half + 4), bounds.width - half - 4);
    readout.style.setProperty('--readout-x', `${clamped}px`);
    for (const active of svg.querySelectorAll('.is-active')) active.classList.remove('is-active');
    group.classList.add('is-active');
  };

  const hide = (): void => {
    readout.hidden = true;
    for (const active of svg.querySelectorAll('.is-active')) active.classList.remove('is-active');
  };

  svg.addEventListener('pointermove', show);
  svg.addEventListener('pointerleave', hide);
  // Touch has no hover: a tap shows the readout, and leaving the chart or
  // scrolling dismisses it.
  svg.addEventListener('pointerdown', show);
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

/**
 * A KPI card's shape-at-a-glance: micro bars, no axes, no labels.
 *
 * Drawn on a fixed viewBox and scaled by CSS, unlike the trend chart — at this
 * size there is no type to keep upright and no stroke thin enough to shimmer,
 * and a fixed box means no measurement pass before first paint.
 */
export function renderSparkline(
  points: readonly TimeSeriesPoint[],
  options: { readonly tone?: string } = {},
): SVGSVGElement {
  const width = 96;
  const height = 30;
  const svg = svgEl('svg', {
    viewBox: `0 0 ${width} ${height}`,
    class: 'sparkline',
    preserveAspectRatio: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  });
  if (options.tone) svg.style.setProperty('--spark-tone', options.tone);
  if (points.length === 0) return svg;

  const max = Math.max(1, ...points.map((point) => point.value));
  const slot = width / points.length;
  const barW = Math.max(1.5, slot - 1.5);

  points.forEach((point, index) => {
    const barH = Math.max(1.5, (point.value / max) * (height - 2));
    svg.append(
      svgEl('rect', {
        x: index * slot + (slot - barW) / 2,
        y: height - barH,
        width: barW,
        height: barH,
        rx: Math.min(1.5, barW / 2),
        class: point.value === 0 ? 'sparkline-bar sparkline-bar--zero' : 'sparkline-bar',
      }),
    );
  });
  return svg;
}

// ---------------------------------------------------------------------------
// Progress ring
// ---------------------------------------------------------------------------

/**
 * A circular gauge for a single share.
 *
 * Used where the number is a *proportion of a whole* rather than a count — a
 * bar would need a labelled 100% end to say the same thing, and at KPI-card
 * size there is no room for one.
 */
export function renderRing(value: number, options: { readonly tone?: string } = {}): SVGSVGElement {
  const size = 44;
  const stroke = 5;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(1, Math.max(0, value));

  const svg = svgEl('svg', {
    viewBox: `0 0 ${size} ${size}`,
    class: 'ring',
    'aria-hidden': 'true',
    focusable: 'false',
  });
  if (options.tone) svg.style.setProperty('--ring-tone', options.tone);

  const shared = { cx: size / 2, cy: size / 2, r: radius, 'stroke-width': stroke, fill: 'none' };
  const arc = svgEl('circle', {
    ...shared,
    class: 'ring-value',
    'stroke-dasharray': `${circumference} ${circumference}`,
    'stroke-linecap': 'round',
    // Start at twelve o'clock rather than three.
    transform: `rotate(-90 ${size / 2} ${size / 2})`,
  });
  const filled = circumference * clamped;
  if (prefersReducedMotion()) {
    arc.setAttribute('stroke-dashoffset', String(circumference - filled));
  } else {
    // Sweep in from empty on the next frame, so the transition has a start
    // value to animate away from.
    arc.setAttribute('stroke-dashoffset', String(circumference));
    requestAnimationFrame(() => {
      arc.style.strokeDashoffset = String(circumference - filled);
    });
  }

  svg.append(svgEl('circle', { ...shared, class: 'ring-track' }), arc);
  return svg;
}

// ---------------------------------------------------------------------------
// Category breakdown
// ---------------------------------------------------------------------------

/** Horizontal bars for finding counts, coloured by severity or category. */
export function renderBreakdown(
  container: HTMLElement,
  items: readonly LabelledCount[],
  options: { readonly colorBy?: 'severity' | 'category' } = {},
): void {
  const populated = items.filter((item) => item.count > 0);
  if (populated.length === 0) {
    container.replaceChildren(emptyNote('Находок пока нет'));
    return;
  }

  const max = Math.max(...populated.map((item) => item.count));
  const total = populated.reduce((sum, item) => sum + item.count, 0);
  const animate = !prefersReducedMotion();

  container.replaceChildren(
    ...populated.map((item, index) => {
      const fill = el('div', {
        class: 'meter-fill',
        style: {
          '--meter-color':
            options.colorBy === 'category'
              ? `var(--category-${item.key}, var(--accent))`
              : severityColor(item.severity),
          // Widths are set after paint so the fill grows from zero; without
          // the deferral the browser has no previous value to transition from.
          'inline-size': animate ? '0%' : `${Math.max(2, (item.count / max) * 100)}%`,
          '--meter-delay': `${index * 45}ms`,
        },
      });
      if (animate) {
        requestAnimationFrame(() => {
          fill.style.setProperty('inline-size', `${Math.max(2, (item.count / max) * 100)}%`);
        });
      }

      return el(
        'div',
        { class: 'bar-row' },
        el('span', { class: 'bar-label', title: item.label }, item.label),
        el('div', { class: 'meter', role: 'presentation' }, fill),
        el('span', { class: 'bar-value' }, formatNumber(item.count)),
        // Share of all findings: the count alone says how many, not how much.
        el('span', { class: 'bar-share' }, `${Math.round((item.count / total) * 100)}%`),
      );
    }),
  );
}

function severityColor(severity: Severity | null): string {
  return severity ? `var(--severity-${severity})` : 'var(--accent)';
}

function emptyNote(message: string): HTMLElement {
  return el('p', { class: 'chart-empty' }, message);
}
