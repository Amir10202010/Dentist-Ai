/**
 * Interactive radiograph viewer.
 *
 * The image loads once and findings are drawn as SVG on top, so toggling a
 * class costs no network, zoom and pan stay sharp, and selection is two-way
 * with the findings list.
 */

import { all, el, must, on } from '../lib/dom';
import { SEVERITY_ORDER } from '../lib/format';
import type { Finding, Severity, Study } from '../lib/types';

const SVG_NS = 'http://www.w3.org/2000/svg';

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.25;

interface ViewportState {
  scale: number;
  translateX: number;
  translateY: number;
}

export interface ViewerOptions {
  readonly container: HTMLElement;
  readonly study: Study;
  readonly onSelect?: (findingId: number | null) => void;
}

export class StudyViewer {
  private readonly root: HTMLElement;
  private readonly study: Study;
  private readonly stage: HTMLElement;
  private readonly image: HTMLImageElement;
  private readonly svg: SVGSVGElement;
  private readonly shapes = new Map<number, SVGGElement>();
  private readonly onSelect: ((id: number | null) => void) | undefined;

  private view: ViewportState = { scale: 1, translateX: 0, translateY: 0 };
  private hiddenClasses = new Set<string>();
  private minSeverityRank = SEVERITY_ORDER.length - 1;
  private minConfidence = 0;
  private selectedId: number | null = null;
  private panPointerId: number | null = null;
  private panOrigin = { x: 0, y: 0, translateX: 0, translateY: 0 };

  constructor(options: ViewerOptions) {
    this.root = options.container;
    this.study = options.study;
    this.onSelect = options.onSelect;

    this.image = el('img', {
      class: 'viewer-image',
      src: this.study.imageUrl,
      alt: `Рентгеновский снимок ${this.study.originalFilename}`,
      // The image is the primary content; letting it load lazily would delay
      // the thing the user came for.
      loading: 'eager',
      decoding: 'async',
      draggable: false,
    });

    this.svg = document.createElementNS(SVG_NS, 'svg');
    this.svg.setAttribute('class', 'viewer-overlay');
    // A unit viewBox means finding coordinates — already normalised — are used
    // verbatim, with no resolution maths anywhere in the render path.
    this.svg.setAttribute('viewBox', '0 0 1 1');
    this.svg.setAttribute('preserveAspectRatio', 'none');
    this.svg.setAttribute('aria-hidden', 'true');

    this.stage = el('div', { class: 'viewer-stage' }, this.image, this.svg);
    this.root.append(this.stage);

    this.buildShapes();
    this.bindInteractions();
    this.applyTransform();
    this.render();
  }

  // -- public API -------------------------------------------------------
  setClassVisibility(classKey: string, visible: boolean): void {
    if (visible) this.hiddenClasses.delete(classKey);
    else this.hiddenClasses.add(classKey);
    this.render();
  }

  setMinSeverity(severity: Severity): void {
    this.minSeverityRank = SEVERITY_ORDER.indexOf(severity);
    this.render();
  }

  setMinConfidence(value: number): void {
    this.minConfidence = value;
    this.render();
  }

  select(findingId: number | null): void {
    this.selectedId = findingId;
    this.render();
    if (findingId !== null) this.focusOn(findingId);
  }

  zoomBy(delta: number): void {
    this.setScale(this.view.scale + delta);
  }

  reset(): void {
    this.view = { scale: 1, translateX: 0, translateY: 0 };
    this.applyTransform();
  }

  /** Findings currently passing every active filter. */
  visibleFindings(): readonly Finding[] {
    return this.study.findings.filter((finding) => this.isVisible(finding));
  }

  destroy(): void {
    this.stage.remove();
  }

  // -- rendering --------------------------------------------------------
  private buildShapes(): void {
    for (const finding of this.study.findings) {
      const group = document.createElementNS(SVG_NS, 'g');
      group.setAttribute('class', 'finding');
      group.dataset['findingId'] = String(finding.id);
      group.dataset['severity'] = finding.severity;
      group.dataset['category'] = finding.category;

      const rect = document.createElementNS(SVG_NS, 'rect');
      rect.setAttribute('x', String(finding.box.x));
      rect.setAttribute('y', String(finding.box.y));
      rect.setAttribute('width', String(finding.box.width));
      rect.setAttribute('height', String(finding.box.height));
      rect.setAttribute('class', 'finding-box');
      // preserveAspectRatio="none" stretches the unit viewBox, which would
      // also stretch stroke width; this keeps borders visually uniform.
      rect.setAttribute('vector-effect', 'non-scaling-stroke');

      group.append(rect);
      this.svg.append(group);
      this.shapes.set(finding.id, group);
    }
  }

  private isVisible(finding: Finding): boolean {
    if (this.hiddenClasses.has(finding.classKey)) return false;
    if (finding.review === 'rejected') return false;
    if (SEVERITY_ORDER.indexOf(finding.severity) > this.minSeverityRank) return false;
    return finding.confidence >= this.minConfidence;
  }

  private render(): void {
    for (const finding of this.study.findings) {
      const shape = this.shapes.get(finding.id);
      if (!shape) continue;
      const visible = this.isVisible(finding);
      shape.style.display = visible ? '' : 'none';
      shape.classList.toggle('is-selected', visible && finding.id === this.selectedId);
      shape.classList.toggle(
        'is-dimmed',
        visible && this.selectedId !== null && finding.id !== this.selectedId,
      );
    }
  }

  // -- interaction ------------------------------------------------------
  private bindInteractions(): void {
    // Ctrl/⌘ + wheel zooms (matching every image tool); a plain wheel keeps
    // scrolling the page, which is what people expect on a long report.
    on(this.stage, 'wheel', (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      this.setScale(this.view.scale - Math.sign(event.deltaY) * ZOOM_STEP, {
        x: event.offsetX / this.stage.clientWidth,
        y: event.offsetY / this.stage.clientHeight,
      });
    }, { passive: false });

    on(this.stage, 'dblclick', () => {
      this.view.scale > 1 ? this.reset() : this.setScale(2);
    });

    on(this.stage, 'pointerdown', (event) => {
      if (this.view.scale <= 1 || event.button !== 0) return;
      this.panPointerId = event.pointerId;
      this.panOrigin = {
        x: event.clientX,
        y: event.clientY,
        translateX: this.view.translateX,
        translateY: this.view.translateY,
      };
      this.stage.setPointerCapture(event.pointerId);
      this.stage.dataset['panning'] = 'true';
    });

    on(this.stage, 'pointermove', (event) => {
      if (this.panPointerId !== event.pointerId) return;
      this.view.translateX = this.panOrigin.translateX + (event.clientX - this.panOrigin.x);
      this.view.translateY = this.panOrigin.translateY + (event.clientY - this.panOrigin.y);
      this.applyTransform();
    });

    const endPan = (event: PointerEvent): void => {
      if (this.panPointerId !== event.pointerId) return;
      this.panPointerId = null;
      delete this.stage.dataset['panning'];
    };
    on(this.stage, 'pointerup', endPan);
    on(this.stage, 'pointercancel', endPan);

    on(this.svg as unknown as HTMLElement, 'click', (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const group = target.closest('g.finding');
      const raw = group instanceof SVGGElement ? group.dataset['findingId'] : undefined;
      const id = raw === undefined ? null : Number(raw);
      this.selectedId = this.selectedId === id ? null : id;
      this.render();
      this.onSelect?.(this.selectedId);
    });
  }

  private setScale(next: number, anchor?: { x: number; y: number }): void {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number(next.toFixed(2))));
    if (clamped === this.view.scale) return;

    if (anchor && clamped > 1) {
      // Keep the point under the cursor stationary while scaling.
      const ratio = clamped / this.view.scale;
      const rect = this.stage.getBoundingClientRect();
      const anchorX = anchor.x * rect.width;
      const anchorY = anchor.y * rect.height;
      this.view.translateX = anchorX - ratio * (anchorX - this.view.translateX);
      this.view.translateY = anchorY - ratio * (anchorY - this.view.translateY);
    }

    this.view.scale = clamped;
    if (clamped === 1) {
      this.view.translateX = 0;
      this.view.translateY = 0;
    }
    this.applyTransform();
  }

  private applyTransform(): void {
    const { scale, translateX, translateY } = this.view;
    this.stage.style.setProperty('--viewer-scale', String(scale));
    const transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    this.image.style.transform = transform;
    this.svg.style.transform = transform;
    this.stage.dataset['zoomed'] = String(scale > 1);
    this.root.dispatchEvent(
      new CustomEvent<number>('viewer:zoom', { detail: scale, bubbles: true }),
    );
  }

  private focusOn(findingId: number): void {
    const finding = this.study.findings.find((item) => item.id === findingId);
    if (!finding) return;
    const shape = this.shapes.get(findingId);
    shape?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

/** Legend chips that toggle finding classes on and off. */
export function buildLegend(
  container: HTMLElement,
  study: Study,
  viewer: StudyViewer,
): void {
  const seen = new Map<string, { label: string; severity: Severity; count: number }>();
  for (const finding of study.findings) {
    const entry = seen.get(finding.classKey);
    if (entry) entry.count += 1;
    else {
      seen.set(finding.classKey, {
        label: finding.label,
        severity: finding.severity,
        count: 1,
      });
    }
  }

  const chips = [...seen.entries()]
    .sort(
      (a, b) =>
        SEVERITY_ORDER.indexOf(a[1].severity) - SEVERITY_ORDER.indexOf(b[1].severity),
    )
    .map(([key, meta]) => {
      const chip = el(
        'button',
        {
          class: 'legend-chip',
          type: 'button',
          dataset: { classKey: key, severity: meta.severity, active: 'true' },
          // aria-pressed carries the state to assistive tech; the styling
          // reads from the same attribute so they cannot diverge.
          aria: { pressed: 'true' },
        },
        el('span', { class: 'dot' }),
        el('span', { class: 'legend-label' }, meta.label),
        el('span', { class: 'legend-count' }, String(meta.count)),
      );

      on(chip, 'click', () => {
        const nowActive = chip.dataset['active'] !== 'true';
        chip.dataset['active'] = String(nowActive);
        chip.setAttribute('aria-pressed', String(nowActive));
        viewer.setClassVisibility(key, nowActive);
      });

      return chip;
    });

  container.replaceChildren(...chips);
}

/** Wire the zoom toolbar buttons. */
export function bindViewerControls(scope: ParentNode, viewer: StudyViewer): void {
  for (const button of all<HTMLButtonElement>('[data-viewer-action]', scope)) {
    on(button, 'click', () => {
      switch (button.dataset['viewerAction']) {
        case 'zoom-in':
          viewer.zoomBy(ZOOM_STEP * 2);
          break;
        case 'zoom-out':
          viewer.zoomBy(-ZOOM_STEP * 2);
          break;
        case 'reset':
          viewer.reset();
          break;
        default:
          break;
      }
    });
  }
}

export function mountViewer(study: Study, onSelect?: (id: number | null) => void): StudyViewer {
  const container = must('[data-viewer]');
  container.replaceChildren();
  return new StudyViewer(
    onSelect ? { container, study, onSelect } : { container, study },
  );
}
