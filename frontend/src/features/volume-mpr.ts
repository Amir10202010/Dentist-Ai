/**
 * The three orthogonal planes: axial, coronal, sagittal.
 *
 * Drawn on the CPU rather than in WebGL, and that is the right call for a
 * reslice. A plane is one strided read out of an array the browser already
 * holds, windowed into an `ImageData` — a few hundred thousand byte operations,
 * well under a frame. Uploading a 3D texture to sample three axis-aligned
 * slices out of it would add a GPU dependency to the part of the viewer that
 * has to work everywhere, for no gain.
 *
 * The pipeline per redraw is deliberately two-stage: the slice is rendered at
 * its **native voxel resolution** onto an offscreen canvas, then blitted to the
 * display canvas through a transform carrying the anisotropic aspect
 * correction, zoom and pan. That gets the browser's own filtering for free, and
 * it keeps overlay geometry — crosshair, finding boxes, measurement handles —
 * in one coordinate space that is easy to reason about.
 *
 * Aspect correction is not cosmetic. A dental CBCT is routinely 0.3 mm in plane
 * and 0.6 mm between slices; drawing a coronal reslice at 1 pixel per voxel
 * would compress it to half its true height and make every measurement taken
 * off it wrong.
 */

import type { Plane } from '../lib/dvol';
import {
  PLANE_LABELS,
  fromPlane,
  hounsfieldAt,
  planeAxis,
  planeExtent,
  readSlice,
  sliceCount,
  toPlane,
} from '../lib/dvol';
import { on } from '../lib/dom';
import type { ViewerStore } from './volume-state';
import { isFindingVisible } from './volume-state';

/** The subset of a finding the renderer needs. */
export interface FindingOverlay {
  readonly id: number;
  readonly classKey: string;
  readonly label: string;
  readonly severity: string;
  readonly box: {
    readonly x: number;
    readonly y: number;
    readonly z: number;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
  };
}

export interface AnnotationOverlay {
  readonly id: number;
  readonly title: string;
  readonly x: number;
  readonly y: number;
  readonly z: number | null;
}

export interface MeasurementOverlay {
  readonly id: number;
  readonly kind: string;
  readonly label: string;
  readonly value: number;
  readonly unit: string;
  readonly points: readonly (readonly number[])[];
}

export interface Overlays {
  findings: readonly FindingOverlay[];
  annotations: readonly AnnotationOverlay[];
  measurements: readonly MeasurementOverlay[];
}

export interface MprPane {
  draw(): void;
  destroy(): void;
}

const MAX_PIXEL_RATIO = 2;
/** Slice thickness, as a fraction of the volume, within which a finding is
 *  considered to intersect the current plane and gets a solid outline. */
const SLICE_TOLERANCE = 0.012;
/** How many rows a label may be pushed down before it is dropped instead. */
const MAX_LABEL_NUDGES = 6;

const SEVERITY_COLOURS: Readonly<Record<string, string>> = {
  critical: '#ff5a5f',
  high: '#ff9f43',
  medium: '#f5c451',
  low: '#4bc0c0',
  info: '#5aa9e6',
};

export function mountMprPane(
  canvas: HTMLCanvasElement,
  store: ViewerStore,
  plane: Plane,
  overlays: Overlays,
  onProbe?: (hu: number, at: readonly [number, number, number]) => void,
): MprPane {
  const canvasContext = canvas.getContext('2d', { alpha: false });
  if (!canvasContext) throw new Error('2D-контекст недоступен.');
  // Bound to a typed const so the null check narrows inside the closures below.
  const context: CanvasRenderingContext2D = canvasContext;

  const volume = store.volume;
  const extent = planeExtent(volume, plane);
  const slices = sliceCount(volume, plane);

  // Scratch buffers, allocated once. A redraw on every slider tick that
  // allocated a 40 000-element array would make the garbage collector part of
  // the interaction.
  const samples = new Uint8Array(extent.cols * extent.rows);
  const offscreen = document.createElement('canvas');
  offscreen.width = extent.cols;
  offscreen.height = extent.rows;
  const rawOffscreenContext = offscreen.getContext('2d', { alpha: false });
  if (!rawOffscreenContext) throw new Error('2D-контекст недоступен.');
  const offscreenContext: CanvasRenderingContext2D = rawOffscreenContext;
  const image = offscreenContext.createImageData(extent.cols, extent.rows);

  let frame = 0;
  let disposed = false;
  /** Display-space placement of the slice, recomputed each draw and reused by
   *  the pointer handlers to convert clicks back into volume coordinates. */
  let placement = { left: 0, top: 0, width: 1, height: 1 };
  /** Label rectangles already drawn this frame, for collision avoidance. */
  const placedLabels: Array<{ left: number; top: number; right: number; bottom: number }> = [];

  function schedule(): void {
    if (frame === 0 && !disposed) frame = requestAnimationFrame(draw);
  }

  function draw(): void {
    frame = 0;
    if (disposed) return;

    const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    const cssWidth = Math.max(1, canvas.clientWidth);
    const cssHeight = Math.max(1, canvas.clientHeight);
    const pixelWidth = Math.round(cssWidth * ratio);
    const pixelHeight = Math.round(cssHeight * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    const state = store.state;
    const axis = planeAxis(plane);
    const index = Math.round((state.position[axis] ?? 0.5) * (slices - 1));

    renderSlice(index, state.windowCenter, state.windowWidth, state.invert);

    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.imageSmoothingEnabled = true;
    context.fillStyle = readVariable(canvas, '--viewer-bg', '#0b0e13');
    context.fillRect(0, 0, cssWidth, cssHeight);

    placement = fit(cssWidth, cssHeight, state.zoom[plane], state.pan[plane]);
    context.drawImage(
      offscreen,
      placement.left,
      placement.top,
      placement.width,
      placement.height,
    );

    drawOverlays(cssWidth, cssHeight);
    drawChrome(cssWidth, cssHeight, index);
  }

  function renderSlice(
    index: number,
    center: number,
    width: number,
    invert: boolean,
  ): void {
    readSlice(volume, plane, index, samples);

    // The window maps a density range onto 0-255. Precomputing the lookup means
    // the per-pixel loop is two array reads, not a multiply and two clamps.
    const lut = windowLut(center, width, invert);
    const data = image.data;
    for (let i = 0, p = 0; i < samples.length; i += 1, p += 4) {
      const value = lut[samples[i] ?? 0] ?? 0;
      data[p] = value;
      data[p + 1] = value;
      data[p + 2] = value;
      data[p + 3] = 255;
    }
    offscreenContext.putImageData(image, 0, 0);
  }

  /**
   * Place the slice inside the pane at its true physical aspect ratio.
   *
   * Fitting by millimetres rather than by voxels is what makes the reslices
   * measurable by eye and keeps a circle in the volume a circle on screen.
   */
  function fit(
    cssWidth: number,
    cssHeight: number,
    zoom: number,
    pan: readonly [number, number],
  ): { left: number; top: number; width: number; height: number } {
    const scale = Math.min(cssWidth / extent.widthMm, cssHeight / extent.heightMm) * zoom;
    const width = extent.widthMm * scale;
    const height = extent.heightMm * scale;
    return {
      left: (cssWidth - width) / 2 + pan[0],
      top: (cssHeight - height) / 2 + pan[1],
      width,
      height,
    };
  }

  function toDisplay(column: number, row: number): readonly [number, number] {
    return [placement.left + column * placement.width, placement.top + row * placement.height];
  }

  function toVolume(clientX: number, clientY: number): [number, number, number] {
    const rect = canvas.getBoundingClientRect();
    const column = (clientX - rect.left - placement.left) / placement.width;
    const row = (clientY - rect.top - placement.top) / placement.height;
    return fromPlane(
      plane,
      Math.min(1, Math.max(0, column)),
      Math.min(1, Math.max(0, row)),
      store.state.position,
    );
  }

  // -- overlays ------------------------------------------------------------
  function drawOverlays(cssWidth: number, cssHeight: number): void {
    const state = store.state;
    const axis = planeAxis(plane);
    const depthPosition = state.position[axis] ?? 0.5;

    // Reset per draw: two findings whose boxes share an edge would otherwise
    // stack their labels on the same pixels and render as one unreadable
    // smear. Cleared here rather than inside `label()` so the whole frame's
    // labels are laid out against each other.
    placedLabels.length = 0;

    if (state.showFindings) {
      for (const finding of overlays.findings) {
        if (!isFindingVisible(state, finding.classKey, finding.id)) continue;
        drawFinding(finding, axis, depthPosition, finding.id === state.selectedFindingId);
      }
    }

    if (state.showMeasurements) {
      for (const measurement of overlays.measurements) {
        drawMeasurement(measurement, depthPosition, axis);
      }
    }
    if (state.pending) drawPending(state.pending.points);

    if (state.showAnnotations) {
      for (const annotation of overlays.annotations) {
        if (annotation.z === null) continue;
        const position: readonly [number, number, number] = [
          annotation.x,
          annotation.y,
          annotation.z,
        ];
        if (Math.abs((position[axis] ?? 0) - depthPosition) > SLICE_TOLERANCE * 3) continue;
        const [column, row] = toPlane(plane, position);
        drawPin(toDisplay(column, row), annotation.title);
      }
    }

    if (state.showCrosshair) drawCrosshair(cssWidth, cssHeight);
  }

  function drawFinding(
    finding: FindingOverlay,
    axis: 0 | 1 | 2,
    depthPosition: number,
    selected: boolean,
  ): void {
    const box = finding.box;
    const start = [box.x, box.y, box.z][axis] ?? 0;
    const size = [box.width, box.height, box.depth][axis] ?? 0;
    // A prism intersects the plane, or it does not. Boxes the current slice
    // passes through are drawn solid; the rest are dashed ghosts, so a
    // clinician can see there is something to scroll to without mistaking it
    // for something visible here.
    const intersects =
      depthPosition >= start - SLICE_TOLERANCE && depthPosition <= start + size + SLICE_TOLERANCE;

    const [column, row] = toPlane(plane, [box.x, box.y, box.z]);
    const [spanColumn, spanRow] = toPlane(plane, [box.width, box.height, box.depth]);
    const [left, top] = toDisplay(column, row);
    const width = spanColumn * placement.width;
    const height = spanRow * placement.height;

    const colour = SEVERITY_COLOURS[finding.severity] ?? SEVERITY_COLOURS['info'] ?? '#5aa9e6';
    context.save();
    context.lineWidth = selected ? 2.5 : 1.5;
    context.strokeStyle = colour;
    context.globalAlpha = intersects ? 1 : 0.28;
    context.setLineDash(intersects ? [] : [4, 4]);
    context.strokeRect(left, top, width, height);

    if (intersects) {
      context.globalAlpha = selected ? 0.18 : 0.08;
      context.fillStyle = colour;
      context.fillRect(left, top, width, height);
      context.globalAlpha = 1;
      label(finding.label, left, top - 4, colour);
    }
    context.restore();
  }

  function drawMeasurement(
    measurement: MeasurementOverlay,
    depthPosition: number,
    axis: 0 | 1 | 2,
  ): void {
    const points = measurement.points.filter((point) => point.length === 3);
    if (points.length < 2) {
      if (points.length === 1) {
        const single = points[0];
        if (single && Math.abs((single[axis] ?? 0) - depthPosition) <= SLICE_TOLERANCE * 3) {
          const [column, row] = toPlane(plane, single as [number, number, number]);
          drawHandle(toDisplay(column, row));
        }
      }
      return;
    }

    const near = points.some(
      (point) => Math.abs((point[axis] ?? 0) - depthPosition) <= SLICE_TOLERANCE * 4,
    );
    context.save();
    context.globalAlpha = near ? 1 : 0.3;
    context.strokeStyle = '#6ee7b7';
    context.lineWidth = 1.6;
    context.beginPath();
    points.forEach((point, index) => {
      const [column, row] = toPlane(plane, point as [number, number, number]);
      const [x, y] = toDisplay(column, row);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    for (const point of points) {
      const [column, row] = toPlane(plane, point as [number, number, number]);
      drawHandle(toDisplay(column, row));
    }
    if (near) {
      const first = points[0];
      if (first) {
        const [column, row] = toPlane(plane, first as [number, number, number]);
        const [x, y] = toDisplay(column, row);
        label(`${measurement.value} ${measurement.unit}`, x, y - 8, '#6ee7b7');
      }
    }
    context.restore();
  }

  function drawPending(points: readonly (readonly [number, number, number])[]): void {
    context.save();
    context.strokeStyle = '#facc15';
    context.setLineDash([5, 4]);
    context.lineWidth = 1.6;
    context.beginPath();
    points.forEach((point, index) => {
      const [column, row] = toPlane(plane, point);
      const [x, y] = toDisplay(column, row);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    for (const point of points) {
      const [column, row] = toPlane(plane, point);
      drawHandle(toDisplay(column, row), '#facc15');
    }
    context.restore();
  }

  function drawHandle(at: readonly [number, number], colour = '#6ee7b7'): void {
    context.save();
    context.fillStyle = colour;
    context.strokeStyle = '#0b0e13';
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(at[0], at[1], 3.5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
  }

  function drawPin(at: readonly [number, number], title: string): void {
    context.save();
    context.fillStyle = '#a78bfa';
    context.beginPath();
    context.arc(at[0], at[1], 4.5, 0, Math.PI * 2);
    context.fill();
    label(title, at[0] + 8, at[1] - 6, '#a78bfa');
    context.restore();
  }

  function drawCrosshair(cssWidth: number, cssHeight: number): void {
    const [column, row] = toPlane(plane, store.state.position);
    const [x, y] = toDisplay(column, row);
    context.save();
    context.strokeStyle = 'rgba(120, 200, 255, 0.55)';
    context.lineWidth = 1;
    context.setLineDash([6, 5]);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(cssWidth, y);
    context.moveTo(x, 0);
    context.lineTo(x, cssHeight);
    context.stroke();
    context.restore();
  }

  function drawChrome(cssWidth: number, cssHeight: number, index: number): void {
    context.save();
    context.font =
      '500 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillStyle = 'rgba(226, 236, 248, 0.72)';
    context.textBaseline = 'top';
    context.fillText(`${PLANE_LABELS[plane]}  ${index + 1}/${slices}`, 8, 8);

    // Scale bar: the reason the aspect correction above has to be right.
    const scale = placement.width / extent.widthMm;
    const targetMm = niceLength(60 / scale);
    const barPixels = targetMm * scale;
    const baseY = cssHeight - 16;
    context.strokeStyle = 'rgba(226, 236, 248, 0.72)';
    context.lineWidth = 1.5;
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(cssWidth - 16 - barPixels, baseY);
    context.lineTo(cssWidth - 16, baseY);
    context.moveTo(cssWidth - 16 - barPixels, baseY - 4);
    context.lineTo(cssWidth - 16 - barPixels, baseY + 4);
    context.moveTo(cssWidth - 16, baseY - 4);
    context.lineTo(cssWidth - 16, baseY + 4);
    context.stroke();
    context.textAlign = 'right';
    context.textBaseline = 'bottom';
    context.fillText(`${targetMm} мм`, cssWidth - 16, baseY - 6);
    context.restore();
  }

  /**
   * Draw a label, nudging it clear of the ones already placed this frame.
   *
   * Overlapping boxes are the normal case, not an edge case: the canal is
   * detected as several segments, and a lesion sits inside the bone whose
   * finding also has a box. Stacking their labels produces one illegible
   * smear, so each is pushed down until it finds room, and dropped if it
   * cannot.
   */
  function label(text: string, x: number, y: number, colour: string): void {
    context.save();
    context.font = '600 11px system-ui, sans-serif';
    context.textBaseline = 'bottom';
    const width = context.measureText(text).width + 6;
    const height = 15;

    let top = y - 13;
    for (let attempt = 0; attempt < MAX_LABEL_NUDGES; attempt += 1) {
      const clashes = placedLabels.some(
        (box) =>
          x - 3 < box.right && x - 3 + width > box.left && top < box.bottom && top + height > box.top,
      );
      if (!clashes) break;
      top += height + 2;
      if (attempt === MAX_LABEL_NUDGES - 1) {
        context.restore();
        return;
      }
    }

    placedLabels.push({ left: x - 3, top, right: x - 3 + width, bottom: top + height });
    context.fillStyle = 'rgba(11, 14, 19, 0.78)';
    context.fillRect(x - 3, top, width, height);
    context.fillStyle = colour;
    context.fillText(text, x, top + height - 3);
    context.restore();
  }

  // -- interaction ---------------------------------------------------------
  const teardown: Array<() => void> = [];
  let dragging: { x: number; y: number; button: number } | null = null;

  teardown.push(
    on(canvas, 'pointerdown', (event) => {
      canvas.setPointerCapture(event.pointerId);
      dragging = { x: event.clientX, y: event.clientY, button: event.button };
      const point = toVolume(event.clientX, event.clientY);
      handlePrimary(point, event);
    }),
  );

  teardown.push(
    on(canvas, 'pointermove', (event) => {
      if (!dragging) return;
      const dx = event.clientX - dragging.x;
      const dy = event.clientY - dragging.y;
      dragging = { ...dragging, x: event.clientX, y: event.clientY };

      const state = store.state;
      // Middle button pans and right button windows regardless of the active
      // tool, matching what every DICOM workstation does — a clinician's hands
      // already know these.
      if (dragging.button === 1 || state.tool === 'pan' || event.shiftKey) {
        const pan = state.pan[plane];
        store.update({
          pan: { ...state.pan, [plane]: [(pan[0] ?? 0) + dx, (pan[1] ?? 0) + dy] },
        });
        return;
      }
      if (dragging.button === 2 || state.tool === 'window') {
        store.update({
          windowCenter: state.windowCenter + dy * 0.6,
          windowWidth: Math.max(2, state.windowWidth + dx * 0.9),
          presetKey: null,
        });
        return;
      }
      if (state.tool === 'cursor') {
        store.update({ position: toVolume(event.clientX, event.clientY) });
      }
    }),
  );

  const release = (event: PointerEvent): void => {
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    dragging = null;
  };
  teardown.push(on(canvas, 'pointerup', release));
  teardown.push(on(canvas, 'pointercancel', release));
  teardown.push(on(canvas, 'contextmenu', (event) => event.preventDefault()));

  teardown.push(
    on(
      canvas,
      'wheel',
      (event) => {
        event.preventDefault();
        const state = store.state;
        if (event.ctrlKey || event.metaKey) {
          const zoom = Math.min(8, Math.max(0.4, (state.zoom[plane] ?? 1) * (event.deltaY > 0 ? 0.9 : 1.1)));
          store.update({ zoom: { ...state.zoom, [plane]: zoom } });
          return;
        }
        // Plain wheel steps through slices, which is the action a clinician
        // takes thousands of times per case.
        const axis = planeAxis(plane);
        const step = (event.deltaY > 0 ? 1 : -1) / (slices - 1);
        const next = [...state.position] as [number, number, number];
        next[axis] = Math.min(1, Math.max(0, (next[axis] ?? 0.5) + step));
        store.update({ position: next });
      },
      { passive: false },
    ),
  );

  function handlePrimary(point: [number, number, number], event: PointerEvent): void {
    if (event.button !== 0) return;
    const state = store.state;

    if (state.tool === 'probe') {
      const hu = hounsfieldAt(volume, point[0], point[1], point[2]);
      store.update({ probe: { hu, at: point }, position: point });
      onProbe?.(hu, point);
      return;
    }

    if (state.tool === 'distance' || state.tool === 'angle') {
      const wanted = state.tool === 'distance' ? 2 : 3;
      const existing = state.pending?.plane === plane ? state.pending.points : [];
      const points = [...existing, point].slice(-wanted);
      store.update({ pending: { kind: state.tool, plane, points } });
      return;
    }

    if (state.tool === 'cursor' || state.tool === 'annotate') {
      store.update({ position: point });
    }
  }

  const unsubscribe = store.subscribe(schedule);
  const observer = new ResizeObserver(schedule);
  observer.observe(canvas);
  const themeObserver = new MutationObserver(schedule);
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });

  schedule();

  return {
    draw: schedule,
    destroy(): void {
      disposed = true;
      if (frame) cancelAnimationFrame(frame);
      unsubscribe();
      observer.disconnect();
      themeObserver.disconnect();
      for (const off of teardown) off();
    },
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const lutCache = { center: Number.NaN, width: Number.NaN, invert: false, table: new Uint8Array(256) };

/** 256-entry window lookup, rebuilt only when the window actually changes. */
function windowLut(center: number, width: number, invert: boolean): Uint8Array {
  if (lutCache.center === center && lutCache.width === width && lutCache.invert === invert) {
    return lutCache.table;
  }
  const low = center - width / 2;
  const table = lutCache.table;
  for (let value = 0; value < 256; value += 1) {
    const scaled = ((value - low) / width) * 255;
    const clamped = scaled < 0 ? 0 : scaled > 255 ? 255 : scaled;
    table[value] = invert ? 255 - clamped : clamped;
  }
  lutCache.center = center;
  lutCache.width = width;
  lutCache.invert = invert;
  return table;
}

/** Round a millimetre length to something a scale bar should be labelled with. */
function niceLength(raw: number): number {
  const steps = [1, 2, 5, 10, 20, 50, 100];
  for (const step of steps) {
    if (raw <= step) return step;
  }
  return 100;
}

function readVariable(element: HTMLElement, name: string, fallback: string): string {
  const value = getComputedStyle(element).getPropertyValue(name).trim();
  return value || fallback;
}
