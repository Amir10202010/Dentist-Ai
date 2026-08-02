/**
 * The CBCT workstation: three reslices, one volume rendering, one state.
 *
 * This module owns no rendering of its own. It mounts the four renderers,
 * points them all at a single {@link ViewerStore}, and binds the toolbar to
 * that store. Everything the user does is a patch to one object, and the
 * renderers redraw because they subscribed — which is why clicking a lesion in
 * the axial pane moves the other two slices, swings the 3D clipping plane and
 * updates the readout without any of those four things knowing about each
 * other.
 *
 * The 3D view degrades rather than fails. On a machine without WebGL2 the
 * volume pane shows why and the three MPR panes — the diagnostically important
 * half — carry on working.
 */

import type { Plane } from '../lib/dvol';
import { PLANES, WINDOW_PRESETS, planeAxis, sliceCount } from '../lib/dvol';
import type { Volume } from '../lib/dvol';
import { all, maybe, must, on } from '../lib/dom';
import type { MeasurementOverlay, MprPane, Overlays } from './volume-mpr';
import { mountMprPane } from './volume-mpr';
import type { VolumeRenderer } from './volume-3d';
import { VolumeRendererError, mountVolumeRenderer } from './volume-3d';
import type { Layout, RenderMode, Tool, ViewerStore } from './volume-state';
import { createViewerStore, toggleClass } from './volume-state';

export interface ViewerCallbacks {
  /** A completed distance or angle, ready to be persisted. */
  onMeasurement?(
    kind: 'distance' | 'angle',
    plane: Plane,
    points: readonly (readonly [number, number, number])[],
  ): void;
  /** The annotate tool was used at a position. */
  onAnnotate?(position: readonly [number, number, number]): void;
  onFindingSelected?(id: number | null): void;
}

export interface Viewer {
  readonly store: ViewerStore;
  /** Move the crosshair to a finding's centre and select it. */
  focusFinding(id: number, centre: readonly [number, number, number]): void;
  /** Replace the overlay data after a save, without remounting. */
  setOverlays(next: Partial<Overlays>): void;
  /** A PNG data URL of one pane, for a report screenshot. */
  snapshot(plane: Plane | 'volume'): string | null;
  clearPending(): void;
  destroy(): void;
}

export function mountViewer(
  root: HTMLElement,
  volume: Volume,
  overlays: Overlays,
  callbacks: ViewerCallbacks = {},
): Viewer {
  const store = createViewerStore(volume);
  const live: Overlays = {
    findings: overlays.findings,
    annotations: overlays.annotations,
    measurements: overlays.measurements,
  };

  const panes = new Map<Plane, MprPane>();
  const canvases = new Map<Plane | 'volume', HTMLCanvasElement>();
  let renderer: VolumeRenderer | null = null;

  for (const plane of PLANES) {
    const canvas = maybe<HTMLCanvasElement>(`[data-pane="${plane}"] canvas`, root);
    if (!canvas) continue;
    canvases.set(plane, canvas);
    panes.set(plane, mountMprPane(canvas, store, plane, live, reportProbe));
  }

  const volumeCanvas = maybe<HTMLCanvasElement>('[data-pane="volume"] canvas', root);
  if (volumeCanvas) {
    canvases.set('volume', volumeCanvas);
    try {
      renderer = mountVolumeRenderer(volumeCanvas, store, live.findings);
    } catch (error) {
      const notice = maybe('[data-volume-error]', root);
      if (notice) {
        notice.hidden = false;
        notice.textContent =
          error instanceof VolumeRendererError
            ? error.message
            : 'Объёмный рендеринг недоступен на этом устройстве.';
      }
      volumeCanvas.hidden = true;
      // Without a 3D pane the split layout is just the MPR layout with a gap.
      store.update({ layout: 'mpr' });
    }
  }

  // -- toolbar -------------------------------------------------------------
  const teardown: Array<() => void> = [];

  function bindGroup<T extends string>(
    attribute: string,
    apply: (value: T) => void,
  ): void {
    for (const button of all<HTMLButtonElement>(`[${attribute}]`, root)) {
      teardown.push(
        on(button, 'click', () => {
          const value = button.getAttribute(attribute);
          if (value) apply(value as T);
        }),
      );
    }
  }

  bindGroup<Layout>('data-layout', (layout) => store.update({ layout }));
  bindGroup<RenderMode>('data-render-mode', (renderMode) => store.update({ renderMode }));
  bindGroup<Tool>('data-tool', (tool) => store.update({ tool, pending: null }));
  bindGroup<string>('data-preset', (key) => store.applyPreset(key));

  for (const plane of PLANES) {
    const slider = maybe<HTMLInputElement>(`[data-slice="${plane}"]`, root);
    if (!slider) continue;
    const slices = sliceCount(volume, plane);
    slider.min = '0';
    slider.max = String(slices - 1);
    slider.step = '1';
    teardown.push(
      on(slider, 'input', () => {
        const axis = planeAxis(plane);
        const next = [...store.state.position] as [number, number, number];
        next[axis] = slices > 1 ? Number(slider.value) / (slices - 1) : 0.5;
        store.update({ position: next });
      }),
    );
  }

  bindRange('[data-window-center]', (value) =>
    store.update({ windowCenter: value, presetKey: null }),
  );
  bindRange('[data-window-width]', (value) =>
    store.update({ windowWidth: Math.max(2, value), presetKey: null }),
  );
  for (const [index, selector] of ['[data-clip-x]', '[data-clip-y]', '[data-clip-z]'].entries()) {
    bindRange(selector, (value) => {
      const clip = [...store.state.clip] as [number, number, number];
      clip[index] = Math.max(0.05, value / 100);
      store.update({ clip });
    });
  }

  bindToggle('[data-invert]', (on_) => store.update({ invert: on_ }));
  bindToggle('[data-crosshair]', (on_) => store.update({ showCrosshair: on_ }));
  bindToggle('[data-show-findings]', (on_) => store.update({ showFindings: on_ }));
  bindToggle('[data-show-annotations]', (on_) => store.update({ showAnnotations: on_ }));
  bindToggle('[data-show-measurements]', (on_) => store.update({ showMeasurements: on_ }));

  const resetButton = maybe<HTMLButtonElement>('[data-reset-view]', root);
  if (resetButton) teardown.push(on(resetButton, 'click', () => store.reset()));

  function bindRange(selector: string, apply: (value: number) => void): void {
    const input = maybe<HTMLInputElement>(selector, root);
    if (!input) return;
    teardown.push(on(input, 'input', () => apply(Number(input.value))));
  }

  function bindToggle(selector: string, apply: (checked: boolean) => void): void {
    const input = maybe<HTMLInputElement>(selector, root);
    if (!input) return;
    teardown.push(on(input, 'change', () => apply(input.checked)));
  }

  // -- keyboard ------------------------------------------------------------
  const TOOL_KEYS: Readonly<Record<string, Tool>> = {
    v: 'cursor',
    w: 'window',
    h: 'pan',
    d: 'distance',
    a: 'angle',
    p: 'probe',
    n: 'annotate',
  };
  const LAYOUT_KEYS: Readonly<Record<string, Layout>> = { '1': 'mpr', '2': 'volume', '3': 'split' };

  teardown.push(
    on(document.documentElement, 'keydown', (event) => {
      // Never steal a keystroke from a field the clinician is typing in.
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const key = event.key.toLowerCase();
      const tool = TOOL_KEYS[key];
      if (tool) {
        store.update({ tool, pending: null });
        event.preventDefault();
        return;
      }
      const layout = LAYOUT_KEYS[event.key];
      if (layout) {
        store.update({ layout });
        event.preventDefault();
        return;
      }
      if (key === 'i') {
        store.update({ invert: !store.state.invert });
        event.preventDefault();
        return;
      }
      if (key === 'escape' && store.state.pending) {
        store.update({ pending: null });
        event.preventDefault();
      }
    }),
  );

  // -- reactive chrome -----------------------------------------------------
  const unsubscribe = store.subscribe(syncChrome);

  function syncChrome(): void {
    const state = store.state;
    root.dataset['layout'] = state.layout;
    root.dataset['tool'] = state.tool;

    setPressed('data-layout', state.layout);
    setPressed('data-render-mode', state.renderMode);
    setPressed('data-tool', state.tool);
    setPressed('data-preset', state.presetKey ?? '');

    for (const plane of PLANES) {
      const slider = maybe<HTMLInputElement>(`[data-slice="${plane}"]`, root);
      if (!slider) continue;
      const slices = sliceCount(volume, plane);
      const index = Math.round((state.position[planeAxis(plane)] ?? 0.5) * (slices - 1));
      if (slider.value !== String(index)) slider.value = String(index);
      const readout = maybe(`[data-slice-readout="${plane}"]`, root);
      if (readout) readout.textContent = `${index + 1} / ${slices}`;
    }

    setValue('[data-window-center]', state.windowCenter);
    setValue('[data-window-width]', state.windowWidth);
    const windowReadout = maybe('[data-window-readout]', root);
    if (windowReadout) {
      const centerHu = Math.round(state.windowCenter * volume.huSlope + volume.huIntercept);
      const widthHu = Math.round(state.windowWidth * volume.huSlope);
      windowReadout.textContent = `C ${centerHu} / W ${widthHu} HU`;
    }

    const positionReadout = maybe('[data-position-readout]', root);
    if (positionReadout) {
      const [x, y, z] = state.position;
      positionReadout.textContent = `${(x * volume.physicalSize[0]).toFixed(1)} · ${(
        y * volume.physicalSize[1]
      ).toFixed(1)} · ${(z * volume.physicalSize[2]).toFixed(1)} мм`;
    }

    // Announce a finished measurement exactly once, then hand it upward.
    const pending = state.pending;
    if (pending) {
      const wanted = pending.kind === 'distance' ? 2 : 3;
      if (pending.points.length === wanted) {
        store.update({ pending: null });
        callbacks.onMeasurement?.(pending.kind, pending.plane, pending.points);
      }
    }
  }

  function setPressed(attribute: string, active: string): void {
    for (const button of all<HTMLButtonElement>(`[${attribute}]`, root)) {
      const value = button.getAttribute(attribute);
      button.setAttribute('aria-pressed', String(value === active));
    }
  }

  function setValue(selector: string, value: number): void {
    const input = maybe<HTMLInputElement>(selector, root);
    if (!input) return;
    const rounded = String(Math.round(value));
    if (input.value !== rounded) input.value = rounded;
  }

  function reportProbe(hu: number, at: readonly [number, number, number]): void {
    const readout = maybe('[data-probe-readout]', root);
    if (readout) readout.textContent = `${Math.round(hu)} HU`;
    if (store.state.tool === 'annotate') callbacks.onAnnotate?.(at);
  }

  // The annotate tool needs a click even when the probe is not active, so it is
  // wired on the panes' own container rather than through the probe callback.
  for (const [plane, canvas] of canvases) {
    if (plane === 'volume') continue;
    teardown.push(
      on(canvas, 'dblclick', () => {
        if (store.state.tool === 'annotate') callbacks.onAnnotate?.(store.state.position);
      }),
    );
  }

  syncChrome();

  return {
    store,
    focusFinding(id: number, centre: readonly [number, number, number]): void {
      store.update({
        selectedFindingId: id,
        position: [centre[0], centre[1], centre[2]],
      });
      callbacks.onFindingSelected?.(id);
    },
    setOverlays(next: Partial<Overlays>): void {
      if (next.findings) live.findings = next.findings;
      if (next.annotations) live.annotations = next.annotations;
      if (next.measurements) live.measurements = next.measurements;
      for (const pane of panes.values()) pane.draw();
      renderer?.draw();
    },
    snapshot(plane: Plane | 'volume'): string | null {
      const canvas = canvases.get(plane);
      if (!canvas) return null;
      // The 3D canvas is drawn without a preserved buffer, so it has to be
      // redrawn in the same frame the pixels are read back.
      if (plane === 'volume') renderer?.draw();
      try {
        return canvas.toDataURL('image/png');
      } catch {
        return null;
      }
    },
    clearPending(): void {
      store.update({ pending: null });
    },
    destroy(): void {
      unsubscribe();
      for (const off of teardown) off();
      for (const pane of panes.values()) pane.destroy();
      renderer?.destroy();
    },
  };
}

/** Class-visibility chips, rendered by the page and wired here. */
export function bindClassFilters(root: HTMLElement, store: ViewerStore): () => void {
  const offs: Array<() => void> = [];
  for (const chip of all<HTMLButtonElement>('[data-class-toggle]', root)) {
    offs.push(
      on(chip, 'click', () => {
        const key = chip.dataset['classToggle'];
        if (!key) return;
        const hidden = toggleClass(store.state.hiddenClasses, key);
        store.update({ hiddenClasses: hidden });
        chip.dataset['hidden'] = String(hidden.has(key));
      }),
    );
  }
  return () => {
    for (const off of offs) off();
  };
}

/** The window presets, for a page that wants to render its own buttons. */
export const PRESETS = WINDOW_PRESETS;

/** Convert a stored measurement into the overlay shape the panes expect. */
export function toMeasurementOverlay(measurement: {
  readonly id: number;
  readonly kind: string;
  readonly label: string;
  readonly value: number;
  readonly unit: string;
  readonly points: readonly (readonly number[])[];
}): MeasurementOverlay {
  return measurement;
}

/** Element lookup that throws — a missing viewer hook is a template bug. */
export function requireViewerRoot(): HTMLElement {
  return must('[data-viewer]');
}
