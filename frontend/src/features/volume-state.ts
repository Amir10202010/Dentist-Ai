/**
 * Shared state for the CBCT viewer.
 *
 * The viewer is four renderers — three MPR panes and a 3D view — that must
 * agree on one thing: where the clinician is looking. Clicking a lesion in the
 * axial pane has to move the coronal and sagittal slices to it, the 3D view's
 * clipping plane has to follow, and a measurement started in one pane has to
 * be visible in the others.
 *
 * A single store with subscribers is what makes that true by construction
 * rather than by remembering to call four update functions. Each renderer
 * subscribes, reads what it needs, and redraws on the next frame; none of them
 * knows the others exist.
 *
 * Kept in its own module so the panes and the 3D view can share the types
 * without importing each other.
 */

import type { Plane, Volume, WindowPreset } from '../lib/dvol';
import { WINDOW_PRESETS, windowToStored } from '../lib/dvol';

/** What a pointer drag does in the 2D panes. */
export type Tool =
  | 'cursor'
  | 'window'
  | 'pan'
  | 'distance'
  | 'angle'
  | 'probe'
  | 'annotate';

/** How the 3D view composites samples along a ray. */
export type RenderMode = 'mip' | 'composite' | 'xray';

export type Layout = 'mpr' | 'volume' | 'split';

export interface PendingMeasurement {
  readonly kind: 'distance' | 'angle';
  readonly plane: Plane;
  readonly points: readonly (readonly [number, number, number])[];
}

export interface ViewerState {
  /** Crosshair, in normalised volume coordinates. The single source of truth
   *  for which slice each pane shows. */
  position: readonly [number, number, number];
  /** Window in stored 0-255 units — what the renderers actually apply. */
  windowCenter: number;
  windowWidth: number;
  /** Which preset is active, or `null` after a manual adjustment. */
  presetKey: string | null;
  invert: boolean;

  layout: Layout;
  renderMode: RenderMode;
  /** Per-plane zoom and pan, so magnifying the axial view leaves the others
   *  framed as they were. */
  zoom: Record<Plane, number>;
  pan: Record<Plane, readonly [number, number]>;

  /** 3D camera. */
  yaw: number;
  pitch: number;
  distance: number;
  /** Fraction of the volume kept along each axis, from the far side. Lets the
   *  clinician cut into the reconstruction rather than only orbit it. */
  clip: readonly [number, number, number];

  tool: Tool;
  /** Class keys currently drawn. Empty means "all", which keeps the common
   *  case free of a set the size of the taxonomy. */
  hiddenClasses: ReadonlySet<string>;
  selectedFindingId: number | null;
  /** When set, only this finding's region is rendered — the "isolate" control. */
  isolatedFindingId: number | null;
  showFindings: boolean;
  showAnnotations: boolean;
  showMeasurements: boolean;
  showCrosshair: boolean;
  /** Measurement being placed, point by point. */
  pending: PendingMeasurement | null;
  /** Last probed Hounsfield value, for the readout. */
  probe: { readonly hu: number; readonly at: readonly [number, number, number] } | null;
}

export interface ViewerStore {
  readonly volume: Volume;
  readonly state: ViewerState;
  /** Merge a patch and notify subscribers on the next frame. */
  update(patch: Partial<ViewerState>): void;
  subscribe(listener: () => void): () => void;
  /** Apply a named window preset, converting from Hounsfield units. */
  applyPreset(key: string): void;
  reset(): void;
}

const DEFAULT_PRESET = 'bone';

export function createViewerStore(volume: Volume): ViewerStore {
  const initial = (): ViewerState => {
    const preset = WINDOW_PRESETS.find((item) => item.key === DEFAULT_PRESET);
    const stored = preset
      ? windowToStored(volume, preset.centerHu, preset.widthHu)
      : { center: volume.windowCenter, width: volume.windowWidth };
    return {
      position: [0.5, 0.5, 0.5],
      windowCenter: stored.center,
      windowWidth: stored.width,
      presetKey: preset ? preset.key : null,
      invert: false,
      layout: 'mpr',
      renderMode: 'composite',
      zoom: { axial: 1, coronal: 1, sagittal: 1 },
      pan: { axial: [0, 0], coronal: [0, 0], sagittal: [0, 0] },
      yaw: 0.6,
      pitch: -0.5,
      distance: 2.4,
      clip: [1, 1, 1],
      tool: 'cursor',
      hiddenClasses: new Set<string>(),
      selectedFindingId: null,
      isolatedFindingId: null,
      showFindings: true,
      showAnnotations: true,
      showMeasurements: true,
      showCrosshair: true,
      pending: null,
      probe: null,
    };
  };

  const state = initial();
  const listeners = new Set<() => void>();
  let frame = 0;

  const notify = (): void => {
    // Coalesced to one frame: a pointer drag can patch the store several times
    // between paints, and each notify redraws four canvases.
    if (frame !== 0) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      for (const listener of listeners) listener();
    });
  };

  return {
    volume,
    state,
    update(patch: Partial<ViewerState>): void {
      Object.assign(state, patch);
      notify();
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    applyPreset(key: string): void {
      const preset: WindowPreset | undefined = WINDOW_PRESETS.find((item) => item.key === key);
      if (!preset) return;
      const stored = windowToStored(volume, preset.centerHu, preset.widthHu);
      Object.assign(state, {
        windowCenter: stored.center,
        windowWidth: stored.width,
        presetKey: preset.key,
      });
      notify();
    },
    reset(): void {
      Object.assign(state, initial());
      notify();
    },
  };
}

/** Whether a finding should be drawn, given the current filters. */
export function isFindingVisible(state: ViewerState, classKey: string, id: number): boolean {
  if (!state.showFindings) return false;
  if (state.isolatedFindingId !== null) return state.isolatedFindingId === id;
  return !state.hiddenClasses.has(classKey);
}

/** Toggle one class in the hidden set, returning a new set. */
export function toggleClass(current: ReadonlySet<string>, classKey: string): Set<string> {
  const next = new Set(current);
  if (next.has(classKey)) next.delete(classKey);
  else next.add(classKey);
  return next;
}
