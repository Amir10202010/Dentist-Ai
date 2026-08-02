/**
 * Reader for the canonical `DVOL` container.
 *
 * The server normalises every CBCT — DICOM series, multi-frame DICOM, NIfTI —
 * into a 64-byte header followed by 8-bit voxels in slice-major order, for the
 * same reason it normalises meshes to binary STL: this is then the only volume
 * parser the client needs, and it is thirty lines.
 *
 * Everything downstream works in **normalised volume coordinates** — `x`, `y`
 * and `z` each in `[0, 1]` — rather than in voxels. Findings, measurements and
 * annotations are all stored that way on the server, so a scan re-ingested at
 * a different decimation factor keeps every annotation attached to the anatomy
 * instead of to a grid index.
 */

const MAGIC = 'DVOL0001';
const HEADER_BYTES = 64;

export interface VolumeGeometry {
  /** Voxel counts along x, y and z. */
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  /** Millimetres per voxel, `[x, y, z]`. */
  readonly spacing: readonly [number, number, number];
  /** `hounsfield = stored * huSlope + huIntercept`. */
  readonly huSlope: number;
  readonly huIntercept: number;
  /** Default window, in stored 0-255 units. */
  readonly windowCenter: number;
  readonly windowWidth: number;
}

export interface Volume extends VolumeGeometry {
  /** `[z][y][x]`, flattened. */
  readonly voxels: Uint8Array;
  /** Field of view in millimetres, `[x, y, z]`. */
  readonly physicalSize: readonly [number, number, number];
}

export class VolumeParseError extends Error {}

export function parseVolume(buffer: ArrayBuffer): Volume {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new VolumeParseError('Файл тома повреждён.');
  }

  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC) {
    throw new VolumeParseError('Файл не является каноническим томом.');
  }

  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const depth = view.getUint32(16, true);
  const spacing: [number, number, number] = [
    view.getFloat32(20, true),
    view.getFloat32(24, true),
    view.getFloat32(28, true),
  ];
  const huSlope = view.getFloat32(32, true);
  const huIntercept = view.getFloat32(36, true);
  const windowCenter = view.getFloat32(40, true);
  const windowWidth = view.getFloat32(44, true);

  const expected = width * height * depth;
  if (expected === 0 || buffer.byteLength - HEADER_BYTES < expected) {
    throw new VolumeParseError('Данные тома обрываются на середине.');
  }

  return {
    width,
    height,
    depth,
    spacing,
    huSlope,
    huIntercept,
    windowCenter,
    windowWidth,
    voxels: new Uint8Array(buffer, HEADER_BYTES, expected),
    physicalSize: [width * spacing[0], height * spacing[1], depth * spacing[2]],
  };
}

/** Stored sample at a normalised position, or 0 outside the volume. */
export function sampleAt(volume: Volume, x: number, y: number, z: number): number {
  const ix = Math.min(volume.width - 1, Math.max(0, Math.round(x * volume.width)));
  const iy = Math.min(volume.height - 1, Math.max(0, Math.round(y * volume.height)));
  const iz = Math.min(volume.depth - 1, Math.max(0, Math.round(z * volume.depth)));
  return volume.voxels[iz * volume.height * volume.width + iy * volume.width + ix] ?? 0;
}

/** Hounsfield value at a normalised position. */
export function hounsfieldAt(volume: Volume, x: number, y: number, z: number): number {
  return sampleAt(volume, x, y, z) * volume.huSlope + volume.huIntercept;
}

/** Millimetre distance between two normalised points. */
export function distanceMm(
  volume: Volume,
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  const dx = (a[0] - b[0]) * volume.physicalSize[0];
  const dy = (a[1] - b[1]) * volume.physicalSize[1];
  const dz = (a[2] - b[2]) * volume.physicalSize[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Angle at `vertex`, in degrees, between the rays to `a` and `b`. */
export function angleDegrees(
  volume: Volume,
  a: readonly [number, number, number],
  vertex: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  const toMm = (
    p: readonly [number, number, number],
    q: readonly [number, number, number],
  ): [number, number, number] => [
    (p[0] - q[0]) * volume.physicalSize[0],
    (p[1] - q[1]) * volume.physicalSize[1],
    (p[2] - q[2]) * volume.physicalSize[2],
  ];

  const armA = toMm(a, vertex);
  const armB = toMm(b, vertex);
  const lengthA = Math.hypot(...armA);
  const lengthB = Math.hypot(...armB);
  if (lengthA < 1e-6 || lengthB < 1e-6) return 0;

  const dot = armA[0] * armB[0] + armA[1] * armB[1] + armA[2] * armB[2];
  return (Math.acos(Math.max(-1, Math.min(1, dot / (lengthA * lengthB)))) * 180) / Math.PI;
}

/**
 * The three orthogonal planes, in the order clinicians read them.
 *
 * `axial` steps along z, `coronal` along y, `sagittal` along x — matching the
 * DICOM patient frame the server sorts every series into.
 */
export type Plane = 'axial' | 'coronal' | 'sagittal';

export const PLANES: readonly Plane[] = ['axial', 'coronal', 'sagittal'] as const;

export const PLANE_LABELS: Readonly<Record<Plane, string>> = {
  axial: 'Аксиальная',
  coronal: 'Коронарная',
  sagittal: 'Сагиттальная',
};

/** Number of slices available in a plane. */
export function sliceCount(volume: VolumeGeometry, plane: Plane): number {
  if (plane === 'axial') return volume.depth;
  if (plane === 'coronal') return volume.height;
  return volume.width;
}

/** In-plane pixel dimensions and the millimetre size they span. */
export function planeExtent(
  volume: Volume,
  plane: Plane,
): { readonly cols: number; readonly rows: number; readonly widthMm: number; readonly heightMm: number } {
  if (plane === 'axial') {
    return {
      cols: volume.width,
      rows: volume.height,
      widthMm: volume.physicalSize[0],
      heightMm: volume.physicalSize[1],
    };
  }
  if (plane === 'coronal') {
    return {
      cols: volume.width,
      rows: volume.depth,
      widthMm: volume.physicalSize[0],
      heightMm: volume.physicalSize[2],
    };
  }
  return {
    cols: volume.height,
    rows: volume.depth,
    widthMm: volume.physicalSize[1],
    heightMm: volume.physicalSize[2],
  };
}

/**
 * Extract one slice into `out` as raw stored samples.
 *
 * Written as three explicit loops rather than one generic gather because the
 * axial case is a contiguous copy and the other two are strided reads; a
 * single indexed path would make the common case as slow as the worst one, and
 * this runs on every slider tick.
 */
export function readSlice(volume: Volume, plane: Plane, index: number, out: Uint8Array): void {
  const { width, height, depth, voxels } = volume;
  const sliceStride = width * height;

  if (plane === 'axial') {
    const z = Math.min(depth - 1, Math.max(0, index));
    out.set(voxels.subarray(z * sliceStride, z * sliceStride + sliceStride));
    return;
  }

  if (plane === 'coronal') {
    const y = Math.min(height - 1, Math.max(0, index));
    for (let z = 0; z < depth; z += 1) {
      const from = z * sliceStride + y * width;
      out.set(voxels.subarray(from, from + width), z * width);
    }
    return;
  }

  const x = Math.min(width - 1, Math.max(0, index));
  for (let z = 0; z < depth; z += 1) {
    const rowBase = z * height;
    const sliceBase = z * sliceStride + x;
    for (let y = 0; y < height; y += 1) {
      out[rowBase + y] = voxels[sliceBase + y * width] ?? 0;
    }
  }
}

/**
 * Map a normalised volume position to `(column, row)` within a plane, and back.
 *
 * These two are the contract between the 2D panes and everything stored on the
 * server: a click becomes a normalised point through `fromPlane`, and a stored
 * finding becomes a screen position through `toPlane`.
 */
export function toPlane(
  plane: Plane,
  position: readonly [number, number, number],
): readonly [number, number] {
  const [x, y, z] = position;
  if (plane === 'axial') return [x, y];
  if (plane === 'coronal') return [x, z];
  return [y, z];
}

export function fromPlane(
  plane: Plane,
  column: number,
  row: number,
  current: readonly [number, number, number],
): [number, number, number] {
  const [x, y, z] = current;
  if (plane === 'axial') return [column, row, z];
  if (plane === 'coronal') return [column, y, row];
  return [x, column, row];
}

/** Which axis a plane's slider moves along, as an index into a position. */
export function planeAxis(plane: Plane): 0 | 1 | 2 {
  if (plane === 'axial') return 2;
  if (plane === 'coronal') return 1;
  return 0;
}

/**
 * Window presets, in Hounsfield units.
 *
 * A window is not a brightness control — it is the range of tissue densities
 * mapped onto the display's 256 levels, and reading bone and soft tissue needs
 * different ranges. These are the standard dental settings; the caller
 * converts them into stored units with the volume's own HU mapping.
 */
export interface WindowPreset {
  readonly key: string;
  readonly label: string;
  readonly centerHu: number;
  readonly widthHu: number;
}

export const WINDOW_PRESETS: readonly WindowPreset[] = [
  { key: 'bone', label: 'Кость', centerHu: 700, widthHu: 2800 },
  { key: 'teeth', label: 'Зубы', centerHu: 1400, widthHu: 3600 },
  { key: 'soft', label: 'Мягкие ткани', centerHu: 60, widthHu: 400 },
  { key: 'sinus', label: 'Пазухи', centerHu: -300, widthHu: 1600 },
  { key: 'wide', label: 'Весь диапазон', centerHu: 500, widthHu: 4000 },
] as const;

/** Convert a Hounsfield window into the stored 0-255 units the renderer uses. */
export function windowToStored(
  volume: VolumeGeometry,
  centerHu: number,
  widthHu: number,
): { readonly center: number; readonly width: number } {
  const slope = volume.huSlope || 1;
  return {
    center: (centerHu - volume.huIntercept) / slope,
    width: Math.max(widthHu / slope, 1),
  };
}
