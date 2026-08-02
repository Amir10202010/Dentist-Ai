/**
 * WebGL viewer for a 3D scan.
 *
 * The server normalises every upload to binary STL, so this is the only mesh
 * parser the client needs and it is thirty lines. Everything else here is a
 * minimal renderer — one program, one buffer, flat facet shading — which is
 * why the app ships no 3D library: a scan is opaque triangles under a fixed
 * light, and that is the whole requirement.
 */

import { on } from '../lib/dom';

const STL_HEADER_BYTES = 84;
const STL_RECORD_BYTES = 50;
const FLOATS_PER_VERTEX = 6;
const VERTICES_PER_TRIANGLE = 3;

const MIN_DISTANCE_FACTOR = 0.35;
const MAX_DISTANCE_FACTOR = 6;
const MAX_PIXEL_RATIO = 2;
/** Straight up would gimbal-lock the orbit; stop just short of the poles. */
const MAX_PITCH = Math.PI / 2 - 0.05;

const VERTEX_SHADER = `
attribute vec3 aPosition;
attribute vec3 aNormal;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;

varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vClipCoord;

void main() {
  vec4 viewPosition = uModelView * vec4(aPosition, 1.0);
  vNormal = uNormalMatrix * aNormal;
  vViewPosition = viewPosition.xyz;
  vClipCoord = aPosition.z;
  gl_Position = uProjection * viewPosition;
}
`;

const FRAGMENT_SHADER = `
precision highp float;

varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vClipCoord;

uniform float uClipPlane;
uniform vec3 uBaseColor;
uniform vec3 uCutColor;

void main() {
  if (vClipCoord > uClipPlane) discard;

  // The cut exposes the inside of the shell, so backfaces have to be lit too.
  vec3 normal = normalize(vNormal);
  if (!gl_FrontFacing) normal = -normal;

  vec3 key = normalize(vec3(0.35, 0.55, 0.75));
  vec3 fill = normalize(vec3(-0.6, -0.25, 0.35));
  vec3 viewDirection = normalize(-vViewPosition);

  float keyLight = max(dot(normal, key), 0.0);
  float fillLight = max(dot(normal, fill), 0.0) * 0.30;
  float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 2.5) * 0.22;

  vec3 albedo = gl_FrontFacing ? uBaseColor : uCutColor;
  vec3 color = albedo * (0.30 + keyLight * 0.72 + fillLight) + rim;
  gl_FragColor = vec4(color, 1.0);
}
`;

export interface MeshStats {
  readonly triangles: number;
  readonly bytes: number;
}

export interface MeshViewer {
  readonly stats: MeshStats;
  resetView(): void;
  /** 0 hides everything above the model's lowest point, 1 shows all of it. */
  setClip(ratio: number): void;
  destroy(): void;
}

export class MeshViewerError extends Error {}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------
interface ParsedMesh {
  readonly interleaved: Float32Array;
  readonly triangles: number;
  readonly min: [number, number, number];
  readonly max: [number, number, number];
}

function parseBinaryStl(buffer: ArrayBuffer): ParsedMesh {
  if (buffer.byteLength < STL_HEADER_BYTES) {
    throw new MeshViewerError('Файл модели повреждён.');
  }
  const view = new DataView(buffer);
  const triangles = view.getUint32(80, true);
  if (buffer.byteLength < STL_HEADER_BYTES + triangles * STL_RECORD_BYTES) {
    throw new MeshViewerError('Файл модели обрывается на середине.');
  }

  const interleaved = new Float32Array(triangles * VERTICES_PER_TRIANGLE * FLOATS_PER_VERTEX);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  let write = 0;
  for (let index = 0; index < triangles; index += 1) {
    const base = STL_HEADER_BYTES + index * STL_RECORD_BYTES;
    const nx = view.getFloat32(base, true);
    const ny = view.getFloat32(base + 4, true);
    const nz = view.getFloat32(base + 8, true);

    for (let corner = 0; corner < VERTICES_PER_TRIANGLE; corner += 1) {
      const offset = base + 12 + corner * 12;
      const x = view.getFloat32(offset, true);
      const y = view.getFloat32(offset + 4, true);
      const z = view.getFloat32(offset + 8, true);

      interleaved[write] = x;
      interleaved[write + 1] = y;
      interleaved[write + 2] = z;
      interleaved[write + 3] = nx;
      interleaved[write + 4] = ny;
      interleaved[write + 5] = nz;
      write += FLOATS_PER_VERTEX;

      if (x < min[0]) min[0] = x;
      if (y < min[1]) min[1] = y;
      if (z < min[2]) min[2] = z;
      if (x > max[0]) max[0] = x;
      if (y > max[1]) max[1] = y;
      if (z > max[2]) max[2] = z;
    }
  }

  if (triangles === 0) throw new MeshViewerError('В модели нет треугольников.');
  return { interleaved, triangles, min, max };
}

// ---------------------------------------------------------------------------
// Matrices — column-major, the order WebGL expects
// ---------------------------------------------------------------------------
type Mat4 = Float32Array;

function identity(): Mat4 {
  const out = new Float32Array(16);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  return out;
}

function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += (a[k * 4 + row] ?? 0) * (b[column * 4 + k] ?? 0);
      }
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

function translation(x: number, y: number, z: number): Mat4 {
  const out = identity();
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

function rotationX(angle: number): Mat4 {
  const out = identity();
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  out[5] = c;
  out[6] = s;
  out[9] = -s;
  out[10] = c;
  return out;
}

function rotationZ(angle: number): Mat4 {
  const out = identity();
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  out[0] = c;
  out[1] = s;
  out[4] = -s;
  out[5] = c;
  return out;
}

/** Upper-left 3×3. Valid as a normal matrix because we only rotate and translate. */
function normalMatrix(modelView: Mat4): Float32Array {
  return new Float32Array([
    modelView[0] ?? 0,
    modelView[1] ?? 0,
    modelView[2] ?? 0,
    modelView[4] ?? 0,
    modelView[5] ?? 0,
    modelView[6] ?? 0,
    modelView[8] ?? 0,
    modelView[9] ?? 0,
    modelView[10] ?? 0,
  ]);
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------
function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new MeshViewerError('Не удалось создать шейдер.');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? '';
    gl.deleteShader(shader);
    throw new MeshViewerError(`Шейдер не скомпилировался: ${log}`);
  }
  return shader;
}

function buildProgram(gl: WebGLRenderingContext): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new MeshViewerError('Не удалось создать программу WebGL.');
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new MeshViewerError(gl.getProgramInfoLog(program) ?? 'Программа WebGL не собралась.');
  }
  return program;
}

function readColor(canvas: HTMLElement, variable: string, fallback: [number, number, number]) {
  const raw = getComputedStyle(canvas).getPropertyValue(variable).trim();
  const match = /^#?([0-9a-f]{6})$/i.exec(raw);
  if (!match?.[1]) return fallback;
  const value = Number.parseInt(match[1], 16);
  return [
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
  ] as [number, number, number];
}

export async function mountMeshViewer(
  canvas: HTMLCanvasElement,
  meshUrl: string,
  signal?: AbortSignal,
): Promise<MeshViewer> {
  const response = await fetch(meshUrl, { credentials: 'same-origin', ...(signal ? { signal } : {}) });
  if (!response.ok) throw new MeshViewerError('Не удалось загрузить модель.');
  const buffer = await response.arrayBuffer();
  const mesh = parseBinaryStl(buffer);

  const context =
    canvas.getContext('webgl2', { antialias: true, alpha: false }) ??
    canvas.getContext('webgl', { antialias: true, alpha: false });
  if (!context) throw new MeshViewerError('Браузер не поддерживает WebGL.');
  // The shaders are GLSL ES 1.00, which both context versions accept, so the
  // narrower interface is enough for everything below.
  const gl: WebGLRenderingContext = context;

  const program = buildProgram(gl);
  const buffer3d = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer3d);
  gl.bufferData(gl.ARRAY_BUFFER, mesh.interleaved, gl.STATIC_DRAW);

  const positionLocation = gl.getAttribLocation(program, 'aPosition');
  const normalLocation = gl.getAttribLocation(program, 'aNormal');
  const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(normalLocation);
  gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, stride, 12);

  const uniforms = {
    modelView: gl.getUniformLocation(program, 'uModelView'),
    projection: gl.getUniformLocation(program, 'uProjection'),
    normal: gl.getUniformLocation(program, 'uNormalMatrix'),
    clip: gl.getUniformLocation(program, 'uClipPlane'),
    base: gl.getUniformLocation(program, 'uBaseColor'),
    cut: gl.getUniformLocation(program, 'uCutColor'),
  };

  const centre: [number, number, number] = [
    (mesh.min[0] + mesh.max[0]) / 2,
    (mesh.min[1] + mesh.max[1]) / 2,
    (mesh.min[2] + mesh.max[2]) / 2,
  ];
  const radius =
    Math.max(
      mesh.max[0] - mesh.min[0],
      mesh.max[1] - mesh.min[1],
      mesh.max[2] - mesh.min[2],
    ) / 2 || 1;

  const HOME = { yaw: 0, pitch: -0.9, distance: radius * 3.1 };
  let yaw = HOME.yaw;
  let pitch = HOME.pitch;
  let distance = HOME.distance;
  let panX = 0;
  let panY = 0;
  let clipPlane = mesh.max[2];

  let frame = 0;
  let disposed = false;

  function draw(): void {
    frame = 0;
    if (disposed) return;

    const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    gl.viewport(0, 0, width, height);
    gl.enable(gl.DEPTH_TEST);
    // Both sides are drawn: the clipped-open shell would otherwise be hollow.
    gl.disable(gl.CULL_FACE);
    const background = readColor(canvas, '--mesh-bg', [0.06, 0.08, 0.11]);
    gl.clearColor(background[0], background[1], background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(program);

    const modelView = multiply(
      multiply(
        translation(panX, panY, -distance),
        multiply(rotationX(pitch), rotationZ(yaw)),
      ),
      translation(-centre[0], -centre[1], -centre[2]),
    );
    const projection = perspective(
      (38 * Math.PI) / 180,
      width / height,
      Math.max(radius * 0.02, 0.05),
      distance + radius * 8,
    );

    gl.uniformMatrix4fv(uniforms.modelView, false, modelView);
    gl.uniformMatrix4fv(uniforms.projection, false, projection);
    gl.uniformMatrix3fv(uniforms.normal, false, normalMatrix(modelView));
    gl.uniform1f(uniforms.clip, clipPlane);
    gl.uniform3fv(uniforms.base, readColor(canvas, '--mesh-base', [0.9, 0.89, 0.86]));
    gl.uniform3fv(uniforms.cut, readColor(canvas, '--mesh-cut', [0.85, 0.5, 0.45]));

    gl.drawArrays(gl.TRIANGLES, 0, mesh.triangles * VERTICES_PER_TRIANGLE);
  }

  function schedule(): void {
    if (frame === 0 && !disposed) frame = requestAnimationFrame(draw);
  }

  // -- interaction ----------------------------------------------------------
  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDistance = 0;

  const teardown: Array<() => void> = [];

  teardown.push(
    on(canvas, 'pointerdown', (event) => {
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }),
  );

  teardown.push(
    on(canvas, 'pointermove', (event) => {
      const previous = pointers.get(event.pointerId);
      if (!previous) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

      if (pointers.size >= 2) {
        const [a, b] = [...pointers.values()];
        if (!a || !b) return;
        const spread = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDistance > 0) {
          distance = clampDistance(distance * (pinchDistance / spread));
        }
        pinchDistance = spread;
        schedule();
        return;
      }

      const dx = event.clientX - previous.x;
      const dy = event.clientY - previous.y;

      if (event.shiftKey || event.buttons === 4) {
        const scale = (distance * 0.0022);
        panX += dx * scale;
        panY -= dy * scale;
      } else {
        yaw += dx * 0.008;
        pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch + dy * 0.008));
      }
      schedule();
    }),
  );

  const release = (event: PointerEvent): void => {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinchDistance = 0;
  };
  teardown.push(on(canvas, 'pointerup', release));
  teardown.push(on(canvas, 'pointercancel', release));

  teardown.push(
    on(
      canvas,
      'wheel',
      (event) => {
        event.preventDefault();
        distance = clampDistance(distance * (event.deltaY > 0 ? 1.12 : 1 / 1.12));
        schedule();
      },
      { passive: false },
    ),
  );

  teardown.push(
    on(canvas, 'keydown', (event) => {
      const step = 0.12;
      switch (event.key) {
        case 'ArrowLeft':
          yaw -= step;
          break;
        case 'ArrowRight':
          yaw += step;
          break;
        case 'ArrowUp':
          pitch = Math.max(-MAX_PITCH, pitch - step);
          break;
        case 'ArrowDown':
          pitch = Math.min(MAX_PITCH, pitch + step);
          break;
        default:
          return;
      }
      event.preventDefault();
      schedule();
    }),
  );

  function clampDistance(value: number): number {
    return Math.max(radius * MIN_DISTANCE_FACTOR, Math.min(radius * MAX_DISTANCE_FACTOR, value));
  }

  const observer = new ResizeObserver(schedule);
  observer.observe(canvas);

  const themeObserver = new MutationObserver(schedule);
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });

  schedule();

  return {
    stats: { triangles: mesh.triangles, bytes: buffer.byteLength },
    resetView(): void {
      yaw = HOME.yaw;
      pitch = HOME.pitch;
      distance = HOME.distance;
      panX = 0;
      panY = 0;
      schedule();
    },
    setClip(ratio: number): void {
      clipPlane = mesh.min[2] + (mesh.max[2] - mesh.min[2]) * ratio;
      schedule();
    },
    destroy(): void {
      disposed = true;
      if (frame) cancelAnimationFrame(frame);
      observer.disconnect();
      themeObserver.disconnect();
      for (const off of teardown) off();
      gl.deleteBuffer(buffer3d);
      gl.deleteProgram(program);
    },
  };
}
