/**
 * Ray-marched volume rendering of a CBCT, in WebGL2.
 *
 * The reconstruction is uploaded once as a single-channel 3D texture and
 * rendered by marching a ray per pixel through it. That is the whole technique,
 * and it is the reason this needs WebGL2: `sampler3D` and `R8` textures do not
 * exist in WebGL1, and the alternatives — slicing the volume into a texture
 * atlas and compositing hundreds of quads — cost more code and look worse.
 * Where WebGL2 is unavailable the caller keeps the three MPR panes, which are
 * the diagnostically important half.
 *
 * Three projections, because they answer different questions:
 *
 * - **composite** builds a shaded surface with a transfer function, which is
 *   what makes bone read as bone and is the view a patient understands.
 * - **mip** keeps the brightest sample along each ray. Metal, root fillings and
 *   cortical outlines pop out; it is the fastest way to find a restoration.
 * - **xray** averages along the ray, which is a synthetic radiograph — useful
 *   for comparing a CBCT against the panoramic images already on the chart.
 *
 * The composite mode shades from the volume's own gradient, computed with
 * central differences at sample time. Six extra texture fetches per opaque
 * sample is the difference between a recognisable jaw and a fog bank, and they
 * are skipped where the sample contributes nothing.
 */

import { on } from '../lib/dom';
import { sampleAt } from '../lib/dvol';
import type { RenderMode, ViewerStore } from './volume-state';
import type { FindingOverlay } from './volume-mpr';
import { isFindingVisible } from './volume-state';

const MAX_PIXEL_RATIO = 1.75;
/** Mode codes the fragment shader switches on. */
const RENDER_MODE_CODES: Readonly<Record<RenderMode, number>> = {
  mip: 0,
  composite: 1,
  xray: 2,
};
/** Samples per ray. Enough to avoid banding on a 256³ volume without making a
 *  mid-range integrated GPU miss frames while orbiting. */
const STEPS_STILL = 384;
/** Reduced while the pointer is down: a responsive drag matters more than a
 *  clean image nobody is looking at yet. */
const STEPS_INTERACTIVE = 160;

const VERTEX_SHADER = `#version 300 es
in vec2 aPosition;
out vec2 vNdc;

void main() {
  vNdc = aPosition;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vNdc;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform mat3 uOrientation;
uniform vec3 uCamera;
uniform vec3 uHalfExtent;
uniform vec3 uClip;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uWindowLow;
uniform float uWindowWidth;
uniform float uSteps;
uniform int uMode;
uniform vec3 uBackground;

/** Slab method. Returns false when the ray misses the box entirely. */
bool intersectBox(vec3 origin, vec3 direction, vec3 halfSize, out float near, out float far) {
  // Named inv, not inverse: the latter is a builtin in GLSL ES 3.0.
  vec3 inv = 1.0 / direction;
  vec3 t0 = (-halfSize - origin) * inv;
  vec3 t1 = (halfSize - origin) * inv;
  vec3 low = min(t0, t1);
  vec3 high = max(t0, t1);
  near = max(max(low.x, low.y), low.z);
  far = min(min(high.x, high.y), high.z);
  return far > max(near, 0.0);
}

/** World position to texture coordinate, honouring the clip fractions. */
vec3 toTexture(vec3 position) {
  return (position / uHalfExtent) * 0.5 + 0.5;
}

float windowed(float raw) {
  return clamp((raw - uWindowLow) / uWindowWidth, 0.0, 1.0);
}

/**
 * Transfer function: density to colour.
 *
 * Three stops chosen to match how the tissues actually read — soft tissue a
 * desaturated red-brown, trabecular and cortical bone a warm cream, enamel and
 * metal near white. A greyscale ramp is technically honest and clinically
 * useless here, because the whole point of the 3D view is separating tissues at
 * a glance.
 */
vec3 tissueColour(float density) {
  vec3 soft = vec3(0.72, 0.44, 0.38);
  vec3 bone = vec3(0.95, 0.90, 0.79);
  vec3 enamel = vec3(1.0, 0.99, 0.96);
  if (density < 0.5) return mix(soft, bone, density / 0.5);
  return mix(bone, enamel, (density - 0.5) / 0.5);
}

/** Central-difference gradient, used as a surface normal for shading. */
vec3 gradientAt(vec3 texel, vec3 step) {
  float dx = texture(uVolume, texel + vec3(step.x, 0.0, 0.0)).r
           - texture(uVolume, texel - vec3(step.x, 0.0, 0.0)).r;
  float dy = texture(uVolume, texel + vec3(0.0, step.y, 0.0)).r
           - texture(uVolume, texel - vec3(0.0, step.y, 0.0)).r;
  float dz = texture(uVolume, texel + vec3(0.0, 0.0, step.z)).r
           - texture(uVolume, texel - vec3(0.0, 0.0, step.z)).r;
  return vec3(dx, dy, dz);
}

void main() {
  vec3 direction = normalize(uOrientation * vec3(vNdc.x * uAspect * uTanHalfFov, vNdc.y * uTanHalfFov, -1.0));

  // The clip fractions shrink the box from the far side of each axis, which is
  // what lets a clinician cut into the reconstruction instead of only orbiting
  // the outside of it.
  vec3 halfSize = uHalfExtent * uClip;
  vec3 centreShift = uHalfExtent * (uClip - 1.0);
  vec3 origin = uCamera - centreShift;

  float near;
  float far;
  if (!intersectBox(origin, direction, halfSize, near, far)) {
    fragColor = vec4(uBackground, 1.0);
    return;
  }
  near = max(near, 0.0);

  float span = far - near;
  float stepLength = span / uSteps;
  vec3 texelStep = 1.0 / vec3(textureSize(uVolume, 0));

  vec3 accumulated = vec3(0.0);
  float alpha = 0.0;
  float brightest = 0.0;
  float total = 0.0;
  int counted = 0;

  for (float i = 0.0; i < uSteps; i += 1.0) {
    vec3 position = origin + direction * (near + stepLength * (i + 0.5));
    vec3 texel = toTexture(position + centreShift);
    if (any(lessThan(texel, vec3(0.0))) || any(greaterThan(texel, vec3(1.0)))) continue;

    float density = windowed(texture(uVolume, texel).r);

    if (uMode == 0) {
      brightest = max(brightest, density);
      continue;
    }
    if (uMode == 2) {
      total += density;
      counted += 1;
      continue;
    }

    // Composite. The exponent is the whole difference between a recognisable
    // jaw and a featureless blob: soft tissue and bone are only a factor of
    // three apart in density, so a gentle curve accumulates the skin surface
    // to full opacity long before the ray reaches the skeleton. Raised to 4.5,
    // soft tissue contributes about a thirtieth of what cortical bone does per
    // step and the mineralised structures show through it.
    float opacity = pow(density, 4.5) * 0.62;
    opacity = 1.0 - pow(1.0 - opacity, stepLength * 220.0);
    if (opacity < 0.002) continue;

    vec3 colour = tissueColour(density);
    vec3 gradient = gradientAt(texel, texelStep);
    float magnitude = length(gradient);
    if (magnitude > 0.0001) {
      vec3 normal = -gradient / magnitude;
      vec3 key = normalize(uOrientation * vec3(0.35, 0.55, 0.75));
      float diffuse = max(dot(normal, key), 0.0);
      float rim = pow(1.0 - max(dot(normal, -direction), 0.0), 2.0) * 0.18;
      colour *= 0.42 + 0.68 * diffuse;
      colour += rim;
    }

    accumulated += (1.0 - alpha) * opacity * colour;
    alpha += (1.0 - alpha) * opacity;
    if (alpha > 0.985) break;
  }

  if (uMode == 0) {
    vec3 colour = tissueColour(brightest) * brightest;
    fragColor = vec4(mix(uBackground, colour, min(brightest * 1.35, 1.0)), 1.0);
    return;
  }
  if (uMode == 2) {
    float mean = counted > 0 ? total / float(counted) : 0.0;
    // Boosted, because averaging through mostly-air collapses the range.
    float exposed = clamp(pow(mean * 3.4, 0.85), 0.0, 1.0);
    fragColor = vec4(mix(uBackground, vec3(exposed), 1.0), 1.0);
    return;
  }

  fragColor = vec4(accumulated + uBackground * (1.0 - alpha), 1.0);
}
`;

const LINE_VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
uniform mat4 uViewProjection;
void main() {
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
}
`;

const LINE_FRAGMENT_SHADER = `#version 300 es
precision highp float;
out vec4 fragColor;
uniform vec4 uColour;
void main() { fragColor = uColour; }
`;

export interface VolumeRenderer {
  draw(): void;
  /** Ray-cast from a screen point to the first mineralised sample, so clicking
   *  a structure in 3D moves the MPR crosshair to it. */
  pick(clientX: number, clientY: number): readonly [number, number, number] | null;
  destroy(): void;
}

export class VolumeRendererError extends Error {}

export function mountVolumeRenderer(
  canvas: HTMLCanvasElement,
  store: ViewerStore,
  findings: readonly FindingOverlay[],
): VolumeRenderer {
  const context = canvas.getContext('webgl2', { alpha: false, antialias: false });
  if (!context) {
    throw new VolumeRendererError(
      'Объёмный рендеринг требует WebGL2. Двумерные проекции доступны.',
    );
  }
  // Bound to its own const so the null check narrows for every closure below;
  // narrowing on the `getContext` result itself does not survive into them.
  const gl: WebGL2RenderingContext = context;

  const volume = store.volume;
  const program = buildProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  const lineProgram = buildProgram(gl, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);

  // -- volume texture ------------------------------------------------------
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, texture);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(
    gl.TEXTURE_3D,
    0,
    gl.R8,
    volume.width,
    volume.height,
    volume.depth,
    0,
    gl.RED,
    gl.UNSIGNED_BYTE,
    volume.voxels,
  );
  const uploadError = gl.getError();
  if (uploadError !== gl.NO_ERROR) {
    throw new VolumeRendererError(
      'Не удалось загрузить том в память видеокарты. Попробуйте двумерный режим.',
    );
  }

  // -- geometry ------------------------------------------------------------
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const quadLocation = gl.getAttribLocation(program, 'aPosition');

  const lineBuffer = gl.createBuffer();
  const lineLocation = gl.getAttribLocation(lineProgram, 'aPosition');

  const uniforms = {
    volume: gl.getUniformLocation(program, 'uVolume'),
    orientation: gl.getUniformLocation(program, 'uOrientation'),
    camera: gl.getUniformLocation(program, 'uCamera'),
    halfExtent: gl.getUniformLocation(program, 'uHalfExtent'),
    clip: gl.getUniformLocation(program, 'uClip'),
    tanHalfFov: gl.getUniformLocation(program, 'uTanHalfFov'),
    aspect: gl.getUniformLocation(program, 'uAspect'),
    windowLow: gl.getUniformLocation(program, 'uWindowLow'),
    windowWidth: gl.getUniformLocation(program, 'uWindowWidth'),
    steps: gl.getUniformLocation(program, 'uSteps'),
    mode: gl.getUniformLocation(program, 'uMode'),
    background: gl.getUniformLocation(program, 'uBackground'),
  };
  const lineUniforms = {
    viewProjection: gl.getUniformLocation(lineProgram, 'uViewProjection'),
    colour: gl.getUniformLocation(lineProgram, 'uColour'),
  };

  // The volume is normalised so its longest physical axis is 1, which keeps a
  // 0.3 × 0.3 × 0.6 mm acquisition from rendering as a squashed slab.
  const longest = Math.max(...volume.physicalSize);
  const halfExtent: [number, number, number] = [
    volume.physicalSize[0] / longest / 2,
    volume.physicalSize[1] / longest / 2,
    volume.physicalSize[2] / longest / 2,
  ];
  const fovY = (36 * Math.PI) / 180;

  let frame = 0;
  let disposed = false;
  let interacting = false;

  function schedule(): void {
    if (frame === 0 && !disposed) frame = requestAnimationFrame(draw);
  }

  function camera(): { position: [number, number, number]; orientation: Float32Array } {
    const { yaw, pitch, distance } = store.state;
    const cosPitch = Math.cos(pitch);
    const forward: [number, number, number] = [
      Math.sin(yaw) * cosPitch,
      Math.sin(pitch),
      Math.cos(yaw) * cosPitch,
    ];
    const position: [number, number, number] = [
      forward[0] * distance,
      forward[1] * distance,
      forward[2] * distance,
    ];

    // Look-at basis, written out rather than composed from rotations so the
    // same matrix can be reused by the CPU picker below.
    const zAxis = normalise(position);
    const worldUp: [number, number, number] = Math.abs(zAxis[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
    const xAxis = normalise(cross(worldUp, zAxis));
    const yAxis = cross(zAxis, xAxis);

    return {
      position,
      // Column-major 3×3: camera space to world space.
      orientation: new Float32Array([
        xAxis[0], xAxis[1], xAxis[2],
        yAxis[0], yAxis[1], yAxis[2],
        zAxis[0], zAxis[1], zAxis[2],
      ]),
    };
  }

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

    const state = store.state;
    const background = readColour(canvas, '--viewer-bg', [0.043, 0.055, 0.075]);
    gl.viewport(0, 0, width, height);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(background[0] ?? 0, background[1] ?? 0, background[2] ?? 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const view = camera();
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(quadLocation);
    gl.vertexAttribPointer(quadLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, texture);
    gl.uniform1i(uniforms.volume, 0);
    gl.uniformMatrix3fv(uniforms.orientation, false, view.orientation);
    gl.uniform3fv(uniforms.camera, view.position);
    gl.uniform3fv(uniforms.halfExtent, halfExtent);
    gl.uniform3fv(uniforms.clip, clipVector(state.clip));
    gl.uniform1f(uniforms.tanHalfFov, Math.tan(fovY / 2));
    gl.uniform1f(uniforms.aspect, width / height);
    // The shader samples in 0-1, so the stored window converts by /255.
    gl.uniform1f(uniforms.windowLow, (state.windowCenter - state.windowWidth / 2) / 255);
    gl.uniform1f(uniforms.windowWidth, Math.max(state.windowWidth / 255, 1 / 255));
    gl.uniform1f(uniforms.steps, interacting ? STEPS_INTERACTIVE : STEPS_STILL);
    gl.uniform1i(uniforms.mode, RENDER_MODE_CODES[state.renderMode] ?? 1);
    gl.uniform3fv(uniforms.background, background);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (state.showFindings) drawFindingBoxes(view, width, height);
  }

  function drawFindingBoxes(
    view: { position: [number, number, number]; orientation: Float32Array },
    width: number,
    height: number,
  ): void {
    const state = store.state;
    const visible = findings.filter((finding) =>
      isFindingVisible(state, finding.classKey, finding.id),
    );
    if (visible.length === 0) return;

    const viewProjection = viewProjectionMatrix(view, width / height, fovY);
    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(lineUniforms.viewProjection, false, viewProjection);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    for (const finding of visible) {
      const selected = finding.id === state.selectedFindingId;
      const colour = severityColour(finding.severity);
      gl.uniform4f(lineUniforms.colour, colour[0], colour[1], colour[2], selected ? 0.95 : 0.5);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, boxEdges(finding, halfExtent), gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(lineLocation);
      gl.vertexAttribPointer(lineLocation, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, 24);
    }
    gl.disable(gl.BLEND);
  }

  // -- interaction ---------------------------------------------------------
  const teardown: Array<() => void> = [];
  const pointers = new Map<number, { x: number; y: number }>();
  let pinch = 0;
  let moved = false;

  teardown.push(
    on(canvas, 'pointerdown', (event) => {
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      interacting = true;
      moved = false;
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
        if (pinch > 0) {
          store.update({ distance: clampDistance(store.state.distance * (pinch / spread)) });
        }
        pinch = spread;
        return;
      }

      const dx = event.clientX - previous.x;
      const dy = event.clientY - previous.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;

      const state = store.state;
      if (event.shiftKey || event.buttons === 4) {
        store.update({
          windowCenter: state.windowCenter + dy * 0.6,
          windowWidth: Math.max(2, state.windowWidth + dx * 0.9),
          presetKey: null,
        });
        return;
      }
      store.update({
        yaw: state.yaw + dx * 0.008,
        // Stop just short of the poles: straight down would gimbal-lock the
        // look-at basis.
        pitch: Math.max(-1.52, Math.min(1.52, state.pitch + dy * 0.008)),
      });
    }),
  );

  const release = (event: PointerEvent): void => {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinch = 0;
    if (pointers.size === 0) {
      interacting = false;
      // A click that did not orbit is a click *at* something: jump the MPR
      // crosshair to the first mineralised sample under the pointer.
      if (!moved && event.button === 0) {
        const hit = pick(event.clientX, event.clientY);
        if (hit) store.update({ position: hit });
      }
      schedule();
    }
  };
  teardown.push(on(canvas, 'pointerup', release));
  teardown.push(on(canvas, 'pointercancel', release));

  teardown.push(
    on(
      canvas,
      'wheel',
      (event) => {
        event.preventDefault();
        store.update({
          distance: clampDistance(store.state.distance * (event.deltaY > 0 ? 1.1 : 1 / 1.1)),
        });
      },
      { passive: false },
    ),
  );

  teardown.push(
    on(canvas, 'keydown', (event) => {
      const step = 0.12;
      const state = store.state;
      switch (event.key) {
        case 'ArrowLeft':
          store.update({ yaw: state.yaw - step });
          break;
        case 'ArrowRight':
          store.update({ yaw: state.yaw + step });
          break;
        case 'ArrowUp':
          store.update({ pitch: Math.max(-1.52, state.pitch - step) });
          break;
        case 'ArrowDown':
          store.update({ pitch: Math.min(1.52, state.pitch + step) });
          break;
        default:
          return;
      }
      event.preventDefault();
    }),
  );

  /**
   * CPU ray march to the first sample above the window's midpoint.
   *
   * Mirrors the shader's camera and box maths so a click lands where the pixel
   * it hit was drawn. Two hundred steps through an array already in memory is
   * well under a frame, and it avoids a readback or a picking framebuffer.
   */
  function pick(clientX: number, clientY: number): readonly [number, number, number] | null {
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;

    const view = camera();
    const tanHalfFov = Math.tan(fovY / 2);
    const aspect = rect.width / rect.height;
    const local: [number, number, number] = [ndcX * aspect * tanHalfFov, ndcY * tanHalfFov, -1];
    const orientation = view.orientation;
    const direction = normalise([
      orientation[0]! * local[0] + orientation[3]! * local[1] + orientation[6]! * local[2],
      orientation[1]! * local[0] + orientation[4]! * local[1] + orientation[7]! * local[2],
      orientation[2]! * local[0] + orientation[5]! * local[1] + orientation[8]! * local[2],
    ]);

    const clip = clipVector(store.state.clip);
    const clipX = clip[0] ?? 1;
    const clipY = clip[1] ?? 1;
    const clipZ = clip[2] ?? 1;
    const half: [number, number, number] = [
      halfExtent[0] * clipX,
      halfExtent[1] * clipY,
      halfExtent[2] * clipZ,
    ];
    const shift: [number, number, number] = [
      halfExtent[0] * (clipX - 1),
      halfExtent[1] * (clipY - 1),
      halfExtent[2] * (clipZ - 1),
    ];
    const origin: [number, number, number] = [
      view.position[0] - shift[0],
      view.position[1] - shift[1],
      view.position[2] - shift[2],
    ];

    let near = -Infinity;
    let far = Infinity;
    for (let axis = 0; axis < 3; axis += 1) {
      const inverse = 1 / (direction[axis] ?? 1e-6);
      const t0 = (-(half[axis] ?? 0) - (origin[axis] ?? 0)) * inverse;
      const t1 = ((half[axis] ?? 0) - (origin[axis] ?? 0)) * inverse;
      near = Math.max(near, Math.min(t0, t1));
      far = Math.min(far, Math.max(t0, t1));
    }
    if (far <= Math.max(near, 0)) return null;
    near = Math.max(near, 0);

    const state = store.state;
    const threshold = state.windowCenter;
    const steps = 220;
    const stepLength = (far - near) / steps;
    for (let i = 0; i < steps; i += 1) {
      const t = near + stepLength * (i + 0.5);
      const px = (origin[0] ?? 0) + direction[0]! * t + shift[0];
      const py = (origin[1] ?? 0) + direction[1]! * t + shift[1];
      const pz = (origin[2] ?? 0) + direction[2]! * t + shift[2];
      const u = px / halfExtent[0] / 2 + 0.5;
      const v = py / halfExtent[1] / 2 + 0.5;
      const w = pz / halfExtent[2] / 2 + 0.5;
      if (u < 0 || u > 1 || v < 0 || v > 1 || w < 0 || w > 1) continue;
      if (sampleAt(volume, u, v, w) >= threshold) return [u, v, w];
    }
    return null;
  }

  function clampDistance(value: number): number {
    return Math.max(0.9, Math.min(6, value));
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
    pick,
    destroy(): void {
      disposed = true;
      if (frame) cancelAnimationFrame(frame);
      unsubscribe();
      observer.disconnect();
      themeObserver.disconnect();
      for (const off of teardown) off();
      gl.deleteTexture(texture);
      gl.deleteBuffer(quad);
      gl.deleteBuffer(lineBuffer);
      gl.deleteProgram(program);
      gl.deleteProgram(lineProgram);
    },
  };
}

// ---------------------------------------------------------------------------
// GL helpers
// ---------------------------------------------------------------------------
function buildProgram(gl: WebGL2RenderingContext, vertex: string, fragment: string): WebGLProgram {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertex));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new VolumeRendererError(gl.getProgramInfoLog(program) ?? 'Программа WebGL не собралась.');
  }
  return program;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new VolumeRendererError('Не удалось создать шейдер.');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? '';
    gl.deleteShader(shader);
    throw new VolumeRendererError(`Шейдер не скомпилировался: ${log}`);
  }
  return shader;
}

/** Clip fractions, floored so the volume never collapses to nothing. */
function clipVector(clip: readonly [number, number, number]): Float32Array {
  return new Float32Array([
    Math.max(0.05, clip[0]),
    Math.max(0.05, clip[1]),
    Math.max(0.05, clip[2]),
  ]);
}

/** Wireframe edges of a finding's box, in the renderer's world space. */
function boxEdges(
  finding: FindingOverlay,
  halfExtent: readonly [number, number, number],
): Float32Array {
  const box = finding.box;
  const toWorld = (u: number, v: number, w: number): [number, number, number] => [
    (u - 0.5) * 2 * halfExtent[0],
    (v - 0.5) * 2 * halfExtent[1],
    (w - 0.5) * 2 * halfExtent[2],
  ];

  const x0 = box.x;
  const x1 = box.x + box.width;
  const y0 = box.y;
  const y1 = box.y + box.height;
  const z0 = box.z;
  const z1 = box.z + box.depth;

  const corners = [
    toWorld(x0, y0, z0),
    toWorld(x1, y0, z0),
    toWorld(x1, y1, z0),
    toWorld(x0, y1, z0),
    toWorld(x0, y0, z1),
    toWorld(x1, y0, z1),
    toWorld(x1, y1, z1),
    toWorld(x0, y1, z1),
  ];
  const edges = [
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7,
  ];

  const out = new Float32Array(edges.length * 3);
  edges.forEach((index, slot) => {
    const corner = corners[index] ?? [0, 0, 0];
    out[slot * 3] = corner[0];
    out[slot * 3 + 1] = corner[1];
    out[slot * 3 + 2] = corner[2];
  });
  return out;
}

function viewProjectionMatrix(
  view: { position: [number, number, number]; orientation: Float32Array },
  aspect: number,
  fovY: number,
): Float32Array {
  const o = view.orientation;
  // Inverse of a rotation is its transpose; the translation follows from it.
  const right: [number, number, number] = [o[0] ?? 0, o[1] ?? 0, o[2] ?? 0];
  const up: [number, number, number] = [o[3] ?? 0, o[4] ?? 0, o[5] ?? 0];
  const forward: [number, number, number] = [o[6] ?? 0, o[7] ?? 0, o[8] ?? 0];
  const eye = view.position;

  const viewMatrix = new Float32Array([
    right[0], up[0], forward[0], 0,
    right[1], up[1], forward[1], 0,
    right[2], up[2], forward[2], 0,
    -dot(right, eye), -dot(up, eye), -dot(forward, eye), 1,
  ]);

  const f = 1 / Math.tan(fovY / 2);
  const near = 0.05;
  const far = 20;
  const projection = new Float32Array(16);
  projection[0] = f / aspect;
  projection[5] = f;
  projection[10] = (far + near) / (near - far);
  projection[11] = -1;
  projection[14] = (2 * far * near) / (near - far);

  return multiply4(projection, viewMatrix);
}

function multiply4(a: Float32Array, b: Float32Array): Float32Array {
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

function normalise(v: readonly [number, number, number]): [number, number, number] {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function cross(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function severityColour(severity: string): [number, number, number] {
  const table: Record<string, [number, number, number]> = {
    critical: [1, 0.35, 0.37],
    high: [1, 0.62, 0.26],
    medium: [0.96, 0.77, 0.32],
    low: [0.29, 0.75, 0.75],
    info: [0.35, 0.66, 0.9],
  };
  return table[severity] ?? table['info'] ?? [0.35, 0.66, 0.9];
}

function readColour(
  element: HTMLElement,
  name: string,
  fallback: [number, number, number],
): Float32Array {
  const raw = getComputedStyle(element).getPropertyValue(name).trim();
  const match = /^#?([0-9a-f]{6})$/i.exec(raw);
  if (!match?.[1]) return new Float32Array(fallback);
  const value = Number.parseInt(match[1], 16);
  return new Float32Array([
    ((value >> 16) & 0xff) / 255,
    ((value >> 8) & 0xff) / 255,
    (value & 0xff) / 255,
  ]);
}
