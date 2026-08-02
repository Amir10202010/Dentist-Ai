/**
 * Typed API client.
 *
 * One place that knows about CSRF headers, error shapes and request
 * cancellation. Callers get a typed value or a typed `ApiError`, never a bare
 * `Response` to remember to check.
 */

import type {
  AnnotationKind,
  ApiErrorCode,
  AskResult,
  AssistantThread,
  Dashboard,
  Finding,
  FindingReview,
  Page,
  Patient,
  PatientOverview,
  PatientSummary,
  PlanItem,
  PlanItemStatus,
  PlanStatus,
  ProblemDocument,
  ProcedureOption,
  Scan,
  ScanArch,
  ScanKind,
  Study,
  StudyListItem,
  StudyReport,
  TreatmentPlan,
  User,
  Measurement,
  MeasurementKind,
  PipelineDescriptor,
  ViewPlane,
  Volume,
  VolumeAnnotation,
  VolumeFieldOfView,
  VolumeFinding,
  VolumeListItem,
  TreatmentApproach,
} from './types';

const CSRF_HEADER = 'X-CSRF-Token';
const CSRF_COOKIE = 'dentist_ai_csrf';

export class ApiError extends Error {
  readonly status: number;
  readonly code: ApiErrorCode;
  readonly fieldErrors: Readonly<Record<string, string>>;

  constructor(problem: ProblemDocument) {
    super(problem.title);
    this.name = 'ApiError';
    this.status = problem.status;
    this.code = problem.code;
    this.fieldErrors = problem.errors ?? {};
  }

  /** Whether retrying the identical request could plausibly succeed. */
  get isTransient(): boolean {
    return this.status >= 500 || this.code === 'rate_limited';
  }
}

/** Thrown when the network itself failed — offline, DNS, connection reset. */
export class NetworkError extends Error {
  constructor(cause: unknown) {
    super('Не удалось связаться с сервером. Проверьте подключение.');
    this.name = 'NetworkError';
    this.cause = cause;
  }
}

function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  for (const part of document.cookie.split('; ')) {
    if (part.startsWith(prefix)) {
      return decodeURIComponent(part.slice(prefix.length));
    }
  }
  return null;
}

function csrfToken(): string {
  // The server-rendered meta tag is authoritative on first paint; the cookie
  // carries the token forward after a client-side refresh.
  const meta = document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]');
  return meta?.content || readCookie(CSRF_COOKIE) || '';
}

interface RequestOptions {
  readonly method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  readonly body?: unknown;
  readonly signal?: AbortSignal;
  readonly query?: Readonly<Record<string, string | number | boolean | undefined>>;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== '') params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? `${path}?${encoded}` : path;
}

async function toProblem(response: Response): Promise<ProblemDocument> {
  try {
    const parsed: unknown = await response.json();
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      'code' in parsed &&
      'title' in parsed
    ) {
      return parsed as ProblemDocument;
    }
  } catch {
    // Fall through to a synthetic problem below.
  }
  return {
    type: 'about:blank',
    title: 'Непредвиденная ошибка сервера.',
    status: response.status,
    code: response.status >= 500 ? 'internal_error' : 'bad_request',
  };
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, query } = options;

  const headers = new Headers({ Accept: 'application/json' });
  const isFormData = body instanceof FormData;
  if (body !== undefined && !isFormData) {
    headers.set('Content-Type', 'application/json');
  }
  if (method !== 'GET') {
    headers.set(CSRF_HEADER, csrfToken());
  }

  const init: RequestInit = {
    method,
    headers,
    credentials: 'same-origin',
    ...(signal ? { signal } : {}),
  };
  if (body !== undefined) {
    init.body = isFormData ? body : JSON.stringify(body);
  }

  let response: Response;
  try {
    response = await fetch(buildUrl(path, query), init);
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new NetworkError(cause);
  }

  if (!response.ok) {
    const problem = await toProblem(response);
    // An expired session anywhere in the app means the whole page is stale;
    // bouncing to login is less confusing than a wall of 401 toasts.
    if (problem.code === 'unauthenticated' && !location.pathname.startsWith('/login')) {
      location.assign(`/login?next=${encodeURIComponent(location.pathname)}`);
    }
    throw new ApiError(problem);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------
export interface RegisterPayload {
  readonly fullName: string;
  readonly email: string;
  readonly organizationName: string;
  readonly password: string;
  readonly passwordConfirm: string;
}

export interface LoginPayload {
  readonly email: string;
  readonly password: string;
}

interface SessionResponse {
  readonly user: User;
  readonly csrfToken: string;
}

/**
 * Where a study's rendered assets live.
 *
 * List endpoints send these URLs alongside each row; the dashboard's review
 * queue does not, because it is addressed by `publicId` alone and building two
 * strings client-side is cheaper than widening the payload.
 */
export const studyAssets = {
  thumbnail: (publicId: string): string => `/api/v1/studies/${publicId}/thumbnail`,
  page: (publicId: string): string => `/app/studies/${publicId}`,
} as const;

/**
 * Fetch a CBCT's voxels as a raw buffer.
 *
 * Outside the JSON client on purpose: the payload is up to 16 MB of binary and
 * the request has to stay a plain conditional GET so the browser's HTTP cache
 * and the endpoint's `ETag` do their job. Wrapping it in `request()` would
 * parse it as JSON and defeat both.
 */
export async function fetchVoxels(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetch(url, {
      credentials: 'same-origin',
      ...(signal ? { signal } : {}),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new NetworkError(cause);
  }
  if (!response.ok) throw new ApiError(await toProblem(response));
  return response.arrayBuffer();
}

export const api = {
  auth: {
    register: (body: RegisterPayload): Promise<SessionResponse> =>
      request('/api/v1/auth/register', { method: 'POST', body }),
    login: (body: LoginPayload): Promise<SessionResponse> =>
      request('/api/v1/auth/login', { method: 'POST', body }),
    logout: (): Promise<{ ok: boolean }> =>
      request('/api/v1/auth/logout', { method: 'POST' }),
    me: (signal?: AbortSignal): Promise<User> =>
      request('/api/v1/auth/me', signal ? { signal } : {}),
  },

  patients: {
    list: (
      query: { q?: string; limit?: number; offset?: number; includeArchived?: boolean },
      signal?: AbortSignal,
    ): Promise<Page<PatientSummary>> =>
      request('/api/v1/patients', {
        query: {
          q: query.q,
          limit: query.limit,
          offset: query.offset,
          include_archived: query.includeArchived,
        },
        ...(signal ? { signal } : {}),
      }),
    get: (id: number): Promise<Patient> => request(`/api/v1/patients/${id}`),
    create: (body: Partial<Patient> & { fullName: string }): Promise<Patient> =>
      request('/api/v1/patients', { method: 'POST', body }),
    update: (id: number, body: Partial<Patient> & { fullName: string }): Promise<Patient> =>
      request(`/api/v1/patients/${id}`, { method: 'PUT', body }),
    archive: (id: number): Promise<{ ok: boolean }> =>
      request(`/api/v1/patients/${id}`, { method: 'DELETE' }),
    restore: (id: number): Promise<Patient> =>
      request(`/api/v1/patients/${id}/restore`, { method: 'POST' }),
    overview: (id: number, signal?: AbortSignal): Promise<PatientOverview> =>
      request(`/api/v1/patients/${id}/overview`, signal ? { signal } : {}),
  },

  studies: {
    list: (
      query: { q?: string; patientId?: number; limit?: number; offset?: number },
      signal?: AbortSignal,
    ): Promise<Page<StudyListItem>> =>
      request('/api/v1/studies', {
        query: {
          q: query.q,
          patient_id: query.patientId,
          limit: query.limit,
          offset: query.offset,
        },
        ...(signal ? { signal } : {}),
      }),
    get: (publicId: string, signal?: AbortSignal): Promise<Study> =>
      request(`/api/v1/studies/${publicId}`, signal ? { signal } : {}),
    upload: (file: File, patientId?: number): Promise<Study> => {
      const form = new FormData();
      form.append('file', file);
      if (patientId !== undefined) form.append('patient_id', String(patientId));
      return request('/api/v1/studies', { method: 'POST', body: form });
    },
    update: (
      publicId: string,
      body: { patientId?: number | null; notes?: string | null },
    ): Promise<Study> =>
      request(`/api/v1/studies/${publicId}`, { method: 'PATCH', body }),
    remove: (publicId: string): Promise<{ ok: boolean }> =>
      request(`/api/v1/studies/${publicId}`, { method: 'DELETE' }),
    reviewFinding: (
      publicId: string,
      findingId: number,
      review: FindingReview,
    ): Promise<Finding> =>
      request(`/api/v1/studies/${publicId}/findings/${findingId}`, {
        method: 'PATCH',
        body: { review },
      }),
    setFindingTooth: (
      publicId: string,
      findingId: number,
      toothNumber: number | null,
    ): Promise<Finding> =>
      request(`/api/v1/studies/${publicId}/findings/${findingId}/tooth`, {
        method: 'PUT',
        body: { toothNumber },
      }),
    report: (publicId: string, signal?: AbortSignal): Promise<StudyReport> =>
      request(`/api/v1/studies/${publicId}/report`, signal ? { signal } : {}),
  },

  scans: {
    list: (patientId: number, signal?: AbortSignal): Promise<Page<Scan>> =>
      request('/api/v1/scans', {
        query: { patient_id: patientId },
        ...(signal ? { signal } : {}),
      }),
    get: (publicId: string, signal?: AbortSignal): Promise<Scan> =>
      request(`/api/v1/scans/${publicId}`, signal ? { signal } : {}),
    upload: (
      file: File,
      details: {
        patientId: number;
        kind: ScanKind;
        arch: ScanArch;
        capturedOn?: string;
        notes?: string;
      },
    ): Promise<Scan> => {
      const form = new FormData();
      form.append('file', file);
      form.append('patient_id', String(details.patientId));
      form.append('kind', details.kind);
      form.append('arch', details.arch);
      if (details.capturedOn) form.append('captured_on', details.capturedOn);
      if (details.notes) form.append('notes', details.notes);
      return request('/api/v1/scans', { method: 'POST', body: form });
    },
    update: (
      publicId: string,
      body: { kind: ScanKind; arch: ScanArch; capturedOn: string | null; notes: string | null },
    ): Promise<Scan> => request(`/api/v1/scans/${publicId}`, { method: 'PATCH', body }),
    remove: (publicId: string): Promise<{ ok: boolean }> =>
      request(`/api/v1/scans/${publicId}`, { method: 'DELETE' }),
  },

  volumes: {
    list: (
      query: { patientId?: number; limit?: number; offset?: number },
      signal?: AbortSignal,
    ): Promise<Page<VolumeListItem>> =>
      request('/api/v1/volumes', {
        query: { patient_id: query.patientId, limit: query.limit, offset: query.offset },
        ...(signal ? { signal } : {}),
      }),
    get: (publicId: string, signal?: AbortSignal): Promise<Volume> =>
      request(`/api/v1/volumes/${publicId}`, signal ? { signal } : {}),
    upload: (
      file: File,
      details: {
        patientId: number;
        fieldOfView: VolumeFieldOfView;
        capturedOn?: string;
        notes?: string;
      },
    ): Promise<Volume> => {
      const form = new FormData();
      form.append('file', file);
      form.append('patient_id', String(details.patientId));
      form.append('field_of_view', details.fieldOfView);
      if (details.capturedOn) form.append('captured_on', details.capturedOn);
      if (details.notes) form.append('notes', details.notes);
      return request('/api/v1/volumes', { method: 'POST', body: form });
    },
    update: (
      publicId: string,
      body: { fieldOfView: VolumeFieldOfView; capturedOn: string | null; notes: string | null },
    ): Promise<Volume> => request(`/api/v1/volumes/${publicId}`, { method: 'PATCH', body }),
    remove: (publicId: string): Promise<{ ok: boolean }> =>
      request(`/api/v1/volumes/${publicId}`, { method: 'DELETE' }),
    reanalyse: (publicId: string, pipeline?: string): Promise<Volume> =>
      request(`/api/v1/volumes/${publicId}/analyse`, {
        method: 'POST',
        body: { pipeline: pipeline ?? null },
      }),
    reviewFinding: (
      publicId: string,
      findingId: number,
      review: FindingReview,
    ): Promise<VolumeFinding> =>
      request(`/api/v1/volumes/${publicId}/findings/${findingId}`, {
        method: 'PATCH',
        body: { review },
      }),
    addMeasurement: (
      publicId: string,
      body: {
        kind: MeasurementKind;
        plane: ViewPlane;
        points: readonly (readonly number[])[];
        label?: string;
        notes?: string | null;
      },
    ): Promise<Measurement> =>
      request(`/api/v1/volumes/${publicId}/measurements`, { method: 'POST', body }),
    removeMeasurement: (publicId: string, id: number): Promise<{ ok: boolean }> =>
      request(`/api/v1/volumes/${publicId}/measurements/${id}`, { method: 'DELETE' }),
    addAnnotation: (
      publicId: string,
      body: {
        kind: AnnotationKind;
        plane: ViewPlane;
        x: number;
        y: number;
        z: number;
        title: string;
        body?: string | null;
        volumeFindingId?: number | null;
      },
    ): Promise<VolumeAnnotation> =>
      request(`/api/v1/volumes/${publicId}/annotations`, { method: 'POST', body }),
    removeAnnotation: (publicId: string, id: number): Promise<{ ok: boolean }> =>
      request(`/api/v1/volumes/${publicId}/annotations/${id}`, { method: 'DELETE' }),
    pipelines: (signal?: AbortSignal): Promise<readonly PipelineDescriptor[]> =>
      request('/api/v1/volumes/pipelines', signal ? { signal } : {}),
  },

  planning: {
    /** Propose a **draft** plan. Nothing is scheduled until an option is accepted. */
    generate: (body: {
      patientId: number;
      volumePublicId?: string;
      studyPublicId?: string;
    }): Promise<TreatmentPlan> =>
      request('/api/v1/planning/generate', { method: 'POST', body }),
    accept: (publicId: string, approach: TreatmentApproach): Promise<TreatmentPlan> =>
      request(`/api/v1/planning/${publicId}/accept`, {
        method: 'POST',
        body: { approach },
      }),
  },

  assistant: {
    ask: (body: {
      question: string;
      threadPublicId?: string;
      patientId?: number;
      volumePublicId?: string;
    }): Promise<AskResult> => request('/api/v1/assistant/ask', { method: 'POST', body }),
    thread: (publicId: string, signal?: AbortSignal): Promise<AssistantThread> =>
      request(`/api/v1/assistant/threads/${publicId}`, signal ? { signal } : {}),
  },

  treatment: {
    procedures: (signal?: AbortSignal): Promise<readonly ProcedureOption[]> =>
      request('/api/v1/treatment/procedures', signal ? { signal } : {}),
    plans: (patientId: number, signal?: AbortSignal): Promise<readonly TreatmentPlan[]> =>
      request('/api/v1/treatment/plans', {
        query: { patient_id: patientId },
        ...(signal ? { signal } : {}),
      }),
    createPlan: (body: { patientId: number; title?: string }): Promise<TreatmentPlan> =>
      request('/api/v1/treatment/plans', { method: 'POST', body }),
    proposeFromStudy: (studyPublicId: string, planPublicId?: string): Promise<TreatmentPlan> =>
      request('/api/v1/treatment/plans/propose', {
        method: 'POST',
        body: { studyPublicId, planPublicId: planPublicId ?? null },
      }),
    updatePlan: (
      publicId: string,
      body: { title: string; status: PlanStatus; notes: string | null },
    ): Promise<TreatmentPlan> =>
      request(`/api/v1/treatment/plans/${publicId}`, { method: 'PUT', body }),
    removePlan: (publicId: string): Promise<{ ok: boolean }> =>
      request(`/api/v1/treatment/plans/${publicId}`, { method: 'DELETE' }),
    addItem: (
      publicId: string,
      body: { procedureCode: string; toothNumber: number | null; notes?: string | null },
    ): Promise<PlanItem> =>
      request(`/api/v1/treatment/plans/${publicId}/items`, { method: 'POST', body }),
    updateItem: (
      publicId: string,
      itemId: number,
      body: {
        status: PlanItemStatus;
        toothNumber: number | null;
        scheduledFor: string | null;
        estimatedVisits: number;
        estimatedMinutes: number;
        notes: string | null;
      },
    ): Promise<PlanItem> =>
      request(`/api/v1/treatment/plans/${publicId}/items/${itemId}`, {
        method: 'PATCH',
        body,
      }),
    removeItem: (publicId: string, itemId: number): Promise<{ ok: boolean }> =>
      request(`/api/v1/treatment/plans/${publicId}/items/${itemId}`, { method: 'DELETE' }),
  },

  dashboard: (signal?: AbortSignal): Promise<Dashboard> =>
    request('/api/v1/dashboard', signal ? { signal } : {}),

  settings: {
    updateProfile: (body: { fullName: string; locale: string }): Promise<User> =>
      request('/api/v1/settings/profile', { method: 'PUT', body }),
    changePassword: (body: {
      currentPassword: string;
      newPassword: string;
      newPasswordConfirm: string;
    }): Promise<{ ok: boolean }> =>
      request('/api/v1/settings/password', { method: 'PUT', body }),
  },
} as const;
