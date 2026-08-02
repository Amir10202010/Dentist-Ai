/**
 * Wire types.
 *
 * These mirror the Pydantic models in `src/dentist_ai/schemas/`, hand-written
 * so the build needs no codegen step.
 */

export type Severity = 'critical' | 'high' | 'medium' | 'low' | 'info';

export type Category =
  | 'pathology'
  | 'restoration'
  | 'orthodontic'
  | 'anatomy'
  | 'condition';

export type StudyStatus = 'pending' | 'processing' | 'completed' | 'failed';

export type FindingReview = 'unreviewed' | 'confirmed' | 'rejected';

export type UserRole = 'owner' | 'dentist' | 'assistant';

export type Locale = 'ru' | 'en' | 'kk';

export interface Organization {
  readonly id: number;
  readonly name: string;
  readonly slug: string;
}

export interface User {
  readonly id: number;
  readonly email: string;
  readonly fullName: string;
  readonly initials: string;
  readonly role: UserRole;
  readonly locale: Locale;
  readonly lastLoginAt: string | null;
  readonly organization: Organization;
}

export interface BoundingBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface Finding {
  readonly id: number;
  readonly classId: number;
  readonly classKey: string;
  readonly label: string;
  readonly category: Category;
  readonly severity: Severity;
  readonly severityLabel: string;
  readonly confidence: number;
  readonly box: BoundingBox;
  /** FDI number, estimated on ingest unless a clinician corrected it. */
  readonly toothNumber: number | null;
  readonly toothName: string | null;
  readonly toothConfirmed: boolean;
  readonly review: FindingReview;
  readonly reviewedAt: string | null;
}

export interface Patient {
  readonly id: number;
  readonly fullName: string;
  readonly phone: string | null;
  readonly email: string | null;
  readonly dateOfBirth: string | null;
  readonly sex: 'male' | 'female' | 'unspecified';
  readonly medicalRecordNumber: string | null;
  readonly notes: string | null;
  readonly createdAt: string;
  readonly archivedAt: string | null;
  readonly age: number | null;
}

export interface PatientSummary extends Patient {
  readonly studyCount: number;
  readonly lastStudyAt: string | null;
  readonly scanCount: number;
  readonly openPlanItems: number;
}

export interface CategoryCount {
  readonly category: Category;
  readonly label: string;
  readonly count: number;
}

export interface Study {
  readonly publicId: string;
  readonly status: StudyStatus;
  readonly originalFilename: string;
  readonly width: number;
  readonly height: number;
  readonly byteSize: number;
  readonly modelVersion: string | null;
  readonly inferenceMs: number | null;
  readonly failureReason: string | null;
  readonly notes: string | null;
  readonly createdAt: string;
  readonly analyzedAt: string | null;
  readonly patient: Patient | null;
  readonly uploadedByName: string | null;
  readonly imageUrl: string;
  readonly thumbnailUrl: string;
  readonly findings: readonly Finding[];
  readonly categoryCounts: readonly CategoryCount[];
  readonly attentionCount: number;
  readonly findingCount: number;
  readonly topConfidence: number | null;
}

export interface StudyListItem {
  readonly publicId: string;
  readonly status: StudyStatus;
  readonly originalFilename: string;
  readonly createdAt: string;
  readonly thumbnailUrl: string;
  readonly patientId: number | null;
  readonly patientName: string | null;
  readonly findingCount: number;
  readonly attentionCount: number;
  readonly topSeverity: Severity | null;
  readonly topSeverityLabel: string | null;
}

export interface PageMeta {
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
  readonly hasMore: boolean;
}

export interface Page<T> {
  readonly items: readonly T[];
  readonly meta: PageMeta;
}

export interface TimeSeriesPoint {
  readonly date: string;
  readonly value: number;
}

export interface LabelledCount {
  readonly key: string;
  readonly label: string;
  readonly count: number;
  readonly severity: Severity | null;
}

/** Editorial register, not clinical severity. See `schemas/clinical.py`. */
export type Tone = 'positive' | 'info' | 'warning' | 'critical';

export interface MetricDelta {
  readonly current: number;
  readonly previous: number;
  /** `null` when the previous period was zero — growth from nothing is not a percentage. */
  readonly change: number | null;
}

export interface ReviewQueueItem {
  readonly publicId: string;
  readonly patientName: string | null;
  readonly originalFilename: string;
  readonly createdAt: string;
  readonly pendingCount: number;
  readonly topSeverity: Severity;
  readonly topSeverityLabel: string;
  readonly topFindingLabel: string;
  readonly topConfidence: number;
}

export interface ActivityItem {
  readonly id: number;
  readonly action: string;
  readonly actorName: string | null;
  readonly summary: string;
  readonly icon: string;
  readonly tone: Tone;
  readonly resourceType: string;
  readonly resourceId: string | null;
  readonly createdAt: string;
}

export interface Insight {
  readonly key: string;
  readonly tone: Tone;
  readonly icon: string;
  readonly title: string;
  readonly body: string;
  readonly metric: string | null;
  readonly actionLabel: string | null;
  readonly actionHref: string | null;
}

export interface PipelineStatus {
  readonly pending: number;
  readonly processing: number;
  readonly completedToday: number;
  readonly failedRecent: number;
}

export interface ReviewStats {
  readonly confirmed: number;
  readonly rejected: number;
  readonly unreviewed: number;
  readonly agreementRate: number | null;
  readonly averageConfidence: number | null;
}

export interface Dashboard {
  readonly generatedAt: string;
  readonly totalPatients: number;
  readonly newPatientsThisWeek: number;
  readonly totalStudies: number;
  readonly studiesThisWeek: number;
  readonly findingsNeedingAttention: number;
  readonly averageInferenceMs: number | null;
  readonly reviewedShare: number;
  readonly studiesOverTime: readonly TimeSeriesPoint[];
  readonly topFindings: readonly LabelledCount[];
  readonly categoryBreakdown: readonly LabelledCount[];
  readonly studiesDelta: MetricDelta;
  readonly patientsDelta: MetricDelta;
  readonly attentionDelta: MetricDelta;
  readonly reviewQueue: readonly ReviewQueueItem[];
  readonly reviewQueueTotal: number;
  readonly pendingFindings: number;
  readonly oldestPendingAt: string | null;
  readonly activity: readonly ActivityItem[];
  readonly insights: readonly Insight[];
  readonly pipeline: PipelineStatus;
  readonly reviewStats: ReviewStats;
}

// ---------------------------------------------------------------------------
// Odontogram and report
// ---------------------------------------------------------------------------
export interface ToothCell {
  readonly toothNumber: number;
  readonly severity: Severity | null;
  readonly findingCount: number;
  readonly hasRestoration: boolean;
  readonly isMissing: boolean;
}

export interface ToothGroup {
  readonly toothNumber: number;
  readonly toothName: string;
  readonly findings: readonly Finding[];
}

export type Priority = 'urgent' | 'high' | 'routine' | 'optional';

export type ProcedureCategory =
  | 'therapy'
  | 'endodontics'
  | 'surgery'
  | 'periodontics'
  | 'orthodontics'
  | 'prosthetics'
  | 'diagnostics';

export interface Recommendation {
  readonly procedureCode: string;
  readonly label: string;
  readonly category: ProcedureCategory;
  readonly categoryLabel: string;
  readonly priority: Priority;
  readonly priorityLabel: string;
  readonly toothNumber: number | null;
  readonly reason: string;
  readonly sourceFindingId: number | null;
}

export interface StudyReport {
  readonly studyPublicId: string;
  readonly generatedAt: string;
  readonly patientName: string | null;
  readonly summary: string;
  readonly findingCount: number;
  readonly attentionCount: number;
  readonly reviewedCount: number;
  readonly affectedTeeth: number;
  readonly chart: readonly ToothCell[];
  readonly teeth: readonly ToothGroup[];
  readonly regional: readonly Finding[];
  readonly recommendations: readonly Recommendation[];
  readonly disclaimer: string;
}

// ---------------------------------------------------------------------------
// 3D scans
// ---------------------------------------------------------------------------
export type MeshFormat = 'stl' | 'ply' | 'obj';

export type ScanKind = 'intraoral' | 'plaster_model' | 'cbct_surface' | 'restoration_design';

export type ScanArch = 'upper' | 'lower' | 'both';

export interface ScanBounds {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly size: readonly [number, number, number];
}

export interface Scan {
  readonly publicId: string;
  readonly patientId: number;
  readonly patientName: string | null;
  readonly originalFilename: string;
  readonly sourceFormat: MeshFormat;
  readonly kind: ScanKind;
  readonly kindLabel: string;
  readonly arch: ScanArch;
  readonly archLabel: string;
  readonly triangleCount: number;
  readonly byteSize: number;
  readonly bounds: ScanBounds;
  readonly capturedOn: string | null;
  readonly notes: string | null;
  readonly createdAt: string;
  readonly uploadedByName: string | null;
  readonly meshUrl: string;
  readonly pageUrl: string;
}

// ---------------------------------------------------------------------------
// Treatment plans
// ---------------------------------------------------------------------------
export type PlanStatus = 'draft' | 'active' | 'completed' | 'cancelled';

export type PlanItemStatus =
  | 'proposed'
  | 'accepted'
  | 'scheduled'
  | 'in_progress'
  | 'done'
  | 'declined';

export interface ProcedureOption {
  readonly code: string;
  readonly label: string;
  readonly category: ProcedureCategory;
  readonly categoryLabel: string;
  readonly priority: Priority;
  readonly priorityLabel: string;
  readonly visits: number;
  readonly minutes: number;
}

export interface PlanItem {
  readonly id: number;
  readonly procedureCode: string;
  readonly procedureLabel: string;
  readonly category: ProcedureCategory;
  readonly categoryLabel: string;
  readonly toothNumber: number | null;
  readonly toothName: string | null;
  readonly priority: Priority;
  readonly priorityLabel: string;
  readonly status: PlanItemStatus;
  readonly statusLabel: string;
  readonly estimatedVisits: number;
  readonly estimatedMinutes: number;
  readonly scheduledFor: string | null;
  readonly completedAt: string | null;
  readonly notes: string | null;
  readonly sourceFindingId: number | null;
  readonly sourceStudyPublicId: string | null;
}

export interface TreatmentPlan {
  readonly publicId: string;
  readonly patientId: number;
  readonly patientName: string | null;
  readonly title: string;
  readonly status: PlanStatus;
  readonly statusLabel: string;
  readonly notes: string | null;
  readonly createdAt: string;
  readonly createdByName: string | null;
  readonly items: readonly PlanItem[];
  readonly openCount: number;
  readonly doneCount: number;
  readonly totalVisits: number;
  readonly totalMinutes: number;

  /** Whether a clinician assembled this plan or the planner proposed it. */
  readonly origin: PlanOrigin;
  readonly complexity: PlanComplexity | null;
  readonly complexityLabel: string | null;
  readonly estimatedWeeks: number | null;
  readonly risks: string | null;
  readonly followUp: string | null;
  readonly rationale: string | null;
  readonly options: readonly TreatmentOption[];
}

// ---------------------------------------------------------------------------
// Patient overview
// ---------------------------------------------------------------------------
export type TimelineKind = 'patient_created' | 'study' | 'scan' | 'plan_item';

export interface TimelineEntry {
  readonly kind: TimelineKind;
  readonly at: string;
  readonly title: string;
  readonly subtitle: string | null;
  readonly href: string | null;
  readonly icon: string;
  readonly severity: Severity | null;
}

export interface PatientOverview {
  readonly patient: PatientSummary;
  readonly studies: readonly StudyListItem[];
  readonly scans: readonly Scan[];
  readonly plans: readonly TreatmentPlan[];
  readonly timeline: readonly TimelineEntry[];
}

/** Mirrors the `code` values in `src/dentist_ai/core/errors.py`. */
export type ApiErrorCode =
  | 'bad_request'
  | 'validation_failed'
  | 'unauthenticated'
  | 'invalid_credentials'
  | 'permission_denied'
  | 'csrf_failed'
  | 'not_found'
  | 'method_not_allowed'
  | 'conflict'
  | 'email_taken'
  | 'payload_too_large'
  | 'unsupported_media_type'
  | 'rate_limited'
  | 'inference_unavailable'
  | 'internal_error';

/** RFC 9457 problem document. */
export interface ProblemDocument {
  readonly type: string;
  readonly title: string;
  readonly status: number;
  readonly code: ApiErrorCode;
  readonly errors?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// CBCT volumes
// ---------------------------------------------------------------------------
export type VolumeFormat = 'dicom' | 'nifti';

export type VolumeFieldOfView =
  | 'full_head'
  | 'both_jaws'
  | 'maxilla'
  | 'mandible'
  | 'tmj'
  | 'sinus'
  | 'implant_site';

export type VolumeCategory =
  | 'pathology'
  | 'anatomy'
  | 'restoration'
  | 'structural'
  | 'quality';

export type MeasurementKind = 'distance' | 'angle' | 'density';

export type ViewPlane = 'axial' | 'coronal' | 'sagittal' | 'volume';

export type AnnotationKind = 'marker' | 'region' | 'question';

export interface VolumeGeometryPayload {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly spacing: readonly [number, number, number];
  readonly huSlope: number;
  readonly huIntercept: number;
  readonly windowCenter: number;
  readonly windowWidth: number;
  readonly physicalSize: readonly [number, number, number];
}

export interface BoundingBox3D {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly width: number;
  readonly height: number;
  readonly depth: number;
}

export interface VolumeFinding {
  readonly id: number;
  readonly classKey: string;
  readonly label: string;
  readonly category: VolumeCategory;
  readonly categoryLabel: string;
  readonly severity: Severity;
  readonly severityLabel: string;
  readonly confidence: number;
  readonly box: BoundingBox3D;
  readonly region: string;
  readonly regionLabel: string;
  readonly toothNumber: number | null;
  readonly toothName: string | null;
  readonly toothConfirmed: boolean;
  readonly extentMm: number | null;
  readonly meanDensity: number | null;
  /** Why the pipeline says this — the taxonomy's fixed explanation. */
  readonly rationale: string;
  readonly nextSteps: string;
  readonly producedBy: string;
  /** Must be presented as needing specialist confirmation, never as a diagnosis. */
  readonly requiresConfirmation: boolean;
  readonly review: FindingReview;
  readonly reviewedAt: string | null;
}

export interface Measurement {
  readonly id: number;
  readonly kind: MeasurementKind;
  readonly plane: ViewPlane;
  readonly label: string;
  readonly points: readonly (readonly number[])[];
  readonly value: number;
  readonly unit: string;
  readonly notes: string | null;
  readonly createdAt: string;
  readonly createdByName: string | null;
}

export interface VolumeAnnotation {
  readonly id: number;
  readonly kind: AnnotationKind;
  readonly plane: ViewPlane;
  readonly x: number;
  readonly y: number;
  readonly z: number | null;
  readonly title: string;
  readonly body: string | null;
  readonly volumeFindingId: number | null;
  readonly createdAt: string;
  readonly createdByName: string | null;
}

export interface VolumeQuality {
  readonly score: number;
  readonly label: string;
  readonly notes: readonly string[];
}

export interface VolumeCategoryCount {
  readonly category: VolumeCategory;
  readonly label: string;
  readonly count: number;
}

export interface Volume {
  readonly publicId: string;
  readonly patientId: number;
  readonly patientName: string | null;
  readonly originalFilename: string;
  readonly sourceFormat: VolumeFormat;
  readonly fieldOfView: VolumeFieldOfView;
  readonly fieldOfViewLabel: string;
  readonly status: StudyStatus;
  readonly failureReason: string | null;
  readonly byteSize: number;
  readonly sourceSliceCount: number;
  readonly geometry: VolumeGeometryPayload;
  readonly quality: VolumeQuality | null;
  readonly pipelineVersion: string | null;
  readonly analysisMs: number | null;
  readonly analyzedAt: string | null;
  readonly capturedOn: string | null;
  readonly notes: string | null;
  readonly createdAt: string;
  readonly uploadedByName: string | null;
  readonly voxelsUrl: string;
  readonly previewUrl: string;
  readonly pageUrl: string;
  readonly findings: readonly VolumeFinding[];
  readonly categoryCounts: readonly VolumeCategoryCount[];
  readonly attentionCount: number;
  readonly findingCount: number;
  readonly measurements: readonly Measurement[];
  readonly annotations: readonly VolumeAnnotation[];
}

export interface VolumeListItem {
  readonly publicId: string;
  readonly patientId: number;
  readonly patientName: string | null;
  readonly originalFilename: string;
  readonly fieldOfView: VolumeFieldOfView;
  readonly fieldOfViewLabel: string;
  readonly status: StudyStatus;
  readonly createdAt: string;
  readonly capturedOn: string | null;
  readonly previewUrl: string;
  readonly pageUrl: string;
  readonly findingCount: number;
  readonly attentionCount: number;
  readonly topSeverity: Severity | null;
  readonly topSeverityLabel: string | null;
  readonly qualityScore: number | null;
  readonly voxelCount: number;
}

export interface PipelineStage {
  readonly name: string;
  readonly kind: string;
  readonly kindLabel: string;
  readonly version: string;
  readonly status?: string;
  readonly ms?: number;
  readonly summary?: string;
}

export interface PipelineDescriptor {
  readonly name: string;
  readonly version: string;
  readonly stages: readonly PipelineStage[];
}

// ---------------------------------------------------------------------------
// Assistant
// ---------------------------------------------------------------------------
export interface Citation {
  readonly kind: string;
  readonly label: string;
  readonly href: string | null;
}

export interface AssistantAnswer {
  /** Which question shape the router matched; `capabilities` means it did not. */
  readonly intent: string;
  readonly body: string;
  readonly citations: readonly Citation[];
  readonly suggestions: readonly string[];
}

export interface AssistantMessage {
  readonly id: number;
  readonly role: 'user' | 'assistant';
  readonly body: string;
  readonly intent: string | null;
  readonly citations: readonly Citation[];
  readonly createdAt: string;
}

export interface AssistantThread {
  readonly publicId: string;
  readonly title: string;
  readonly patientId: number | null;
  readonly createdAt: string;
  readonly messages: readonly AssistantMessage[];
}

export interface AskResult {
  readonly threadPublicId: string;
  readonly answer: AssistantAnswer;
}

// ---------------------------------------------------------------------------
// Generated treatment plans
// ---------------------------------------------------------------------------
export type TreatmentApproach = 'conservative' | 'standard' | 'comprehensive';

export type PlanComplexity = 'simple' | 'moderate' | 'complex' | 'advanced';

export type PlanOrigin = 'manual' | 'generated';

export interface TreatmentOption {
  readonly position: number;
  readonly title: string;
  readonly approach: TreatmentApproach;
  readonly approachLabel: string;
  readonly summary: string;
  readonly priority: Priority;
  readonly priorityLabel: string;
  readonly complexity: PlanComplexity;
  readonly complexityLabel: string;
  readonly estimatedVisits: number;
  readonly estimatedMinutes: number;
  /** Calendar weeks including healing, which is not the same as chair time. */
  readonly estimatedWeeks: number;
  readonly benefits: string;
  readonly risks: string;
  readonly procedureCodes: readonly string[];
  readonly isSelected: boolean;
}
