/**
 * The CBCT case screen.
 *
 * Fetches the study, mounts the viewer, and keeps the side panels in step with
 * it. The division of labour is deliberate: this module talks to the API and
 * owns the DOM outside the canvases; {@link mountViewer} owns the canvases and
 * the shared view state. Neither reaches into the other except through the
 * store and a handful of callbacks.
 *
 * The metadata arrives as JSON and the voxels as a separate binary request, so
 * the panels, the finding list and the geometry readouts are all populated
 * while the 16 MB payload is still in flight. On a clinic's connection that is
 * the difference between a screen that appears instantly and one that appears
 * after the download.
 */

import { ApiError, api, fetchVoxels } from '../lib/api';
import { all, delegate, el, maybe, must, on, replaceChildren, setBusy } from '../lib/dom';
import { formatBytes, formatDateTime, formatNumber } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { Plane } from '../lib/dvol';
import { parseVolume } from '../lib/dvol';
import type { Measurement, Volume, VolumeAnnotation, VolumeFinding } from '../lib/types';
import type { Viewer } from './volume-viewer';
import { bindClassFilters, mountViewer } from './volume-viewer';
import { mountAssistantPanel } from './assistant-panel';
import { mountPlanPanel } from './plan-panel';

export async function initVolumeDetail(publicId: string): Promise<void> {
  const root = must('[data-viewer]');
  let record: Volume;
  try {
    record = await api.volumes.get(publicId);
  } catch (error) {
    notifyError(error);
    return;
  }

  renderHeader(record);
  renderMeta(record);
  renderQuality(record);
  renderFindings(record.findings, null);
  renderClassFilters(record.findings);
  renderMeasurements(record.measurements);
  renderAnnotations(record.annotations);
  renderPipeline(record);

  let viewer: Viewer | null = null;
  let findings = [...record.findings];
  let measurements = [...record.measurements];
  let annotations = [...record.annotations];

  // -- viewer ---------------------------------------------------------------
  try {
    const buffer = await fetchVoxels(record.voxelsUrl);
    const volume = parseVolume(buffer);
    viewer = mountViewer(
      root,
      volume,
      {
        findings: findings.map(toFindingOverlay),
        annotations: annotations.map(toAnnotationOverlay),
        measurements: measurements.map(toMeasurementOverlay),
      },
      {
        onMeasurement: (kind, plane, points) => {
          void saveMeasurement(kind, plane, points);
        },
        onAnnotate: (position) => openAnnotationDialog(position),
        onFindingSelected: (id) => renderFindings(findings, id),
      },
    );
    bindClassFilters(root, viewer.store);
    // The store drives the finding list's selected row as well as the canvases,
    // so a click in 3D highlights the same entry a click in the list does.
    viewer.store.subscribe(() => {
      const selected = viewer?.store.state.selectedFindingId ?? null;
      markSelected(selected);
      const clear = maybe<HTMLButtonElement>('[data-clear-isolate]', root);
      if (clear) clear.hidden = viewer?.store.state.isolatedFindingId === null;
    });
  } catch (error) {
    const notice = maybe('[data-volume-error]');
    if (notice) {
      notice.hidden = false;
      notice.textContent =
        error instanceof Error ? error.message : 'Не удалось загрузить том.';
    }
    notifyError(error);
  }

  // -- finding list ---------------------------------------------------------
  const list = must('[data-finding-list]');
  delegate(list, 'click', '[data-focus-finding]', (_event, target) => {
    const id = Number(target.dataset['focusFinding']);
    const finding = findings.find((item) => item.id === id);
    if (!finding || !viewer) return;
    viewer.focusFinding(id, [
      finding.box.x + finding.box.width / 2,
      finding.box.y + finding.box.height / 2,
      finding.box.z + finding.box.depth / 2,
    ]);
  });

  delegate(list, 'click', '[data-isolate-finding]', (event, target) => {
    event.stopPropagation();
    if (!viewer) return;
    const id = Number(target.dataset['isolateFinding']);
    const current = viewer.store.state.isolatedFindingId;
    viewer.store.update({ isolatedFindingId: current === id ? null : id });
  });

  delegate(list, 'click', '[data-review]', (event, target) => {
    event.stopPropagation();
    const id = Number(target.dataset['findingId']);
    const review = target.dataset['review'];
    if (!id || (review !== 'confirmed' && review !== 'rejected')) return;
    void reviewFinding(id, review);
  });

  const clearIsolate = maybe<HTMLButtonElement>('[data-clear-isolate]', root);
  if (clearIsolate) {
    on(clearIsolate, 'click', () => viewer?.store.update({ isolatedFindingId: null }));
  }

  async function reviewFinding(id: number, review: 'confirmed' | 'rejected'): Promise<void> {
    try {
      const updated = await api.volumes.reviewFinding(publicId, id, review);
      findings = findings.map((item) => (item.id === id ? updated : item));
      renderFindings(findings, viewer?.store.state.selectedFindingId ?? null);
      notify.success(review === 'confirmed' ? 'Находка подтверждена.' : 'Находка отклонена.');
    } catch (error) {
      notifyError(error);
    }
  }

  // -- measurements ---------------------------------------------------------
  async function saveMeasurement(
    kind: 'distance' | 'angle',
    plane: Plane,
    points: readonly (readonly [number, number, number])[],
  ): Promise<void> {
    try {
      const created = await api.volumes.addMeasurement(publicId, {
        kind,
        plane,
        points: points.map((point) => [...point]),
        label: kind === 'distance' ? 'Расстояние' : 'Угол',
      });
      measurements = [...measurements, created];
      renderMeasurements(measurements);
      viewer?.setOverlays({ measurements: measurements.map(toMeasurementOverlay) });
      notify.success(`${created.value} ${created.unit}`);
    } catch (error) {
      notifyError(error);
    }
  }

  const measurementList = must('[data-measurement-list]');
  delegate(measurementList, 'click', '[data-remove-measurement]', (_event, target) => {
    const id = Number(target.dataset['removeMeasurement']);
    void (async () => {
      try {
        await api.volumes.removeMeasurement(publicId, id);
        measurements = measurements.filter((item) => item.id !== id);
        renderMeasurements(measurements);
        viewer?.setOverlays({ measurements: measurements.map(toMeasurementOverlay) });
      } catch (error) {
        notifyError(error);
      }
    })();
  });

  // -- annotations ----------------------------------------------------------
  const dialog = maybe<HTMLDialogElement>('[data-annotation-dialog]');
  const form = maybe<HTMLFormElement>('[data-annotation-form]');
  let pendingPosition: readonly [number, number, number] | null = null;

  function openAnnotationDialog(position: readonly [number, number, number]): void {
    if (!dialog || !viewer) return;
    pendingPosition = position;
    const volume = viewer.store.volume;
    const readout = maybe('[data-annotation-position]');
    if (readout) {
      readout.textContent = `Положение: ${(position[0] * volume.physicalSize[0]).toFixed(1)} · ${(
        position[1] * volume.physicalSize[1]
      ).toFixed(1)} · ${(position[2] * volume.physicalSize[2]).toFixed(1)} мм`;
    }
    dialog.showModal();
  }

  const cancel = maybe<HTMLButtonElement>('[data-annotation-cancel]');
  if (cancel && dialog) on(cancel, 'click', () => dialog.close());

  if (form && dialog) {
    on(form, 'submit', (event) => {
      event.preventDefault();
      if (!pendingPosition) return;
      const data = new FormData(form);
      const title = String(data.get('title') ?? '').trim();
      if (!title) return;
      const body = String(data.get('body') ?? '').trim();
      const position = pendingPosition;

      void (async () => {
        try {
          const created = await api.volumes.addAnnotation(publicId, {
            kind: 'marker',
            plane: 'axial',
            x: position[0],
            y: position[1],
            z: position[2],
            title,
            body: body || null,
          });
          annotations = [...annotations, created];
          renderAnnotations(annotations);
          viewer?.setOverlays({ annotations: annotations.map(toAnnotationOverlay) });
          form.reset();
          dialog.close();
          notify.success('Аннотация сохранена.');
        } catch (error) {
          notifyError(error);
        }
      })();
    });
  }

  const annotationList = must('[data-annotation-list]');
  delegate(annotationList, 'click', '[data-remove-annotation]', (_event, target) => {
    const id = Number(target.dataset['removeAnnotation']);
    void (async () => {
      try {
        await api.volumes.removeAnnotation(publicId, id);
        annotations = annotations.filter((item) => item.id !== id);
        renderAnnotations(annotations);
        viewer?.setOverlays({ annotations: annotations.map(toAnnotationOverlay) });
      } catch (error) {
        notifyError(error);
      }
    })();
  });

  // -- assistant and planning ----------------------------------------------
  const assistant = mountAssistantPanel(document.body, { volumePublicId: publicId });
  const plans = mountPlanPanel(document.body, {
    patientId: record.patientId,
    volumePublicId: publicId,
  });

  // "Why did AI find this?" on a row asks the assistant about *that* finding,
  // which is the question a clinician actually has while looking at the list.
  delegate(list, 'click', '[data-explain-finding]', (event, target) => {
    event.stopPropagation();
    const label = target.dataset['explainFinding'];
    if (label) assistant.ask(`Почему AI нашёл: ${label}?`);
  });

  // -- header actions -------------------------------------------------------
  const reanalyse = maybe<HTMLButtonElement>('[data-reanalyse]');
  if (reanalyse) {
    on(reanalyse, 'click', () => {
      setBusy(reanalyse, true);
      void (async () => {
        try {
          const updated = await api.volumes.reanalyse(publicId);
          findings = [...updated.findings];
          renderFindings(findings, null);
          renderClassFilters(findings);
          renderQuality(updated);
          renderPipeline(updated);
          viewer?.setOverlays({ findings: findings.map(toFindingOverlay) });
          notify.success(`Анализ завершён: находок ${updated.findingCount}.`);
        } catch (error) {
          notifyError(error);
        } finally {
          setBusy(reanalyse, false);
        }
      })();
    });
  }

  const deleteButton = maybe<HTMLButtonElement>('[data-delete-volume]');
  const deleteDialog = maybe<HTMLDialogElement>('[data-delete-dialog]');
  const confirmDelete = maybe<HTMLButtonElement>('[data-confirm-delete]');
  if (deleteButton && deleteDialog) {
    on(deleteButton, 'click', () => deleteDialog.showModal());
  }
  if (confirmDelete) {
    on(confirmDelete, 'click', () => {
      setBusy(confirmDelete, true);
      void (async () => {
        try {
          await api.volumes.remove(publicId);
          location.assign('/app/volumes');
        } catch (error) {
          notifyError(error);
          setBusy(confirmDelete, false);
        }
      })();
    });
  }

  window.addEventListener(
    'pagehide',
    () => {
      viewer?.destroy();
      assistant.destroy();
      plans.destroy();
    },
    { once: true },
  );
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderHeader(record: Volume): void {
  setText('[data-volume-patient]', record.patientName ?? 'Без пациента');
  const heading = maybe('[data-volume-patient]');
  heading?.classList.remove('skeleton', 'skeleton--title');
  setText('[data-volume-filename]', record.originalFilename);
  setText('[data-volume-fov]', record.fieldOfViewLabel);
  setText('[data-volume-created]', formatDateTime(record.createdAt));

  const link = maybe<HTMLAnchorElement>('[data-volume-patient-link]');
  if (link) {
    link.href = `/app/patients/${record.patientId}`;
    link.hidden = false;
  }
}

function renderMeta(record: Volume): void {
  const geometry = record.geometry;
  setText(
    '[data-volume-grid]',
    `${geometry.width} × ${geometry.height} × ${geometry.depth} вокс.`,
  );
  setText(
    '[data-volume-spacing]',
    geometry.spacing.map((value) => value.toFixed(2)).join(' × ') + ' мм',
  );
  setText(
    '[data-volume-extent]',
    geometry.physicalSize.map((value) => value.toFixed(0)).join(' × ') + ' мм',
  );
  setText('[data-volume-slices]', formatNumber(record.sourceSliceCount));
  setText('[data-volume-format]', record.sourceFormat.toUpperCase());
  setText('[data-volume-bytes]', formatBytes(record.byteSize));
  setText('[data-volume-captured]', record.capturedOn ?? '—');
  setText('[data-volume-author]', record.uploadedByName ?? '—');
}

function renderQuality(record: Volume): void {
  const card = maybe('[data-quality-card]');
  if (!card) return;
  if (!record.quality) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const score = record.quality.score;
  setText('[data-quality-label]', record.quality.label);
  setText(
    '[data-quality-note]',
    record.analysisMs === null
      ? ''
      : `Оценка ${Math.round(score * 100)} из 100 · анализ ${formatNumber(record.analysisMs)} мс`,
  );

  const gauge = maybe('[data-quality-gauge]');
  if (gauge) {
    // A conic gradient rather than an SVG arc: one property, no geometry, and
    // it animates for free if the score ever updates in place.
    const degrees = Math.round(score * 360);
    const tone = score >= 0.7 ? 'var(--tone-positive)' : score >= 0.5 ? 'var(--tone-warning)' : 'var(--tone-critical)';
    gauge.style.setProperty('--gauge-angle', `${degrees}deg`);
    gauge.style.setProperty('--gauge-tone', tone);
    gauge.textContent = String(Math.round(score * 100));
  }
}

function renderFindings(findings: readonly VolumeFinding[], selectedId: number | null): void {
  const list = maybe('[data-finding-list]');
  if (!list) return;

  setText('[data-finding-count]', String(findings.length));
  const empty = maybe('[data-finding-empty]');
  if (empty) empty.hidden = findings.length > 0;

  replaceChildren(
    list,
    ...findings.map((finding) => {
      const row = el(
        'li',
        {
          class: 'vol-finding-row',
          dataset: {
            focusFinding: finding.id,
            severity: finding.severity,
            findingId: finding.id,
            selected: String(finding.id === selectedId),
            review: finding.review,
          },
        },
        el(
          'div',
          { class: 'vol-finding-main' },
          el(
            'div',
            { class: 'vol-finding-title' },
            el('span', { class: 'severity-dot' }),
            el('span', { class: 'vol-finding-label' }, finding.label),
            el('span', { class: 'vol-finding-confidence' }, `${Math.round(finding.confidence * 100)}%`),
          ),
          el(
            'p',
            { class: 'vol-finding-meta' },
            finding.regionLabel,
            finding.toothName ? ` · ${finding.toothName}` : '',
            finding.extentMm !== null ? ` · ${finding.extentMm} мм` : '',
          ),
          finding.requiresConfirmation
            ? el(
                'p',
                { class: 'vol-finding-warning' },
                icon('alert', { class: 'icon--sm' }),
                'Требует подтверждения специалистом. Не является диагнозом.',
              )
            : null,
          el('p', { class: 'vol-finding-rationale' }, finding.rationale),
          el('p', { class: 'vol-finding-next' }, icon('arrow-right', { class: 'icon--sm' }), finding.nextSteps),
        ),
        el(
          'div',
          { class: 'vol-finding-actions' },
          el(
            'button',
            {
              type: 'button',
              class: 'icon-btn',
              title: 'Показать только эту находку',
              dataset: { isolateFinding: finding.id },
            },
            icon('target', { class: 'icon--sm' }),
          ),
          el(
            'button',
            {
              type: 'button',
              class: 'icon-btn',
              title: 'Спросить ассистента, почему это найдено',
              dataset: { explainFinding: finding.label },
            },
            icon('sparkles', { class: 'icon--sm' }),
          ),
          el(
            'button',
            {
              type: 'button',
              class: 'icon-btn icon-btn--ok',
              title: 'Подтвердить',
              dataset: { review: 'confirmed', findingId: finding.id },
            },
            icon('check', { class: 'icon--sm' }),
          ),
          el(
            'button',
            {
              type: 'button',
              class: 'icon-btn icon-btn--no',
              title: 'Отклонить',
              dataset: { review: 'rejected', findingId: finding.id },
            },
            icon('close', { class: 'icon--sm' }),
          ),
        ),
      );
      return row;
    }),
  );
}

function renderClassFilters(findings: readonly VolumeFinding[]): void {
  const container = maybe('[data-class-filters]');
  if (!container) return;

  const seen = new Map<string, { label: string; severity: string; count: number }>();
  for (const finding of findings) {
    const existing = seen.get(finding.classKey);
    if (existing) existing.count += 1;
    else
      seen.set(finding.classKey, {
        label: finding.label,
        severity: finding.severity,
        count: 1,
      });
  }

  replaceChildren(
    container,
    ...[...seen.entries()].map(([key, meta]) =>
      el(
        'button',
        {
          type: 'button',
          class: 'class-chip',
          dataset: { classToggle: key, severity: meta.severity, hidden: 'false' },
          title: `Скрыть или показать: ${meta.label}`,
        },
        el('span', { class: 'severity-dot' }),
        meta.label,
        el('span', { class: 'chip-count' }, String(meta.count)),
      ),
    ),
  );
}

function renderMeasurements(measurements: readonly Measurement[]): void {
  const list = maybe('[data-measurement-list]');
  if (!list) return;
  setText('[data-measurement-count]', String(measurements.length));

  replaceChildren(
    list,
    ...measurements.map((measurement) =>
      el(
        'li',
        { class: 'measure-row' },
        el(
          'div',
          {},
          el('span', { class: 'measure-value' }, `${measurement.value} ${measurement.unit}`),
          el(
            'p',
            { class: 'measure-meta' },
            measurement.label || (measurement.kind === 'angle' ? 'Угол' : 'Расстояние'),
            ' · ',
            measurement.createdByName ?? '—',
          ),
        ),
        el(
          'button',
          {
            type: 'button',
            class: 'icon-btn icon-btn--no',
            title: 'Удалить измерение',
            dataset: { removeMeasurement: measurement.id },
          },
          icon('trash', { class: 'icon--sm' }),
        ),
      ),
    ),
  );
}

function renderAnnotations(annotations: readonly VolumeAnnotation[]): void {
  const list = maybe('[data-annotation-list]');
  if (!list) return;
  setText('[data-annotation-count]', String(annotations.length));

  replaceChildren(
    list,
    ...annotations.map((annotation) =>
      el(
        'li',
        { class: 'measure-row' },
        el(
          'div',
          {},
          el('span', { class: 'measure-value measure-value--text' }, annotation.title),
          el(
            'p',
            { class: 'measure-meta' },
            annotation.body ?? '',
            annotation.body ? ' · ' : '',
            annotation.createdByName ?? '—',
          ),
        ),
        el(
          'button',
          {
            type: 'button',
            class: 'icon-btn icon-btn--no',
            title: 'Удалить аннотацию',
            dataset: { removeAnnotation: annotation.id },
          },
          icon('trash', { class: 'icon--sm' }),
        ),
      ),
    ),
  );
}

function renderPipeline(record: Volume): void {
  setText('[data-pipeline-version]', record.pipelineVersion ?? 'Анализ не выполнялся');
  const list = maybe('[data-stage-list]');
  if (!list) return;
  // The per-stage log lives on the AI-run record, which the case endpoint does
  // not embed; the version string plus the analysis time is what this screen
  // shows, and the run log has its own view.
  replaceChildren(
    list,
    el(
      'li',
      { class: 'stage-row' },
      el('span', { class: 'stage-name' }, 'Время анализа'),
      el(
        'span',
        { class: 'stage-ms' },
        record.analysisMs === null ? '—' : `${formatNumber(record.analysisMs)} мс`,
      ),
    ),
    el(
      'li',
      { class: 'stage-row' },
      el('span', { class: 'stage-name' }, 'Находок'),
      el('span', { class: 'stage-ms' }, String(record.findingCount)),
    ),
    el(
      'li',
      { class: 'stage-row' },
      el('span', { class: 'stage-name' }, 'Требуют внимания'),
      el('span', { class: 'stage-ms' }, String(record.attentionCount)),
    ),
  );
}

function markSelected(selectedId: number | null): void {
  for (const row of all('[data-finding-id]')) {
    row.dataset['selected'] = String(Number(row.dataset['findingId']) === selectedId);
  }
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------
function toFindingOverlay(finding: VolumeFinding) {
  return {
    id: finding.id,
    classKey: finding.classKey,
    label: finding.label,
    severity: finding.severity,
    box: finding.box,
  };
}

function toAnnotationOverlay(annotation: VolumeAnnotation) {
  return {
    id: annotation.id,
    title: annotation.title,
    x: annotation.x,
    y: annotation.y,
    z: annotation.z,
  };
}

function toMeasurementOverlay(measurement: Measurement) {
  return {
    id: measurement.id,
    kind: measurement.kind,
    label: measurement.label,
    value: measurement.value,
    unit: measurement.unit,
    points: measurement.points,
  };
}

function setText(selector: string, value: string): void {
  const node = maybe(selector);
  if (node) node.textContent = value;
}

/** Re-exported so the error path can distinguish an API failure from a parse one. */
export { ApiError };
