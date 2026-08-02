/**
 * Patient page: one record, four views of it.
 *
 * The timeline is the spine — everything that happened to this patient in one
 * dated column — and studies, scans and the treatment plan hang off it. All of
 * it comes from a single `/overview` request, so switching tabs never waits on
 * the network.
 */

import { api } from '../lib/api';
import { delegate, el, maybe, must, on, replaceChildren, setBusy } from '../lib/dom';
import { formatDate, formatDateTime, formatNumber, formatRelative, plural } from '../lib/format';
import { icon, type IconName } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type {
  PatientOverview,
  ProcedureOption,
  Scan,
  StudyListItem,
  TimelineEntry,
} from '../lib/types';
import { bindPlanActions, renderPlans } from './treatment-plan';

const TIMELINE_ICONS: Readonly<Record<string, IconName>> = {
  'user-plus': 'user-plus',
  scan: 'scan',
  cube: 'cube',
  clipboard: 'clipboard',
};

function timelineEntry(entry: TimelineEntry): HTMLElement {
  const glyph = TIMELINE_ICONS[entry.icon] ?? 'activity';
  const body = el(
    'div',
    { class: 'journey-body' },
    el('p', { class: 'journey-title' }, entry.title),
    entry.subtitle && el('p', { class: 'journey-subtitle' }, entry.subtitle),
    el('time', { class: 'journey-time', dateTime: entry.at }, formatDateTime(entry.at)),
  );

  return el(
    'li',
    {
      class: 'journey-entry',
      dataset: { kind: entry.kind, ...(entry.severity ? { severity: entry.severity } : {}) },
    },
    el('span', { class: 'journey-marker', aria: { hidden: 'true' } }, icon(glyph)),
    entry.href ? el('a', { class: 'journey-link', href: entry.href }, body) : body,
  );
}

function studyCard(study: StudyListItem): HTMLElement {
  return el(
    'a',
    { class: 'record-card', href: `/app/studies/${study.publicId}` },
    el('img', {
      class: 'record-thumb',
      src: study.thumbnailUrl,
      alt: '',
      loading: 'lazy',
      decoding: 'async',
    }),
    el(
      'div',
      { class: 'record-body' },
      el('p', { class: 'record-title' }, study.originalFilename),
      el(
        'p',
        { class: 'record-meta' },
        `${study.findingCount} ${plural(study.findingCount, 'находка', 'находки', 'находок')}`,
        study.attentionCount > 0 ? ` · требуют внимания: ${study.attentionCount}` : '',
      ),
      el('p', { class: 'record-meta record-meta--dim' }, formatRelative(study.createdAt)),
    ),
    study.topSeverityLabel &&
      el(
        'span',
        { class: 'badge badge--severity', dataset: { severity: study.topSeverity ?? 'info' } },
        study.topSeverityLabel,
      ),
  );
}

function scanCard(scan: Scan): HTMLElement {
  const [width, depth, height] = scan.bounds.size;
  return el(
    'a',
    { class: 'record-card record-card--scan', href: scan.pageUrl },
    el('span', { class: 'record-glyph', aria: { hidden: 'true' } }, icon('cube', { class: 'icon--lg' })),
    el(
      'div',
      { class: 'record-body' },
      el('p', { class: 'record-title' }, scan.kindLabel),
      el(
        'p',
        { class: 'record-meta' },
        `${scan.archLabel} · ${formatNumber(scan.triangleCount)} треугольников`,
      ),
      el(
        'p',
        { class: 'record-meta record-meta--dim' },
        `${width} × ${depth} × ${height} мм · `,
        scan.capturedOn ? formatDate(scan.capturedOn) : formatRelative(scan.createdAt),
      ),
    ),
    el('span', { class: 'badge badge--soft' }, scan.sourceFormat.toUpperCase()),
  );
}

function emptyState(glyph: IconName, title: string, body: string): HTMLElement {
  return el(
    'div',
    { class: 'state' },
    el('div', { class: 'state-icon' }, icon(glyph, { class: 'icon--lg' })),
    el('p', { class: 'state-title' }, title),
    el('p', { class: 'state-body' }, body),
  );
}

export function initPatientDetail(patientId: number): void {
  const timelineNode = must('[data-timeline]');
  const studiesNode = must('[data-patient-studies]');
  const scansNode = must('[data-patient-scans]');
  const plansNode = must('[data-patient-plans]');
  const tabs = document.querySelectorAll<HTMLButtonElement>('[data-patient-tab]');

  let overview: PatientOverview | null = null;
  let procedures: readonly ProcedureOption[] = [];

  function renderHeader(data: PatientOverview): void {
    const { patient } = data;
    const values: Record<string, string> = {
      name: patient.fullName,
      chart: patient.medicalRecordNumber ?? '—',
      phone: patient.phone ?? '—',
      birth: patient.dateOfBirth
        ? `${formatDate(patient.dateOfBirth)}${patient.age !== null ? ` · ${patient.age} ${plural(patient.age, 'год', 'года', 'лет')}` : ''}`
        : '—',
      notes: patient.notes ?? '—',
    };
    for (const [key, value] of Object.entries(values)) {
      const node = maybe(`[data-patient-${key}]`);
      if (!node) continue;
      node.textContent = value;
      node.classList.remove('skeleton', 'skeleton--title', 'skeleton--line', 'skeleton--text');
    }

    const counts: Record<string, number> = {
      studies: patient.studyCount,
      scans: patient.scanCount,
      'open-items': patient.openPlanItems,
    };
    for (const [key, value] of Object.entries(counts)) {
      const node = maybe(`[data-count-${key}]`);
      if (node) node.textContent = String(value);
    }
  }

  function render(data: PatientOverview): void {
    renderHeader(data);

    replaceChildren(
      timelineNode,
      data.timeline.length === 0
        ? emptyState('activity', 'Пока ничего не произошло', 'Загрузите снимок или 3D-модель.')
        : el('ul', { class: 'journey' }, ...data.timeline.map(timelineEntry)),
    );

    replaceChildren(
      studiesNode,
      data.studies.length === 0
        ? emptyState('scan', 'Снимков нет', 'Загрузите первый снимок на странице «Снимки».')
        : el('div', { class: 'record-grid' }, ...data.studies.map(studyCard)),
    );

    replaceChildren(
      scansNode,
      data.scans.length === 0
        ? emptyState(
            'cube',
            '3D-моделей нет',
            'Загрузите интраоральный скан или скан модели в формате STL, PLY или OBJ.',
          )
        : el('div', { class: 'record-grid' }, ...data.scans.map(scanCard)),
    );

    renderPlans(plansNode, data.plans, procedures);
  }

  async function reload(): Promise<void> {
    try {
      overview = await api.patients.overview(patientId);
      render(overview);
    } catch (error) {
      notifyError(error);
      replaceChildren(
        timelineNode,
        emptyState('alert', 'Не удалось загрузить карту', 'Обновите страницу.'),
      );
    }
  }

  // -- tabs -----------------------------------------------------------------
  for (const tab of tabs) {
    on(tab, 'click', () => {
      for (const other of tabs) {
        const active = other === tab;
        other.setAttribute('aria-selected', String(active));
        other.tabIndex = active ? 0 : -1;
        const panel = document.getElementById(other.getAttribute('aria-controls') ?? '');
        if (panel) panel.hidden = !active;
      }
    });
  }

  // -- 3D scan upload -------------------------------------------------------
  const uploadDialog = maybe<HTMLDialogElement>('[data-scan-dialog]');
  const uploadTrigger = maybe<HTMLButtonElement>('[data-upload-scan]');
  const uploadForm = maybe<HTMLFormElement>('[data-scan-form]');

  if (uploadTrigger && uploadDialog) {
    on(uploadTrigger, 'click', () => uploadDialog.showModal());
  }

  if (uploadForm) {
    on(uploadForm, 'submit', (event) => {
      event.preventDefault();
      const values = new FormData(uploadForm);
      const file = values.get('file');
      if (!(file instanceof File) || file.size === 0) {
        notify.warning('Выберите файл модели');
        return;
      }

      const submit = uploadForm.querySelector<HTMLButtonElement>('[type="submit"]');
      void (async (): Promise<void> => {
        if (submit) setBusy(submit, true);
        try {
          await api.scans.upload(file, {
            patientId,
            kind: String(values.get('kind') ?? 'intraoral') as Scan['kind'],
            arch: String(values.get('arch') ?? 'both') as Scan['arch'],
            ...(values.get('capturedOn') ? { capturedOn: String(values.get('capturedOn')) } : {}),
            ...(values.get('notes') ? { notes: String(values.get('notes')) } : {}),
          });
          notify.success('3D-модель загружена');
          uploadDialog?.close();
          uploadForm.reset();
          await reload();
        } catch (error) {
          notifyError(error);
        } finally {
          if (submit) setBusy(submit, false);
        }
      })();
    });
  }

  // -- plan actions ---------------------------------------------------------
  bindPlanActions(plansNode, { onChanged: () => void reload() });

  const proposeButton = maybe<HTMLButtonElement>('[data-propose-plan]');
  if (proposeButton) {
    on(proposeButton, 'click', () => {
      const latest = overview?.studies[0];
      if (!latest) {
        notify.warning('Нужен хотя бы один проанализированный снимок');
        return;
      }
      void (async (): Promise<void> => {
        setBusy(proposeButton, true);
        try {
          const plan = await api.treatment.proposeFromStudy(latest.publicId);
          notify.success(
            plan.items.length === 0
              ? 'По этому снимку протокол ничего не предлагает'
              : 'Черновик плана обновлён',
          );
          await reload();
        } catch (error) {
          notifyError(error);
        } finally {
          setBusy(proposeButton, false);
        }
      })();
    });
  }

  const newPlanButton = maybe<HTMLButtonElement>('[data-new-plan]');
  if (newPlanButton) {
    on(newPlanButton, 'click', () => {
      void (async (): Promise<void> => {
        setBusy(newPlanButton, true);
        try {
          await api.treatment.createPlan({ patientId });
          notify.success('План создан');
          await reload();
        } catch (error) {
          notifyError(error);
        } finally {
          setBusy(newPlanButton, false);
        }
      })();
    });
  }

  delegate(document.body, 'click', '[data-close-dialog]', (_event, target) => {
    target.closest('dialog')?.close();
  });

  void (async (): Promise<void> => {
    try {
      procedures = await api.treatment.procedures();
    } catch {
      // The catalogue only powers the "add step" picker; the page is usable
      // without it, and the toast for the overview failure would be duplicated.
      procedures = [];
    }
    await reload();
  })();
}
