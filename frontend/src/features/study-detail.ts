/**
 * Study detail: viewer, findings list, odontogram, clinician review.
 *
 * The findings list and the overlay are two views of one selection, so
 * hovering a row highlights the box and clicking a box scrolls to the row.
 * Review actions are optimistic — the UI updates immediately and rolls back
 * if the server disagrees.
 */

import { api } from '../lib/api';
import { announce, delegate, el, maybe, must, on, replaceChildren, setBusy } from '../lib/dom';
import { formatDateTime, formatDuration, formatPercent } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { Finding, FindingReview, Recommendation, Study, StudyReport } from '../lib/types';
import { highlightTooth, renderOdontogram } from './odontogram';
import { bindViewerControls, buildLegend, mountViewer, type StudyViewer } from './study-viewer';

function toothField(finding: Finding): HTMLElement {
  return el('input', {
    class: 'input input--sm tooth-input',
    type: 'number',
    min: '11',
    max: '48',
    value: finding.toothNumber === null ? '' : String(finding.toothNumber),
    placeholder: '—',
    dataset: { action: 'tooth', confirmed: String(finding.toothConfirmed) },
    title: finding.toothName ?? 'Номер зуба по FDI',
    aria: { label: `Номер зуба для находки «${finding.label}»` },
  });
}

function findingRow(finding: Finding): HTMLElement {
  const reviewed = finding.review !== 'unreviewed';

  return el(
    'li',
    {
      class: 'finding-row',
      dataset: {
        findingId: String(finding.id),
        severity: finding.severity,
        review: finding.review,
        tooth: finding.toothNumber === null ? '' : String(finding.toothNumber),
      },
    },
    el(
      'button',
      {
        class: 'finding-main',
        type: 'button',
        dataset: { action: 'select' },
        aria: {
          label:
            `${finding.label}, ${finding.severityLabel}, ` +
            `уверенность ${formatPercent(finding.confidence)}`,
        },
      },
      el('span', { class: 'finding-severity-bar' }),
      el(
        'span',
        { class: 'finding-text' },
        el('span', { class: 'finding-label' }, finding.label),
        el(
          'span',
          { class: 'finding-meta' },
          el(
            'span',
            { class: 'badge badge--severity', dataset: { severity: finding.severity } },
            finding.severityLabel,
          ),
          el('span', { class: 'finding-confidence' }, formatPercent(finding.confidence)),
        ),
      ),
    ),
    el(
      'div',
      { class: 'finding-actions', dataset: { reviewed: String(reviewed) } },
      toothField(finding),
      el(
        'button',
        {
          class: 'btn btn--sm btn--icon btn--ghost finding-action finding-action--confirm',
          type: 'button',
          dataset: { action: 'confirm' },
          title: 'Подтвердить находку',
          aria: {
            pressed: String(finding.review === 'confirmed'),
            label: `Подтвердить: ${finding.label}`,
          },
        },
        icon('check', { class: 'icon--sm' }),
      ),
      el(
        'button',
        {
          class: 'btn btn--sm btn--icon btn--ghost finding-action finding-action--reject',
          type: 'button',
          dataset: { action: 'reject' },
          title: 'Отклонить находку',
          aria: {
            pressed: String(finding.review === 'rejected'),
            label: `Отклонить: ${finding.label}`,
          },
        },
        icon('close', { class: 'icon--sm' }),
      ),
    ),
  );
}

function recommendationRow(item: Recommendation): HTMLElement {
  return el(
    'li',
    { class: 'recommendation', dataset: { priority: item.priority } },
    el('span', { class: 'plan-priority-bar', aria: { hidden: 'true' } }),
    el(
      'div',
      { class: 'recommendation-body' },
      el(
        'p',
        { class: 'recommendation-title' },
        item.toothNumber !== null && el('span', { class: 'tooth-chip' }, String(item.toothNumber)),
        item.label,
      ),
      el('p', { class: 'recommendation-reason' }, item.reason),
    ),
    el(
      'span',
      { class: 'badge badge--priority', dataset: { priority: item.priority } },
      item.priorityLabel,
    ),
  );
}

export function initStudyDetail(publicId: string): void {
  const listNode = must('[data-findings-list]');
  const legendNode = maybe('[data-legend]');
  const chartNode = maybe('[data-odontogram]');
  const recommendationsNode = maybe('[data-recommendations]');
  const confidenceInput = maybe<HTMLInputElement>('[data-confidence-filter]');
  const confidenceLabel = maybe('[data-confidence-value]');

  let study: Study | null = null;
  let report: StudyReport | null = null;
  let viewer: StudyViewer | null = null;
  let selectedTooth: number | null = null;

  function renderList(): void {
    if (!study) return;
    const visible = viewer?.visibleFindings() ?? study.findings;

    if (visible.length === 0) {
      replaceChildren(
        listNode,
        el(
          'div',
          { class: 'state' },
          el('div', { class: 'state-icon' }, icon('filter', { class: 'icon--lg' })),
          el('p', { class: 'state-title' }, 'Находок нет'),
          el(
            'p',
            { class: 'state-body' },
            'При текущих фильтрах ничего не отображается. Снизьте порог уверенности.',
          ),
        ),
      );
      return;
    }

    replaceChildren(listNode, ...visible.map(findingRow));
    applyToothFilter();
  }

  function select(findingId: number | null): void {
    for (const row of listNode.querySelectorAll<HTMLElement>('.finding-row')) {
      const active = row.dataset['findingId'] === String(findingId);
      row.classList.toggle('is-selected', active);
      if (active) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  function applyToothFilter(): void {
    for (const row of listNode.querySelectorAll<HTMLElement>('.finding-row')) {
      const dimmed = selectedTooth !== null && row.dataset['tooth'] !== String(selectedTooth);
      row.classList.toggle('is-dimmed', dimmed);
    }
    if (chartNode) highlightTooth(chartNode, selectedTooth);
  }

  function renderReport(): void {
    if (!report) return;

    for (const [key, value] of Object.entries({
      summary: report.summary,
      disclaimer: report.disclaimer,
    })) {
      const node = maybe(`[data-report-${key}]`);
      if (node) {
        node.textContent = value;
        node.classList.remove('skeleton', 'skeleton--line', 'skeleton--text');
      }
    }

    if (chartNode) renderOdontogram(chartNode, report.chart, { selected: selectedTooth });

    if (recommendationsNode) {
      replaceChildren(
        recommendationsNode,
        report.recommendations.length === 0
          ? el(
              'p',
              { class: 'panel-subtitle' },
              'Протокол не предлагает вмешательств по этим находкам.',
            )
          : el(
              'ul',
              { class: 'recommendation-list' },
              ...report.recommendations.map(recommendationRow),
            ),
      );
    }
  }

  async function reloadReport(): Promise<void> {
    try {
      report = await api.studies.report(publicId);
      renderReport();
    } catch (error) {
      notifyError(error);
    }
  }

  async function setTooth(finding: Finding, raw: string): Promise<void> {
    const value = raw.trim() === '' ? null : Number(raw);
    if (value !== null && !Number.isInteger(value)) return;
    try {
      const updated = await api.studies.setFindingTooth(publicId, finding.id, value);
      if (study) {
        study = {
          ...study,
          findings: study.findings.map((item) => (item.id === updated.id ? updated : item)),
        };
      }
      announce(
        updated.toothNumber === null
          ? `Находка «${finding.label}» откреплена от зуба`
          : `Находка «${finding.label}» отнесена к зубу ${updated.toothNumber}`,
      );
      await reloadReport();
    } catch (error) {
      notifyError(error);
      renderList();
    }
  }

  async function review(finding: Finding, next: FindingReview): Promise<void> {
    if (!study) return;
    const previous = finding.review;
    const target = previous === next ? 'unreviewed' : next;

    // Optimistic: reflect the decision immediately, reconcile after.
    applyReview(finding.id, target);

    try {
      const updated = await api.studies.reviewFinding(publicId, finding.id, target);
      applyReview(finding.id, updated.review);
      announce(
        updated.review === 'confirmed'
          ? `${finding.label} подтверждена`
          : updated.review === 'rejected'
            ? `${finding.label} отклонена`
            : `Отметка снята с «${finding.label}»`,
      );
    } catch (error) {
      applyReview(finding.id, previous);
      notifyError(error);
    }
  }

  function applyReview(findingId: number, review: FindingReview): void {
    if (!study) return;
    const findings = study.findings.map((item) =>
      item.id === findingId ? { ...item, review } : item,
    );
    study = { ...study, findings };
    // Rejected findings drop out of the overlay, so the viewer needs the new
    // data too — rebuild it rather than mutating its internal copy.
    remount();
  }

  function remount(): void {
    if (!study) return;
    viewer?.destroy();
    viewer = mountViewer(study, select);
    if (legendNode) buildLegend(legendNode, study, viewer);
    bindViewerControls(document, viewer);
    if (confidenceInput) viewer.setMinConfidence(Number(confidenceInput.value) / 100);
    renderList();
  }

  delegate(listNode, 'click', '[data-action]', (_event, target) => {
    const row = target.closest<HTMLElement>('.finding-row');
    const rawId = row?.dataset['findingId'];
    if (!rawId || !study) return;
    const finding = study.findings.find((item) => item.id === Number(rawId));
    if (!finding) return;

    switch (target.dataset['action']) {
      case 'select':
        viewer?.select(finding.id);
        select(finding.id);
        break;
      case 'confirm':
        void review(finding, 'confirmed');
        break;
      case 'reject':
        void review(finding, 'rejected');
        break;
      default:
        break;
    }
  });

  delegate(listNode, 'mouseover', '.finding-row', (_event, row) => {
    const id = row.dataset['findingId'];
    if (id) viewer?.select(Number(id));
  });

  delegate(listNode, 'change', '[data-action="tooth"]', (_event, target) => {
    const input = target as HTMLInputElement;
    const rawId = input.closest<HTMLElement>('.finding-row')?.dataset['findingId'];
    const finding = study?.findings.find((item) => item.id === Number(rawId));
    if (finding) void setTooth(finding, input.value);
  });

  if (chartNode) {
    delegate(chartNode, 'click', '.tooth', (_event, target) => {
      const tooth = Number(target.dataset['tooth']);
      // Clicking the selected tooth again clears the filter.
      selectedTooth = selectedTooth === tooth ? null : tooth;
      applyToothFilter();
    });
  }

  if (confidenceInput) {
    on(confidenceInput, 'input', () => {
      const ratio = Number(confidenceInput.value) / 100;
      viewer?.setMinConfidence(ratio);
      if (confidenceLabel) confidenceLabel.textContent = formatPercent(ratio);
      renderList();
    });
  }

  const proposeButton = maybe<HTMLButtonElement>('[data-propose-plan]');
  if (proposeButton) {
    on(proposeButton, 'click', () => {
      void (async (): Promise<void> => {
        setBusy(proposeButton, true);
        try {
          const plan = await api.treatment.proposeFromStudy(publicId);
          notify.success(
            plan.items.length === 0
              ? 'По этим находкам протокол ничего не предлагает'
              : 'Черновик плана обновлён',
          );
          if (study?.patient) {
            location.assign(`/app/patients/${study.patient.id}#plan`);
          }
        } catch (error) {
          notifyError(error);
        } finally {
          setBusy(proposeButton, false);
        }
      })();
    });
  }

  const deleteButton = maybe<HTMLButtonElement>('[data-delete-study]');
  if (deleteButton) {
    on(deleteButton, 'click', () => {
      const dialog = maybe<HTMLDialogElement>('[data-delete-dialog]');
      dialog?.showModal();
    });
  }

  const confirmDelete = maybe<HTMLButtonElement>('[data-confirm-delete]');
  if (confirmDelete) {
    on(confirmDelete, 'click', () => {
      void (async (): Promise<void> => {
        try {
          await api.studies.remove(publicId);
          notify.success('Снимок удалён');
          location.assign('/app/studies');
        } catch (error) {
          notifyError(error);
        }
      })();
    });
  }

  async function load(): Promise<void> {
    try {
      study = await api.studies.get(publicId);
    } catch (error) {
      notifyError(error);
      replaceChildren(
        listNode,
        el(
          'div',
          { class: 'state state--error' },
          el('div', { class: 'state-icon' }, icon('alert', { class: 'icon--lg' })),
          el('p', { class: 'state-title' }, 'Не удалось загрузить снимок'),
          el(
            'p',
            { class: 'state-body' },
            'Обновите страницу — снимок и находки не были изменены.',
          ),
        ),
      );
      return;
    }

    for (const [key, value] of Object.entries({
      filename: study.originalFilename,
      patient: study.patient?.fullName ?? 'Пациент не указан',
      created: formatDateTime(study.createdAt),
      model: study.modelVersion ?? '—',
      duration: formatDuration(study.inferenceMs),
      findings: String(study.findingCount),
      attention: String(study.attentionCount),
    })) {
      const node = maybe(`[data-study-${key}]`);
      if (node) {
        node.textContent = value;
        // Drop every skeleton class, not just the base one: the modifiers
        // carry a fixed height that would clip the real content.
        node.classList.remove('skeleton', 'skeleton--title', 'skeleton--line', 'skeleton--text');
      }
    }

    const patientLink = maybe<HTMLAnchorElement>('[data-study-patient-link]');
    if (patientLink && study.patient) {
      patientLink.href = `/app/patients/${study.patient.id}`;
      patientLink.hidden = false;
    }
    if (proposeButton) proposeButton.disabled = study.patient === null;

    remount();
    await reloadReport();
  }

  void load();
}
