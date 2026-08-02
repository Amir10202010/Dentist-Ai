/**
 * The CBCT list and upload screen.
 *
 * The upload is the interesting part. A scanner export is a zip of several
 * hundred DICOM files, routinely a couple of hundred megabytes, and it is
 * *dragged from a folder* — so the drop target is the whole panel rather than a
 * small box inside it, and the file is chosen before the metadata form opens
 * rather than after. Asking which patient a scan belongs to only once the
 * clinician has already picked the file keeps the dialog to the two questions
 * the server cannot answer for itself.
 *
 * Progress is reported from a real `XMLHttpRequest` rather than a spinner.
 * `fetch` cannot report upload progress, and a 200 MB transfer with no feedback
 * is indistinguishable from a hung page.
 */

import { api } from '../lib/api';
import { el, maybe, must, on, replaceChildren, setBusy } from '../lib/dom';
import { formatBytes, formatDateTime, formatNumber } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { PatientSummary, VolumeListItem } from '../lib/types';

const ACCEPTED = /\.(zip|nii|nii\.gz|gz|dcm)$/i;

export function initVolumesPage(): void {
  void load();
  initUpload();
}

async function load(): Promise<void> {
  const grid = maybe('[data-volume-grid]');
  if (!grid) return;
  try {
    const page = await api.volumes.list({ limit: 60 });
    render(page.items, page.meta.total);
  } catch (error) {
    notifyError(error);
    replaceChildren(grid);
  }
}

function render(items: readonly VolumeListItem[], total: number): void {
  const grid = must('[data-volume-grid]');
  const empty = maybe('[data-volume-empty]');
  const totalNode = maybe('[data-volume-total]');
  if (totalNode) totalNode.textContent = formatNumber(total);
  if (empty) empty.hidden = items.length > 0;

  replaceChildren(
    grid,
    ...items.map((item) =>
      el(
        'a',
        {
          class: 'volume-card',
          href: item.pageUrl,
          dataset: { severity: item.topSeverity ?? 'none', status: item.status },
        },
        el('img', {
          class: 'volume-card-preview',
          src: item.previewUrl,
          alt: `Превью КЛКТ: ${item.originalFilename}`,
          loading: 'lazy',
          decoding: 'async',
        }),
        el(
          'div',
          { class: 'volume-card-body' },
          el('p', { class: 'volume-card-patient' }, item.patientName ?? 'Без пациента'),
          el('p', { class: 'volume-card-file' }, item.originalFilename),
          el(
            'div',
            { class: 'volume-card-tags' },
            el('span', { class: 'tag' }, item.fieldOfViewLabel),
            item.attentionCount > 0
              ? el(
                  'span',
                  { class: 'tag tag--attention' },
                  icon('alert', { class: 'icon--sm' }),
                  `${item.attentionCount}`,
                )
              : el('span', { class: 'tag tag--ok' }, icon('check', { class: 'icon--sm' }), 'чисто'),
            item.qualityScore !== null
              ? el('span', { class: 'tag' }, `${Math.round(item.qualityScore * 100)}/100`)
              : null,
          ),
          el(
            'p',
            { class: 'volume-card-meta' },
            formatDateTime(item.createdAt),
            ' · ',
            `${formatNumber(Math.round(item.voxelCount / 1_000_000))} Мвокс`,
            item.findingCount > 0 ? ` · находок ${item.findingCount}` : '',
          ),
        ),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
function initUpload(): void {
  const dropzone = maybe('[data-dropzone]');
  const input = maybe<HTMLInputElement>('[data-file-input]');
  const dialog = maybe<HTMLDialogElement>('[data-upload-dialog]');
  const form = maybe<HTMLFormElement>('[data-upload-form]');
  if (!dropzone || !input || !dialog || !form) return;

  let chosen: File | null = null;

  const open = (file: File): void => {
    if (!ACCEPTED.test(file.name)) {
      notify.warning('Поддерживаются .zip, .dcm, .nii и .nii.gz.');
      return;
    }
    chosen = file;
    const label = maybe('[data-upload-filename]');
    if (label) label.textContent = `${file.name} · ${formatBytes(file.size)}`;
    void fillPatients();
    dialog.showModal();
  };

  // No click handler: the dropzone is a <label> wrapping the input, so the
  // browser already opens the picker. Adding one would fire it twice.
  on(dropzone, 'keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });
  on(input, 'change', () => {
    const file = input.files?.[0];
    if (file) open(file);
    // Cleared so choosing the same file twice still fires a change event.
    input.value = '';
  });

  for (const type of ['dragenter', 'dragover'] as const) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.dataset['state'] = 'dragging';
    });
  }
  for (const type of ['dragleave', 'dragend'] as const) {
    dropzone.addEventListener(type, () => {
      dropzone.dataset['state'] = '';
    });
  }
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.dataset['state'] = '';
    const file = event.dataTransfer?.files?.[0];
    if (file) open(file);
  });

  const cancel = maybe<HTMLButtonElement>('[data-upload-cancel]');
  if (cancel) on(cancel, 'click', () => dialog.close());

  on(form, 'submit', (event) => {
    event.preventDefault();
    if (!chosen) return;
    const data = new FormData(form);
    const patientId = Number(data.get('patientId'));
    if (!patientId) {
      notify.warning('Выберите пациента.');
      return;
    }

    const submit = maybe<HTMLButtonElement>('[data-upload-submit]');
    if (submit) setBusy(submit, true);
    dialog.close();

    void upload(chosen, {
      patientId,
      fieldOfView: String(data.get('fieldOfView') ?? 'both_jaws'),
      capturedOn: String(data.get('capturedOn') ?? ''),
    })
      .then((created) => {
        notify.success(
          `Анализ завершён: находок ${created.findingCount}. Открываем исследование…`,
        );
        location.assign(created.pageUrl);
      })
      .catch(notifyError)
      .finally(() => {
        if (submit) setBusy(submit, false);
        chosen = null;
      });
  });
}

async function fillPatients(): Promise<void> {
  const select = maybe<HTMLSelectElement>('[data-patient-select]');
  if (!select || select.options.length > 1) return;
  try {
    const page = await api.patients.list({ limit: 200 });
    for (const patient of page.items as readonly PatientSummary[]) {
      select.append(el('option', { value: String(patient.id) }, patient.fullName));
    }
  } catch (error) {
    notifyError(error);
  }
}

interface UploadDetails {
  readonly patientId: number;
  readonly fieldOfView: string;
  readonly capturedOn: string;
}

/**
 * POST the study with progress.
 *
 * `XMLHttpRequest` rather than `fetch`, purely for `upload.onprogress`: a
 * two-hundred-megabyte transfer needs a bar, and `fetch` still cannot provide
 * one. The CSRF header and the problem-document shape match the JSON client's
 * so the failure path behaves identically.
 */
function upload(file: File, details: UploadDetails): Promise<VolumeDetailResult> {
  const container = maybe('[data-upload-progress]');
  const bar = maybe('[data-progress-bar]');
  const label = maybe('[data-progress-label]');
  if (container) container.hidden = false;
  if (label) label.hidden = false;

  const body = new FormData();
  body.append('file', file);
  body.append('patient_id', String(details.patientId));
  body.append('field_of_view', details.fieldOfView);
  if (details.capturedOn) body.append('captured_on', details.capturedOn);

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', '/api/v1/volumes');
    request.withCredentials = true;
    const token =
      document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]')?.content ?? '';
    request.setRequestHeader('X-CSRF-Token', token);
    request.setRequestHeader('Accept', 'application/json');

    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const ratio = event.loaded / event.total;
      if (bar) bar.style.setProperty('inline-size', `${Math.round(ratio * 100)}%`);
      if (label) {
        label.textContent =
          ratio < 1
            ? `Загрузка ${Math.round(ratio * 100)}% · ${formatBytes(event.loaded)} из ${formatBytes(event.total)}`
            : 'Анализ на сервере…';
      }
    };

    request.onload = () => {
      if (container) container.hidden = true;
      if (label) label.hidden = true;
      if (bar) bar.style.setProperty('inline-size', '0%');
      if (request.status >= 200 && request.status < 300) {
        resolve(JSON.parse(request.responseText) as VolumeDetailResult);
        return;
      }
      reject(new Error(problemTitle(request.responseText)));
    };
    request.onerror = () => {
      if (container) container.hidden = true;
      if (label) label.hidden = true;
      reject(new Error('Не удалось связаться с сервером.'));
    };

    request.send(body);
  });
}

interface VolumeDetailResult {
  readonly pageUrl: string;
  readonly findingCount: number;
}

function problemTitle(raw: string): string {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null && 'title' in parsed) {
      return String((parsed as { title: unknown }).title);
    }
  } catch {
    // Fall through to the generic message below.
  }
  return 'Не удалось загрузить исследование.';
}

/** The newest CBCT, for the command palette's "open latest scan" action. */
export async function newestVolume(): Promise<VolumeListItem | null> {
  const page = await api.volumes.list({ limit: 1 });
  return page.items[0] ?? null;
}
