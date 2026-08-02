/**
 * Radiograph upload with drag-and-drop.
 *
 * Validates locally before touching the network so an obvious mistake costs
 * zero seconds, then shows honest progress: inference is genuinely slow, and
 * a spinner with no explanation reads as a hang.
 */

import { api } from '../lib/api';
import { el, must, on } from '../lib/dom';
import { formatBytes } from '../lib/format';
import { notify, notifyError } from '../lib/toast';
import type { Study } from '../lib/types';

const ACCEPTED_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
  'image/tiff',
]);

const MAX_BYTES = 24 * 1024 * 1024;

export interface UploadOptions {
  readonly dropzone: HTMLElement;
  readonly input: HTMLInputElement;
  readonly onUploaded: (study: Study) => void;
  readonly patientId?: number;
}

function validate(file: File): string | null {
  if (!ACCEPTED_TYPES.has(file.type)) {
    return 'Поддерживаются JPEG, PNG, WebP, BMP и TIFF.';
  }
  if (file.size > MAX_BYTES) {
    return `Файл ${formatBytes(file.size)} — максимум ${formatBytes(MAX_BYTES)}.`;
  }
  if (file.size === 0) return 'Файл пуст.';
  return null;
}

export function initUpload(options: UploadOptions): void {
  const { dropzone, input, onUploaded } = options;
  let busy = false;

  const setState = (state: 'idle' | 'dragging' | 'uploading'): void => {
    dropzone.dataset['state'] = state;
    dropzone.setAttribute('aria-busy', String(state === 'uploading'));
  };

  const preview = el('div', { class: 'dropzone-preview', hidden: true });
  dropzone.append(preview);

  async function handle(files: FileList | null): Promise<void> {
    const file = files?.[0];
    if (!file || busy) return;

    const problem = validate(file);
    if (problem) {
      notify.error(problem);
      return;
    }

    busy = true;
    setState('uploading');

    // A local object URL gives instant visual feedback while the server
    // works, instead of an empty box for several seconds.
    const objectUrl = URL.createObjectURL(file);
    preview.replaceChildren(
      el('img', { src: objectUrl, alt: '', class: 'dropzone-thumb' }),
      el(
        'div',
        { class: 'dropzone-progress' },
        el('span', { class: 'dropzone-filename' }, file.name),
        el('span', { class: 'dropzone-status' }, 'Анализируем снимок…'),
        el('div', { class: 'progress-bar progress-bar--inline' }),
      ),
    );
    preview.hidden = false;

    try {
      const study = await api.studies.upload(
        file,
        options.patientId === undefined ? undefined : options.patientId,
      );
      notify.success(
        study.attentionCount > 0
          ? `Найдено находок: ${study.findingCount}, требуют внимания: ${study.attentionCount}`
          : `Анализ завершён: находок ${study.findingCount}`,
      );
      onUploaded(study);
    } catch (error) {
      notifyError(error);
    } finally {
      URL.revokeObjectURL(objectUrl);
      preview.hidden = true;
      preview.replaceChildren();
      input.value = '';
      busy = false;
      setState('idle');
    }
  }

  on(input, 'change', () => void handle(input.files));

  // Clicking anywhere on the zone opens the picker, but not when the click
  // originated on the label's own <input> (which would double-open it).
  on(dropzone, 'click', (event) => {
    if (event.target !== input && !busy) input.click();
  });

  on(dropzone, 'keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (!busy) input.click();
    }
  });

  for (const type of ['dragenter', 'dragover'] as const) {
    on(dropzone, type, (event) => {
      event.preventDefault();
      if (!busy) setState('dragging');
    });
  }

  for (const type of ['dragleave', 'drop'] as const) {
    on(dropzone, type, (event) => {
      event.preventDefault();
      if (!busy) setState('idle');
    });
  }

  on(dropzone, 'drop', (event) => {
    event.preventDefault();
    void handle(event.dataTransfer?.files ?? null);
  });

  setState('idle');
}

/** Convenience wrapper for pages with the standard dropzone markup. */
export function mountUpload(onUploaded: (study: Study) => void): void {
  const dropzone = document.querySelector<HTMLElement>('[data-dropzone]');
  if (!dropzone) return;
  initUpload({
    dropzone,
    input: must<HTMLInputElement>('[data-upload-input]', dropzone),
    onUploaded,
  });
}
