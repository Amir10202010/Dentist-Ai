/**
 * 3D scan page: metadata panel plus the WebGL viewer and its controls.
 */

import { api } from '../lib/api';
import { el, maybe, must, on, replaceChildren, setBusy } from '../lib/dom';
import { formatBytes, formatDate, formatDateTime, formatNumber } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { Scan } from '../lib/types';
import { mountMeshViewer, MeshViewerError, type MeshViewer } from './mesh-viewer';

function failure(message: string, detail: string): HTMLElement {
  return el(
    'div',
    { class: 'state state--error' },
    el('div', { class: 'state-icon' }, icon('alert', { class: 'icon--lg' })),
    el('p', { class: 'state-title' }, message),
    el('p', { class: 'state-body' }, detail),
  );
}

export function initScanDetail(publicId: string): void {
  const stage = must('[data-scan-stage]');
  const canvas = must<HTMLCanvasElement>('[data-mesh-canvas]');
  const loading = maybe('[data-mesh-loading]');
  const clipInput = maybe<HTMLInputElement>('[data-clip-plane]');
  const resetButton = maybe<HTMLButtonElement>('[data-reset-view]');
  const deleteButton = maybe<HTMLButtonElement>('[data-delete-scan]');

  let viewer: MeshViewer | null = null;

  function fillMetadata(scan: Scan): void {
    const [width, depth, height] = scan.bounds.size;
    const values: Record<string, string> = {
      title: scan.kindLabel,
      patient: scan.patientName ?? '—',
      arch: scan.archLabel,
      filename: scan.originalFilename,
      format: scan.sourceFormat.toUpperCase(),
      triangles: formatNumber(scan.triangleCount),
      size: formatBytes(scan.byteSize),
      dimensions: `${width} × ${depth} × ${height} мм`,
      captured: scan.capturedOn ? formatDate(scan.capturedOn) : '—',
      uploaded: formatDateTime(scan.createdAt),
      author: scan.uploadedByName ?? '—',
      notes: scan.notes ?? '—',
    };

    for (const [key, value] of Object.entries(values)) {
      const node = maybe(`[data-scan-${key}]`);
      if (!node) continue;
      node.textContent = value;
      node.classList.remove('skeleton', 'skeleton--title', 'skeleton--line', 'skeleton--text');
    }

    const patientLink = maybe<HTMLAnchorElement>('[data-scan-patient-link]');
    if (patientLink) patientLink.href = `/app/patients/${scan.patientId}`;

    const download = maybe<HTMLAnchorElement>('[data-scan-download]');
    if (download) download.href = scan.meshUrl;
  }

  async function load(): Promise<void> {
    let scan: Scan;
    try {
      scan = await api.scans.get(publicId);
    } catch (error) {
      notifyError(error);
      replaceChildren(stage, failure('Модель не найдена', 'Возможно, её удалили.'));
      return;
    }

    fillMetadata(scan);

    try {
      viewer = await mountMeshViewer(canvas, scan.meshUrl);
      loading?.remove();
    } catch (error) {
      loading?.remove();
      const detail =
        error instanceof MeshViewerError
          ? error.message
          : 'Попробуйте обновить страницу или открыть её в другом браузере.';
      replaceChildren(stage, failure('Не удалось показать модель', detail));
      return;
    }

    if (clipInput) {
      clipInput.disabled = false;
      on(clipInput, 'input', () => viewer?.setClip(Number(clipInput.value) / 100));
    }
    if (resetButton) {
      resetButton.disabled = false;
      on(resetButton, 'click', () => {
        viewer?.resetView();
        if (clipInput) clipInput.value = '100';
      });
    }
  }

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
        setBusy(confirmDelete, true);
        try {
          const scan = await api.scans.get(publicId);
          await api.scans.remove(publicId);
          notify.success('3D-модель удалена');
          location.assign(`/app/patients/${scan.patientId}`);
        } catch (error) {
          notifyError(error);
          setBusy(confirmDelete, false);
        }
      })();
    });
  }

  window.addEventListener('pagehide', () => viewer?.destroy(), { once: true });
  void load();
}
