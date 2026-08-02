/**
 * The tooth chart.
 *
 * Thirty-two buttons in the order a clinician reads them: upper arch left to
 * right as the viewer sees it, lower arch beneath. The server sends the cells
 * already in that order, so this module only has to split them into two rows.
 */

import { el, replaceChildren } from '../lib/dom';
import { icon } from '../lib/icons';
import type { ToothCell } from '../lib/types';

const TEETH_PER_ROW = 16;

export interface OdontogramOptions {
  readonly selected?: number | null;
  readonly onSelect?: (toothNumber: number) => void;
}

function toothButton(cell: ToothCell, options: OdontogramOptions): HTMLElement {
  const state = cell.isMissing
    ? 'missing'
    : cell.severity
      ? 'finding'
      : cell.hasRestoration
        ? 'restored'
        : 'sound';

  const description = [
    `Зуб ${cell.toothNumber}`,
    cell.isMissing ? 'отсутствует' : null,
    cell.findingCount > 0 ? `находок: ${cell.findingCount}` : 'без находок',
  ]
    .filter(Boolean)
    .join(', ');

  return el(
    'button',
    {
      type: 'button',
      class: 'tooth',
      disabled: cell.findingCount === 0,
      dataset: {
        tooth: String(cell.toothNumber),
        state,
        ...(cell.severity ? { severity: cell.severity } : {}),
        selected: String(options.selected === cell.toothNumber),
      },
      aria: { label: description },
      title: description,
    },
    el('span', { class: 'tooth-glyph', aria: { hidden: 'true' } }, icon('tooth')),
    el('span', { class: 'tooth-number' }, String(cell.toothNumber)),
    cell.findingCount > 1 &&
      el('span', { class: 'tooth-count', aria: { hidden: 'true' } }, String(cell.findingCount)),
  );
}

export function renderOdontogram(
  host: HTMLElement,
  cells: readonly ToothCell[],
  options: OdontogramOptions = {},
): void {
  const upper = cells.slice(0, TEETH_PER_ROW);
  const lower = cells.slice(TEETH_PER_ROW);

  replaceChildren(
    host,
    el(
      'div',
      { class: 'odontogram-row', dataset: { arch: 'upper' } },
      ...upper.map((cell) => toothButton(cell, options)),
    ),
    el(
      'div',
      { class: 'odontogram-row', dataset: { arch: 'lower' } },
      ...lower.map((cell) => toothButton(cell, options)),
    ),
  );
}

export function highlightTooth(host: HTMLElement, toothNumber: number | null): void {
  for (const node of host.querySelectorAll<HTMLElement>('.tooth')) {
    node.dataset['selected'] = String(node.dataset['tooth'] === String(toothNumber));
  }
}
