/**
 * Treatment plan panel.
 *
 * Steps arrive already ordered by priority. Status is the only field edited
 * inline, because it is the one that changes every visit; everything else
 * lives behind an explicit action.
 */

import { api } from '../lib/api';
import { delegate, el, replaceChildren, setBusy } from '../lib/dom';
import { formatDate, plural } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { PlanItem, PlanItemStatus, ProcedureOption, TreatmentPlan } from '../lib/types';

const STATUS_FLOW: readonly PlanItemStatus[] = [
  'proposed',
  'accepted',
  'scheduled',
  'in_progress',
  'done',
  'declined',
];

const STATUS_LABELS: Readonly<Record<PlanItemStatus, string>> = {
  proposed: 'Предложено',
  accepted: 'Согласовано',
  scheduled: 'Записан',
  in_progress: 'В процессе',
  done: 'Выполнено',
  declined: 'Отказ пациента',
};

export interface PlanPanelHandlers {
  /** Re-fetch and re-render after any mutation the panel performed. */
  readonly onChanged: () => void;
}

function itemRow(item: PlanItem): HTMLElement {
  const statusSelect = el(
    'select',
    {
      class: 'select select--sm plan-status',
      dataset: { action: 'status', itemId: String(item.id) },
      aria: { label: `Статус этапа: ${item.procedureLabel}` },
    },
    ...STATUS_FLOW.map((status) =>
      el('option', { value: status, selected: status === item.status }, STATUS_LABELS[status]),
    ),
  );

  return el(
    'li',
    {
      class: 'plan-item',
      dataset: {
        priority: item.priority,
        status: item.status,
        itemId: String(item.id),
        // Kept on the row so a status change can round-trip the rest of the
        // record unchanged — the endpoint takes a whole item, not a patch.
        visits: String(item.estimatedVisits),
        minutes: String(item.estimatedMinutes),
        scheduled: item.scheduledFor ?? '',
      },
    },
    el('span', { class: 'plan-priority-bar', aria: { hidden: 'true' } }),
    el(
      'div',
      { class: 'plan-item-body' },
      el(
        'div',
        { class: 'plan-item-head' },
        item.toothNumber !== null &&
          el(
            'span',
            { class: 'tooth-chip', title: item.toothName ?? '' },
            String(item.toothNumber),
          ),
        el('span', { class: 'plan-item-label' }, item.procedureLabel),
      ),
      el(
        'div',
        { class: 'plan-item-meta' },
        el('span', { class: 'badge badge--soft' }, item.categoryLabel),
        el(
          'span',
          { class: 'badge badge--priority', dataset: { priority: item.priority } },
          item.priorityLabel,
        ),
        el(
          'span',
          { class: 'plan-item-estimate' },
          `${item.estimatedVisits} ${plural(item.estimatedVisits, 'визит', 'визита', 'визитов')}`,
          ' · ',
          `${item.estimatedMinutes} мин`,
        ),
        item.scheduledFor && el('span', { class: 'plan-item-date' }, formatDate(item.scheduledFor)),
        item.sourceStudyPublicId &&
          el(
            'a',
            {
              class: 'plan-item-source',
              href: `/app/studies/${item.sourceStudyPublicId}`,
            },
            icon('scan', { class: 'icon--sm' }),
            'Из снимка',
          ),
      ),
    ),
    el(
      'div',
      { class: 'plan-item-actions' },
      statusSelect,
      el(
        'button',
        {
          type: 'button',
          class: 'btn btn--sm btn--icon btn--ghost',
          dataset: { action: 'remove', itemId: String(item.id) },
          title: 'Убрать этап',
          aria: { label: `Убрать этап: ${item.procedureLabel}` },
        },
        icon('trash', { class: 'icon--sm' }),
      ),
    ),
  );
}

function planCard(plan: TreatmentPlan, procedures: readonly ProcedureOption[]): HTMLElement {
  const hours = Math.round(plan.totalMinutes / 6) / 10;

  return el(
    'section',
    { class: 'card plan-card', dataset: { planId: plan.publicId } },
    el(
      'div',
      { class: 'between plan-card-head' },
      el(
        'div',
        {},
        el('h2', { class: 'panel-title' }, plan.title),
        el(
          'p',
          { class: 'panel-subtitle' },
          `${plan.doneCount} из ${plan.items.length} выполнено`,
          plan.totalVisits > 0
            ? ` · осталось ${plan.totalVisits} ${plural(plan.totalVisits, 'визит', 'визита', 'визитов')} ≈ ${hours} ч`
            : '',
        ),
      ),
      el('span', { class: 'badge badge--plan', dataset: { status: plan.status } }, plan.statusLabel),
    ),

    plan.items.length === 0
      ? el(
          'p',
          { class: 'panel-subtitle plan-empty' },
          'В плане пока нет этапов. Добавьте их вручную или составьте черновик по снимку.',
        )
      : el('ul', { class: 'plan-items' }, ...plan.items.map(itemRow)),

    el(
      'form',
      { class: 'plan-add', dataset: { addItem: plan.publicId } },
      el(
        'select',
        { class: 'select select--sm', name: 'procedureCode', aria: { label: 'Процедура' } },
        el('option', { value: '' }, 'Добавить этап…'),
        ...procedures.map((procedure) =>
          el('option', { value: procedure.code }, `${procedure.label} · ${procedure.categoryLabel}`),
        ),
      ),
      el('input', {
        class: 'input input--sm plan-tooth-input',
        name: 'toothNumber',
        type: 'number',
        min: '11',
        max: '48',
        placeholder: 'Зуб',
        aria: { label: 'Номер зуба по FDI' },
      }),
      el(
        'button',
        { type: 'submit', class: 'btn btn--sm' },
        icon('plus', { class: 'icon--sm' }),
        'Добавить',
      ),
    ),
  );
}

export function renderPlans(
  host: HTMLElement,
  plans: readonly TreatmentPlan[],
  procedures: readonly ProcedureOption[],
): void {
  if (plans.length === 0) {
    replaceChildren(
      host,
      el(
        'div',
        { class: 'state' },
        el('div', { class: 'state-icon' }, icon('clipboard', { class: 'icon--lg' })),
        el('p', { class: 'state-title' }, 'Плана лечения пока нет'),
        el(
          'p',
          { class: 'state-body' },
          'Составьте черновик по последнему снимку — предложения берутся из таблицы протоколов, ' +
            'а решение остаётся за вами.',
        ),
      ),
    );
    return;
  }

  replaceChildren(host, ...plans.map((plan) => planCard(plan, procedures)));
}

export function bindPlanActions(host: HTMLElement, handlers: PlanPanelHandlers): void {
  function planIdOf(node: HTMLElement): string | null {
    return node.closest<HTMLElement>('.plan-card')?.dataset['planId'] ?? null;
  }

  delegate(host, 'change', '[data-action="status"]', (_event, target) => {
    const select = target as HTMLSelectElement;
    const planId = planIdOf(select);
    const row = select.closest<HTMLElement>('.plan-item');
    const itemId = Number(select.dataset['itemId']);
    if (!planId || !row || !itemId) return;

    void (async (): Promise<void> => {
      try {
        await api.treatment.updateItem(planId, itemId, {
          status: select.value as PlanItemStatus,
          toothNumber: readTooth(row),
          scheduledFor: row.dataset['scheduled'] || null,
          estimatedVisits: readNumber(row, 'visits', 1),
          estimatedMinutes: readNumber(row, 'minutes', 60),
          notes: null,
        });
        handlers.onChanged();
      } catch (error) {
        notifyError(error);
      }
    })();
  });

  delegate(host, 'click', '[data-action="remove"]', (_event, target) => {
    const button = target as HTMLButtonElement;
    const planId = planIdOf(button);
    const itemId = Number(button.dataset['itemId']);
    if (!planId || !itemId) return;

    void (async (): Promise<void> => {
      setBusy(button, true);
      try {
        await api.treatment.removeItem(planId, itemId);
        notify.success('Этап убран из плана');
        handlers.onChanged();
      } catch (error) {
        notifyError(error);
        setBusy(button, false);
      }
    })();
  });

  delegate(host, 'submit', '[data-add-item]', (event, target) => {
    event.preventDefault();
    const form = target as HTMLFormElement;
    const planId = form.dataset['addItem'];
    const values = new FormData(form);
    const code = String(values.get('procedureCode') ?? '');
    if (!planId || !code) return;

    const rawTooth = String(values.get('toothNumber') ?? '').trim();
    void (async (): Promise<void> => {
      try {
        await api.treatment.addItem(planId, {
          procedureCode: code,
          toothNumber: rawTooth ? Number(rawTooth) : null,
        });
        notify.success('Этап добавлен');
        handlers.onChanged();
      } catch (error) {
        notifyError(error);
      }
    })();
  });
}

function readTooth(row: HTMLElement): number | null {
  const chip = row.querySelector('.tooth-chip')?.textContent?.trim();
  return chip ? Number(chip) : null;
}

function readNumber(row: HTMLElement, key: string, fallback: number): number {
  const raw = row.dataset[key];
  const value = raw ? Number(raw) : Number.NaN;
  return Number.isFinite(value) ? value : fallback;
}
