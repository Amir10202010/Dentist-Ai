/**
 * Generated treatment plans, as a panel beside the case.
 *
 * The whole interface question here is how to present three options without
 * implying a ladder. They are not "good, better, best" — the conservative
 * option is the correct answer for a patient who wants the problem treated and
 * nothing else — so they render as equal cards side by side, with the same
 * weight and the same set of facts, rather than as a pricing table with one
 * highlighted.
 *
 * The other decision is that **nothing is scheduled by generating**. The panel
 * shows a draft until a clinician presses "accept" on one option, and only
 * then do steps appear in the patient's work. That mirrors the service, and it
 * is the line the product must not cross on its own.
 */

import { ApiError, api } from '../lib/api';
import { delegate, el, maybe, on, replaceChildren, setBusy } from '../lib/dom';
import { formatNumber, plural } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { TreatmentApproach, TreatmentOption, TreatmentPlan } from '../lib/types';

export interface PlanPanelOptions {
  readonly patientId: number;
  readonly volumePublicId?: string;
}

export interface PlanPanel {
  destroy(): void;
}

export function mountPlanPanel(root: HTMLElement, options: PlanPanelOptions): PlanPanel {
  const generate = maybe<HTMLButtonElement>('[data-generate-plan]', root);
  const container = maybe('[data-plan-proposal]', root);
  const empty = maybe('[data-plan-empty]', root);
  if (!generate || !container) return { destroy: () => undefined };

  const teardown: Array<() => void> = [];
  let plan: TreatmentPlan | null = null;

  teardown.push(
    on(generate, 'click', () => {
      setBusy(generate, true);
      void api.planning
        .generate({
          patientId: options.patientId,
          ...(options.volumePublicId ? { volumePublicId: options.volumePublicId } : {}),
        })
        .then((created) => {
          plan = created;
          render(created);
          notify.success('План сформирован. Выберите вариант, чтобы принять его.');
        })
        .catch((error: unknown) => {
          // A scan with nothing actionable on it is a legitimate answer, not a
          // failure, so it is stated rather than thrown at the user as an error.
          if (error instanceof ApiError && error.code === 'validation_failed') {
            if (empty) empty.textContent = error.message;
            return;
          }
          notifyError(error);
        })
        .finally(() => setBusy(generate, false));
    }),
  );

  teardown.push(
    delegate(container, 'click', '[data-accept-option]', (_event, target) => {
      const approach = target.dataset['acceptOption'] as TreatmentApproach | undefined;
      if (!approach || !plan) return;
      const button = target as HTMLButtonElement;
      setBusy(button, true);
      void api.planning
        .accept(plan.publicId, approach)
        .then((accepted) => {
          plan = accepted;
          render(accepted);
          notify.success(
            `Принят вариант «${
              accepted.options.find((item) => item.isSelected)?.approachLabel ?? ''
            }»: ${accepted.items.length} ${plural(
              accepted.items.length,
              'этап',
              'этапа',
              'этапов',
            )}.`,
          );
        })
        .catch(notifyError)
        .finally(() => setBusy(button, false));
    }),
  );

  function render(current: TreatmentPlan): void {
    if (empty) empty.hidden = true;
    container!.hidden = false;

    const accepted = current.items.length > 0;
    replaceChildren(
      container!,
      el(
        'div',
        { class: 'plan-head' },
        el('p', { class: 'plan-title' }, current.title),
        el(
          'div',
          { class: 'plan-tags' },
          el('span', { class: 'tag' }, current.statusLabel),
          current.complexityLabel ? el('span', { class: 'tag' }, current.complexityLabel) : null,
          current.estimatedWeeks !== null
            ? el(
                'span',
                { class: 'tag' },
                `${formatNumber(current.estimatedWeeks)} ${plural(
                  current.estimatedWeeks,
                  'неделя',
                  'недели',
                  'недель',
                )}`,
              )
            : null,
        ),
      ),
      current.rationale ? el('p', { class: 'plan-rationale' }, current.rationale) : null,
      accepted ? acceptedSteps(current) : optionGrid(current.options),
      current.risks ? riskList(current.risks) : null,
      current.followUp
        ? el(
            'p',
            { class: 'plan-followup' },
            icon('clock', { class: 'icon--sm' }),
            current.followUp,
          )
        : null,
    );
  }

  function optionGrid(items: readonly TreatmentOption[]): HTMLElement {
    return el(
      'div',
      { class: 'option-grid' },
      ...items.map((option) =>
        el(
          'article',
          { class: 'option-card', dataset: { approach: option.approach } },
          el('h3', { class: 'option-title' }, option.title),
          el('p', { class: 'option-summary' }, option.summary),
          el(
            'dl',
            { class: 'option-figures' },
            figure('Визитов', String(option.estimatedVisits)),
            figure('Кресло', `${Math.round(option.estimatedMinutes / 60)} ч`),
            figure(
              'Срок',
              `${option.estimatedWeeks} ${plural(
                option.estimatedWeeks,
                'нед.',
                'нед.',
                'нед.',
              )}`,
            ),
            figure('Сложность', option.complexityLabel),
          ),
          el(
            'p',
            { class: 'option-note option-note--plus' },
            icon('check', { class: 'icon--sm' }),
            option.benefits,
          ),
          el(
            'p',
            { class: 'option-note option-note--minus' },
            icon('alert', { class: 'icon--sm' }),
            option.risks,
          ),
          el(
            'button',
            {
              type: 'button',
              class: 'btn btn--sm btn--primary option-accept',
              dataset: { acceptOption: option.approach },
            },
            'Принять этот вариант',
          ),
        ),
      ),
    );
  }

  function acceptedSteps(current: TreatmentPlan): HTMLElement {
    return el(
      'ol',
      { class: 'plan-steps' },
      ...current.items.map((item) =>
        el(
          'li',
          { class: 'plan-step', dataset: { priority: item.priority } },
          el('span', { class: 'step-priority' }, item.priorityLabel),
          el(
            'span',
            { class: 'step-body' },
            item.procedureLabel,
            item.toothName ? el('span', { class: 'step-tooth' }, ` · ${item.toothName}`) : null,
          ),
          el('span', { class: 'step-estimate' }, `${item.estimatedMinutes} мин`),
        ),
      ),
    );
  }

  function riskList(risks: string): HTMLElement {
    const lines = risks.split('\n').filter((line) => line.trim().length > 0);
    return el(
      'ul',
      { class: 'plan-risks' },
      ...lines.map((line) =>
        el('li', {}, icon('alert', { class: 'icon--sm' }), line),
      ),
    );
  }

  function figure(label: string, value: string): DocumentFragment {
    const fragment = document.createDocumentFragment();
    fragment.append(el('dt', {}, label), el('dd', {}, value));
    return fragment;
  }

  return {
    destroy(): void {
      for (const off of teardown) off();
    },
  };
}
