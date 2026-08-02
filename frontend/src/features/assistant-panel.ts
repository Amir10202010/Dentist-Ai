/**
 * The case assistant, as a panel beside the viewer.
 *
 * Two interface decisions carry most of the weight here, and both come from
 * what the assistant actually is — a retrieval engine over one case, not a
 * language model.
 *
 * **Citations are rendered, not hidden behind a disclosure.** Every answer
 * names the records it was built from, and that is the property that makes it
 * usable on patient data. Tucking them away would leave a confident block of
 * text with no visible provenance, which is exactly the failure mode the
 * server-side design avoids.
 *
 * **Suggested questions are always visible.** The assistant answers a fixed
 * vocabulary and says so; showing the vocabulary is more honest than an empty
 * input that invites a question it will decline. An unrecognised question is
 * styled differently rather than dressed up as an answer.
 *
 * There is no streaming and no typing animation. The answer is composed from
 * rows in single-digit milliseconds, and pretending otherwise would be
 * theatre that implies a model is thinking.
 */

import { api } from '../lib/api';
import { delegate, el, maybe, on, replaceChildren, setBusy } from '../lib/dom';
import { icon } from '../lib/icons';
import { notifyError } from '../lib/toast';
import type { AssistantAnswer, Citation } from '../lib/types';

export interface AssistantPanelOptions {
  /** The case the questions are about. One of these is required. */
  readonly volumePublicId?: string;
  readonly patientId?: number;
}

export interface AssistantPanel {
  /** Ask on the user's behalf — used by the finding list's "why?" buttons. */
  ask(question: string): void;
  destroy(): void;
}

/** Intents whose answer is an admission rather than a result. */
const UNANSWERED = new Set(['capabilities']);

export function mountAssistantPanel(
  root: HTMLElement,
  options: AssistantPanelOptions,
): AssistantPanel {
  const log = maybe('[data-assistant-log]', root);
  const form = maybe<HTMLFormElement>('[data-assistant-form]', root);
  const input = maybe<HTMLInputElement>('[data-assistant-input]', root);
  const send = maybe<HTMLButtonElement>('[data-assistant-send]', root);
  const suggestionBar = maybe('[data-assistant-suggestions]', root);
  if (!log || !form || !input) {
    return { ask: () => undefined, destroy: () => undefined };
  }

  const teardown: Array<() => void> = [];
  let threadPublicId: string | null = null;
  let pending = false;

  function appendQuestion(text: string): void {
    log!.append(
      el('li', { class: 'chat-turn chat-turn--user' }, el('p', { class: 'chat-body' }, text)),
    );
    log!.scrollTop = log!.scrollHeight;
  }

  function appendAnswer(answer: AssistantAnswer): void {
    const paragraphs = answer.body.split('\n').filter((line) => line.trim().length > 0);
    log!.append(
      el(
        'li',
        {
          class: 'chat-turn chat-turn--assistant',
          dataset: { unanswered: String(UNANSWERED.has(answer.intent)) },
        },
        el(
          'div',
          { class: 'chat-avatar', aria: { hidden: 'true' } },
          icon('sparkles', { class: 'icon--sm' }),
        ),
        el(
          'div',
          { class: 'chat-content' },
          ...paragraphs.map((line) =>
            el(
              'p',
              { class: line.startsWith('•') ? 'chat-body chat-body--item' : 'chat-body' },
              line.replace(/^•\s*/, ''),
            ),
          ),
          answer.citations.length > 0 ? citationList(answer.citations) : null,
        ),
      ),
    );
    log!.scrollTop = log!.scrollHeight;
    renderSuggestions(answer.suggestions);
  }

  function citationList(citations: readonly Citation[]): HTMLElement {
    return el(
      'ul',
      { class: 'chat-citations', aria: { label: 'Источники ответа' } },
      ...citations.map((item) =>
        el(
          'li',
          {},
          item.href
            ? el('a', { class: 'chat-citation', href: item.href }, item.label)
            : el('span', { class: 'chat-citation' }, item.label),
        ),
      ),
    );
  }

  function renderSuggestions(suggestions: readonly string[]): void {
    if (!suggestionBar) return;
    replaceChildren(
      suggestionBar,
      ...suggestions.map((text) =>
        el('button', { type: 'button', class: 'suggestion', dataset: { suggest: text } }, text),
      ),
    );
  }

  function submit(question: string): void {
    const text = question.trim();
    if (!text || pending) return;

    pending = true;
    if (send) setBusy(send, true);
    appendQuestion(text);
    input!.value = '';

    const thinking = el(
      'li',
      { class: 'chat-turn chat-turn--assistant chat-turn--pending' },
      el('span', { class: 'spinner', aria: { hidden: 'true' } }),
    );
    log!.append(thinking);
    log!.scrollTop = log!.scrollHeight;

    void api.assistant
      .ask({
        question: text,
        ...(threadPublicId ? { threadPublicId } : {}),
        ...(options.volumePublicId ? { volumePublicId: options.volumePublicId } : {}),
        ...(options.patientId !== undefined ? { patientId: options.patientId } : {}),
      })
      .then((result) => {
        threadPublicId = result.threadPublicId;
        thinking.remove();
        appendAnswer(result.answer);
      })
      .catch((error: unknown) => {
        thinking.remove();
        notifyError(error);
      })
      .finally(() => {
        pending = false;
        if (send) setBusy(send, false);
        input!.focus();
      });
  }

  teardown.push(
    on(form, 'submit', (event) => {
      event.preventDefault();
      submit(input.value);
    }),
  );

  if (suggestionBar) {
    teardown.push(
      delegate(suggestionBar, 'click', '[data-suggest]', (_event, target) => {
        const question = target.dataset['suggest'];
        if (question) submit(question);
      }),
    );
  }

  // The opening state is the vocabulary, not an empty box: a clinician should
  // be able to see what is answerable before typing anything.
  renderSuggestions([
    'Кратко опиши этот снимок',
    'Почему AI это нашёл?',
    'Какие возможны варианты лечения?',
    'Что проверить на приёме?',
    'Объясни простыми словами для пациента',
  ]);

  return {
    ask: submit,
    destroy(): void {
      for (const off of teardown) off();
    },
  };
}
