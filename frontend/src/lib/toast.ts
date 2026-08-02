/**
 * Toasts. Messages land in an `aria-live` region, so they are announced as
 * well as seen.
 */

import { ApiError, NetworkError } from './api';
import { el, on } from './dom';
import { icon, type IconName } from './icons';

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';

/** One glyph per variant, so the meaning survives without reading the text. */
const VARIANT_ICONS: Readonly<Record<ToastVariant, IconName>> = {
  info: 'alert-circle',
  success: 'check-circle',
  warning: 'alert',
  error: 'x-circle',
};

const DEFAULT_DURATION_MS = 4500;
/** Errors stay put: a message you can act on should not vanish while you read it. */
const ERROR_DURATION_MS = 9000;

let region: HTMLElement | null = null;

function ensureRegion(): HTMLElement {
  if (region?.isConnected) return region;
  region = el('div', {
    class: 'toast-region',
    id: 'toast-region',
    aria: { live: 'polite', atomic: 'false' },
  });
  document.body.append(region);
  return region;
}

export interface ToastOptions {
  readonly variant?: ToastVariant;
  readonly durationMs?: number;
  readonly action?: { readonly label: string; readonly onClick: () => void };
}

export function toast(message: string, options: ToastOptions = {}): void {
  const variant = options.variant ?? 'info';
  const duration =
    options.durationMs ?? (variant === 'error' ? ERROR_DURATION_MS : DEFAULT_DURATION_MS);

  const body = el('div', { class: 'toast-body' }, message);

  const close = el(
    'button',
    { class: 'toast-close', type: 'button', aria: { label: 'Закрыть уведомление' } },
    icon('close', { class: 'icon--sm' }),
  );

  const node = el(
    'div',
    { class: 'toast', role: 'status', dataset: { variant } },
    el('span', { class: 'toast-icon' }, icon(VARIANT_ICONS[variant])),
    body,
    options.action
      ? el(
          'button',
          { class: 'btn btn--sm btn--ghost', type: 'button', onclick: options.action.onClick },
          options.action.label,
        )
      : null,
    close,
  );

  let timer: number | undefined;

  const dismiss = (): void => {
    if (timer !== undefined) window.clearTimeout(timer);
    node.dataset['leaving'] = 'true';
    on(node, 'animationend', () => node.remove(), { once: true });
    // Belt and braces: if the animation never fires (reduced motion, hidden
    // tab), the node must still go away.
    window.setTimeout(() => node.remove(), 400);
  };

  on(close, 'click', dismiss);
  // Pause the countdown while the pointer rests on the toast, so a message
  // cannot expire out from under someone reading it.
  on(node, 'mouseenter', () => {
    if (timer !== undefined) window.clearTimeout(timer);
  });
  on(node, 'mouseleave', () => {
    timer = window.setTimeout(dismiss, 1500);
  });

  ensureRegion().append(node);
  timer = window.setTimeout(dismiss, duration);
}

type VariantOptions = Omit<ToastOptions, 'variant'>;

export const notify = {
  success: (message: string, options?: VariantOptions): void =>
    toast(message, { ...options, variant: 'success' }),
  error: (message: string, options?: VariantOptions): void =>
    toast(message, { ...options, variant: 'error' }),
  warning: (message: string, options?: VariantOptions): void =>
    toast(message, { ...options, variant: 'warning' }),
  info: (message: string, options?: VariantOptions): void =>
    toast(message, { ...options, variant: 'info' }),
} as const;

/**
 * Turn any thrown value into a message worth showing.
 *
 * Field-level validation errors are surfaced next to their inputs by the form
 * code, so only the summary is shown for those.
 */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof NetworkError) return error.message;
  if (error instanceof DOMException && error.name === 'AbortError') return '';
  return 'Что-то пошло не так. Попробуйте ещё раз.';
}

export function notifyError(error: unknown): void {
  const message = describeError(error);
  if (message) notify.error(message);
}
