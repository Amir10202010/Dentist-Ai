/**
 * Progressive form enhancement.
 *
 * Wires a `<form>` to a typed submit handler with loading state, inline field
 * errors from the server's problem document, and focus management. The old
 * code duplicated this per page and lost server-side field errors entirely.
 */

import { ApiError } from './api';
import { all, maybe, must, on, setBusy } from './dom';
import { notifyError } from './toast';

export interface FormController {
  readonly form: HTMLFormElement;
  setFieldError(field: string, message: string | null): void;
  clearErrors(): void;
}

export interface EnhanceOptions<T> {
  /** Runs on submit. Throwing an `ApiError` populates inline field errors. */
  readonly onSubmit: (values: FormData, controller: FormController) => Promise<T>;
  readonly onSuccess?: (result: T, controller: FormController) => void;
  /** Client-side checks; return field -> message for anything invalid. */
  readonly validate?: (values: FormData) => Readonly<Record<string, string>> | null;
  readonly resetOnSuccess?: boolean;
}

function errorSlot(form: HTMLFormElement, field: string): HTMLElement | null {
  return maybe(`[data-error-for="${CSS.escape(field)}"]`, form);
}

function inputFor(form: HTMLFormElement, field: string): HTMLElement | null {
  return maybe(`[name="${CSS.escape(field)}"]`, form);
}

export function enhanceForm<T>(
  formOrSelector: HTMLFormElement | string,
  options: EnhanceOptions<T>,
): FormController {
  const form =
    typeof formOrSelector === 'string'
      ? must<HTMLFormElement>(formOrSelector)
      : formOrSelector;

  const submitButton = maybe<HTMLButtonElement>('button[type="submit"]', form);

  const setFieldError = (field: string, message: string | null): void => {
    const slot = errorSlot(form, field);
    const input = inputFor(form, field);
    if (slot) slot.textContent = message ?? '';
    if (input) {
      input.setAttribute('aria-invalid', message ? 'true' : 'false');
      if (message && slot?.id) input.setAttribute('aria-describedby', slot.id);
    }
  };

  const clearErrors = (): void => {
    for (const slot of all('[data-error-for]', form)) {
      slot.textContent = '';
    }
    for (const input of all('[name]', form)) {
      input.setAttribute('aria-invalid', 'false');
    }
  };

  const controller: FormController = { form, setFieldError, clearErrors };

  // Clear a field's error as soon as the user starts fixing it — leaving it
  // visible while they type reads as the app not noticing.
  on(form, 'input', (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.getAttribute('aria-invalid') === 'true') {
      const name = target.getAttribute('name');
      if (name) setFieldError(name, null);
    }
  });

  on(form, 'submit', (event) => {
    event.preventDefault();
    void submit();
  });

  async function submit(): Promise<void> {
    clearErrors();
    const values = new FormData(form);

    const clientErrors = options.validate?.(values);
    if (clientErrors && Object.keys(clientErrors).length > 0) {
      applyErrors(clientErrors);
      return;
    }

    if (submitButton) setBusy(submitButton, true);
    try {
      const result = await options.onSubmit(values, controller);
      if (options.resetOnSuccess) form.reset();
      options.onSuccess?.(result, controller);
    } catch (error) {
      if (error instanceof ApiError && Object.keys(error.fieldErrors).length > 0) {
        applyErrors(error.fieldErrors);
        // Field errors are shown inline; a toast on top would be noise.
        if (error.code !== 'validation_failed') notifyError(error);
      } else {
        notifyError(error);
      }
    } finally {
      if (submitButton) setBusy(submitButton, false);
    }
  }

  function applyErrors(errors: Readonly<Record<string, string>>): void {
    let firstInvalid: HTMLElement | null = null;
    for (const [field, message] of Object.entries(errors)) {
      setFieldError(field, message);
      firstInvalid ??= inputFor(form, field);
    }
    // Move focus to the first problem so keyboard and screen-reader users are
    // taken straight to it rather than left to hunt.
    firstInvalid?.focus();
  }

  return controller;
}

/** Read a trimmed string field. */
export function text(values: FormData, key: string): string {
  const value = values.get(key);
  return typeof value === 'string' ? value.trim() : '';
}

/** Read an optional string field, collapsing blanks to null. */
export function optionalText(values: FormData, key: string): string | null {
  return text(values, key) || null;
}
