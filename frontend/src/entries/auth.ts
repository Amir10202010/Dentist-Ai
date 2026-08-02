/** Login and registration screens. */

import '../styles/tokens.css';
import '../styles/base.css';
import '../styles/components.css';
// No `marketing.css`: the auth screens render neither the site header nor the
// footer, and the brand mark and eyebrow live in `components.css`.
import '../styles/auth.css';

import { api } from '../lib/api';
import { maybe, on } from '../lib/dom';
import { enhanceForm, text } from '../lib/form';
import { initTheme } from '../lib/theme';

const MIN_PASSWORD_LENGTH = 10;

/** Where to land after signing in, from `?next=` — same-origin paths only. */
function nextPath(): string {
  const requested = new URLSearchParams(location.search).get('next');
  // An open redirect here would let a phishing link bounce a freshly
  // authenticated user to an attacker's page.
  if (requested && requested.startsWith('/') && !requested.startsWith('//')) {
    return requested;
  }
  return '/app';
}

function initPasswordToggles(): void {
  for (const toggle of document.querySelectorAll<HTMLButtonElement>('[data-toggle-password]')) {
    on(toggle, 'click', () => {
      const selector = toggle.dataset['togglePassword'];
      if (!selector) return;
      const input = document.querySelector<HTMLInputElement>(selector);
      if (!input) return;
      const revealed = input.type === 'text';
      input.type = revealed ? 'password' : 'text';
      toggle.setAttribute('aria-pressed', String(!revealed));
      toggle.setAttribute('aria-label', revealed ? 'Показать пароль' : 'Скрыть пароль');
    });
  }
}

/** Live strength meter — feedback while typing beats rejection on submit. */
function initPasswordStrength(): void {
  const input = maybe<HTMLInputElement>('[data-password-strength-input]');
  const meter = maybe('[data-password-strength]');
  const label = maybe('[data-password-strength-label]');
  if (!input || !meter) return;

  on(input, 'input', () => {
    const value = input.value;
    const checks = [
      value.length >= MIN_PASSWORD_LENGTH,
      value.length >= 14,
      /[a-zа-я]/i.test(value) && /\d/.test(value),
      new Set(value).size >= 8,
    ];
    const score = checks.filter(Boolean).length;
    const percent = value.length === 0 ? 0 : Math.max(12, (score / checks.length) * 100);

    meter.setAttribute('style', `inline-size:${percent}%`);
    meter.dataset['level'] = String(score);
    if (label) {
      label.textContent =
        value.length === 0
          ? ''
          : (['Слабый', 'Слабый', 'Средний', 'Хороший', 'Отличный'][score] ?? '');
    }
  });
}

function initLogin(): void {
  const form = maybe<HTMLFormElement>('[data-login-form]');
  if (!form) return;

  enhanceForm(form, {
    validate: (values) => {
      const errors: Record<string, string> = {};
      if (!text(values, 'email').includes('@')) errors['email'] = 'Введите корректный email';
      if (!text(values, 'password')) errors['password'] = 'Введите пароль';
      return Object.keys(errors).length > 0 ? errors : null;
    },
    onSubmit: (values) =>
      api.auth.login({
        email: text(values, 'email'),
        password: text(values, 'password'),
      }),
    onSuccess: () => location.assign(nextPath()),
  });
}

function initRegister(): void {
  const form = maybe<HTMLFormElement>('[data-register-form]');
  if (!form) return;

  enhanceForm(form, {
    validate: (values) => {
      const errors: Record<string, string> = {};
      if (text(values, 'fullName').length < 2) errors['fullName'] = 'Укажите ваше имя';
      if (text(values, 'organizationName').length < 2) {
        errors['organizationName'] = 'Укажите название клиники';
      }
      if (!text(values, 'email').includes('@')) errors['email'] = 'Введите корректный email';
      if (text(values, 'password').length < MIN_PASSWORD_LENGTH) {
        errors['password'] = `Минимум ${MIN_PASSWORD_LENGTH} символов`;
      }
      if (text(values, 'password') !== text(values, 'passwordConfirm')) {
        errors['passwordConfirm'] = 'Пароли не совпадают';
      }
      return Object.keys(errors).length > 0 ? errors : null;
    },
    onSubmit: (values) =>
      api.auth.register({
        fullName: text(values, 'fullName'),
        email: text(values, 'email'),
        organizationName: text(values, 'organizationName'),
        password: text(values, 'password'),
        passwordConfirm: text(values, 'passwordConfirm'),
      }),
    // Registration signs you in, so go straight to the product.
    onSuccess: () => location.assign('/app'),
  });
}

function boot(): void {
  initTheme();
  initPasswordToggles();
  initPasswordStrength();
  initLogin();
  initRegister();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
