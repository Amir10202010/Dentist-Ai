/** Account settings: profile, language, password. */

import { api } from '../lib/api';
import { maybe } from '../lib/dom';
import { enhanceForm, text } from '../lib/form';
import { notify } from '../lib/toast';

const MIN_PASSWORD_LENGTH = 10;

export function initSettingsPage(): void {
  const profileForm = maybe<HTMLFormElement>('[data-profile-form]');
  if (profileForm) {
    enhanceForm(profileForm, {
      validate: (values) =>
        text(values, 'fullName').length < 2 ? { fullName: 'Укажите имя' } : null,
      onSubmit: (values) =>
        api.settings.updateProfile({
          fullName: text(values, 'fullName'),
          locale: text(values, 'locale') || 'ru',
        }),
      onSuccess: (user) => {
        notify.success('Профиль обновлён');
        for (const node of document.querySelectorAll('[data-user-name]')) {
          node.textContent = user.fullName;
        }
        // The interface language is server-rendered, so a reload is the
        // honest way to apply it rather than half-translating in place.
        if (user.locale !== document.documentElement.lang) location.reload();
      },
    });
  }

  const passwordForm = maybe<HTMLFormElement>('[data-password-form]');
  if (passwordForm) {
    enhanceForm(passwordForm, {
      validate: (values) => {
        const next = text(values, 'newPassword');
        const confirm = text(values, 'newPasswordConfirm');
        if (next.length < MIN_PASSWORD_LENGTH) {
          return { newPassword: `Минимум ${MIN_PASSWORD_LENGTH} символов` };
        }
        if (next !== confirm) {
          return { newPasswordConfirm: 'Пароли не совпадают' };
        }
        return null;
      },
      resetOnSuccess: true,
      onSubmit: (values) =>
        api.settings.changePassword({
          currentPassword: text(values, 'currentPassword'),
          newPassword: text(values, 'newPassword'),
          newPasswordConfirm: text(values, 'newPasswordConfirm'),
        }),
      onSuccess: () => notify.success('Пароль изменён'),
    });
  }
}
