/**
 * Patients: searchable table, create/edit dialog, archive with undo.
 *
 * The dialog is a native `<dialog>`: focus trapping, Esc to close and
 * background inertness come for free.
 */

import { ApiError, api } from '../lib/api';
import { debounce, delegate, el, maybe, must, on, replaceChildren } from '../lib/dom';
import { enhanceForm, optionalText, text } from '../lib/form';
import { formatRelative } from '../lib/format';
import { icon } from '../lib/icons';
import { notify, notifyError } from '../lib/toast';
import type { PatientSummary } from '../lib/types';

const PAGE_SIZE = 25;
/** Keep in sync with the `<thead>` in `app/patients.html`. */
const TABLE_COLUMNS = 8;

export function initPatientsPage(): void {
  const tbody = must('[data-patients-body]');
  const searchInput = maybe<HTMLInputElement>('[data-patients-search]');
  const dialog = maybe<HTMLDialogElement>('[data-patient-dialog]');
  const addButton = maybe<HTMLButtonElement>('[data-add-patient]');
  const summary = maybe('[data-patients-summary]');

  let query = '';
  let editingId: number | null = null;
  let inFlight: AbortController | null = null;

  function row(patient: PatientSummary): HTMLElement {
    return el(
      'tr',
      { dataset: { patientId: String(patient.id) } },
      el(
        'td',
        {},
        el(
          'div',
          { class: 'patient-cell' },
          el('span', { class: 'avatar avatar--sm' }, initials(patient.fullName)),
          el(
            'div',
            {},
            el(
              'a',
              { class: 'patient-name patient-link', href: `/app/patients/${patient.id}` },
              patient.fullName,
            ),
            patient.medicalRecordNumber
              ? el('div', { class: 'patient-mrn' }, `Карта №${patient.medicalRecordNumber}`)
              : null,
          ),
        ),
      ),
      el('td', {}, patient.phone ?? '—'),
      el('td', {}, patient.age === null ? '—' : `${patient.age}`),
      el('td', {}, String(patient.studyCount)),
      el(
        'td',
        {},
        patient.scanCount === 0 ? '—' : String(patient.scanCount),
      ),
      el(
        'td',
        {},
        patient.openPlanItems === 0
          ? '—'
          : el('span', { class: 'badge badge--soft' }, String(patient.openPlanItems)),
      ),
      el(
        'td',
        {},
        patient.lastStudyAt ? formatRelative(patient.lastStudyAt) : '—',
      ),
      el(
        'td',
        {},
        /*
         * The flex container must be a child of the cell, never the cell
         * itself. `display: flex` on a `<td>` takes it out of the table
         * formatting context: it stops being a row cell, so it ignores
         * `vertical-align`, sizes to its own content instead of the row, and
         * renders visibly offset from the data it belongs to.
         */
        el(
          'div',
          { class: 'table-actions' },
          el(
            'a',
            {
              class: 'btn btn--sm',
              href: `/app/studies?patient=${patient.id}`,
              title: 'Снимки пациента',
            },
            icon('scan', { class: 'icon--sm' }),
            'Снимки',
          ),
          iconButton('pencil', 'Изменить', 'edit'),
          iconButton('archive', 'В архив', 'archive'),
        ),
      ),
    );
  }

  /** Square icon-only action with the label carried by `aria-label`/`title`. */
  function iconButton(
    name: Parameters<typeof icon>[0],
    label: string,
    action: string,
  ): HTMLElement {
    return el(
      'button',
      {
        class: 'btn btn--sm btn--icon btn--ghost',
        type: 'button',
        title: label,
        dataset: { action },
        aria: { label },
      },
      icon(name, { class: 'icon--sm' }),
    );
  }

  function initials(name: string): string {
    return name
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part.charAt(0).toUpperCase())
      .join('');
  }

  function skeletonRows(): readonly HTMLElement[] {
    return Array.from({ length: 5 }, () =>
      el(
        'tr',
        {},
        ...Array.from({ length: 8 }, () =>
          el('td', {}, el('div', { class: 'skeleton skeleton--text' })),
        ),
      ),
    );
  }

  async function load(): Promise<void> {
    inFlight?.abort();
    inFlight = new AbortController();
    replaceChildren(tbody, ...skeletonRows());

    try {
      const page = await api.patients.list(
        { q: query, limit: PAGE_SIZE },
        inFlight.signal,
      );

      if (page.items.length === 0) {
        replaceChildren(
          tbody,
          el(
            'tr',
            {},
            el(
              'td',
              { colSpan: TABLE_COLUMNS },
              el(
                'div',
                { class: 'state' },
                el(
                  'div',
                  { class: 'state-icon' },
                  icon(query ? 'search' : 'users', { class: 'icon--lg' }),
                ),
                el(
                  'p',
                  { class: 'state-title' },
                  query ? 'Пациенты не найдены' : 'Пациентов пока нет',
                ),
                el(
                  'p',
                  { class: 'state-body' },
                  query
                    ? 'Попробуйте другое имя, телефон или номер карты.'
                    : 'Добавьте первого пациента, чтобы связывать с ним снимки.',
                ),
                // An empty state without a way out is just a dead end.
                query
                  ? el(
                      'button',
                      {
                        class: 'btn btn--sm',
                        type: 'button',
                        onclick: () => {
                          if (searchInput) searchInput.value = '';
                          query = '';
                          void load();
                        },
                      },
                      'Сбросить поиск',
                    )
                  : el(
                      'button',
                      { class: 'btn btn--sm btn--primary', type: 'button', onclick: () => openDialog() },
                      icon('plus', { class: 'icon--sm' }),
                      'Добавить пациента',
                    ),
              ),
            ),
          ),
        );
      } else {
        replaceChildren(tbody, ...page.items.map(row));
      }

      if (summary) {
        summary.textContent = page.meta.total > 0 ? `Всего: ${page.meta.total}` : '';
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      notifyError(error);
      replaceChildren(
        tbody,
        el(
          'tr',
          {},
          el(
            'td',
            { colSpan: TABLE_COLUMNS },
            el(
              'div',
              { class: 'state state--error' },
              el('div', { class: 'state-icon' }, icon('alert', { class: 'icon--lg' })),
              el('p', { class: 'state-title' }, 'Не удалось загрузить список'),
              el(
                'p',
                { class: 'state-body' },
                'Проверьте соединение — данные пациентов не были изменены.',
              ),
              el(
                'button',
                { class: 'btn btn--sm', type: 'button', onclick: () => void load() },
                icon('refresh', { class: 'icon--sm' }),
                'Повторить',
              ),
            ),
          ),
        ),
      );
    }
  }

  function openDialog(patient?: PatientSummary): void {
    if (!dialog) return;
    editingId = patient?.id ?? null;

    const form = must<HTMLFormElement>('form', dialog);
    form.reset();

    const title = maybe('[data-dialog-title]', dialog);
    if (title) title.textContent = patient ? 'Изменить пациента' : 'Новый пациент';

    if (patient) {
      setValue(form, 'fullName', patient.fullName);
      setValue(form, 'phone', patient.phone ?? '');
      setValue(form, 'email', patient.email ?? '');
      setValue(form, 'dateOfBirth', patient.dateOfBirth ?? '');
      setValue(form, 'medicalRecordNumber', patient.medicalRecordNumber ?? '');
      setValue(form, 'notes', patient.notes ?? '');
    }

    dialog.showModal();
    maybe<HTMLInputElement>('[name="fullName"]', form)?.focus();
  }

  function setValue(form: HTMLFormElement, name: string, value: string): void {
    const field = form.elements.namedItem(name);
    if (field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement) {
      field.value = value;
    }
  }

  if (dialog) {
    enhanceForm(must<HTMLFormElement>('form', dialog), {
      validate: (values) =>
        text(values, 'fullName').length < 2 ? { fullName: 'Укажите ФИО пациента' } : null,
      onSubmit: async (values) => {
        const payload = {
          fullName: text(values, 'fullName'),
          phone: optionalText(values, 'phone'),
          email: optionalText(values, 'email'),
          dateOfBirth: optionalText(values, 'dateOfBirth'),
          medicalRecordNumber: optionalText(values, 'medicalRecordNumber'),
          notes: optionalText(values, 'notes'),
        };
        return editingId === null
          ? api.patients.create(payload)
          : api.patients.update(editingId, payload);
      },
      onSuccess: () => {
        dialog.close();
        notify.success(editingId === null ? 'Пациент добавлен' : 'Изменения сохранены');
        void load();
      },
    });

    for (const closer of dialog.querySelectorAll<HTMLButtonElement>('[data-close-dialog]')) {
      on(closer, 'click', () => dialog.close());
    }
  }

  if (addButton) on(addButton, 'click', () => openDialog());

  delegate(tbody, 'click', '[data-action]', (_event, target) => {
    const id = Number(target.closest('tr')?.dataset['patientId']);
    if (!Number.isFinite(id)) return;

    if (target.dataset['action'] === 'edit') {
      void (async (): Promise<void> => {
        try {
          const patient = await api.patients.get(id);
          openDialog({
            ...patient,
            studyCount: 0,
            lastStudyAt: null,
            scanCount: 0,
            openPlanItems: 0,
          });
        } catch (error) {
          notifyError(error);
        }
      })();
      return;
    }

    if (target.dataset['action'] === 'archive') {
      void (async (): Promise<void> => {
        try {
          await api.patients.archive(id);
          void load();
          // Undo instead of a confirm dialog: archiving is reversible, so
          // interrupting every action to ask is worse than offering a way back.
          notify.info('Пациент перемещён в архив', {
            action: {
              label: 'Отменить',
              onClick: () => {
                void api.patients
                  .restore(id)
                  .then(() => {
                    notify.success('Восстановлено');
                    void load();
                  })
                  .catch(notifyError);
              },
            },
          });
        } catch (error) {
          if (error instanceof ApiError && error.code === 'permission_denied') {
            notify.error('Недостаточно прав для архивирования.');
            return;
          }
          notifyError(error);
        }
      })();
    }
  });

  if (searchInput) {
    on(
      searchInput,
      'input',
      debounce(() => {
        query = searchInput.value.trim();
        void load();
      }, 250),
    );
  }

  void load();
}
