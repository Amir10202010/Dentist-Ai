/** Locale-aware formatting. Uses Intl throughout — no hand-rolled date maths. */

import type { Severity } from './types';

const locale = document.documentElement.lang || 'ru';

const dateFormatter = new Intl.DateTimeFormat(locale, {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
});

const dateTimeFormatter = new Intl.DateTimeFormat(locale, {
  day: 'numeric',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
});

const relativeFormatter = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

const percentFormatter = new Intl.NumberFormat(locale, {
  style: 'percent',
  maximumFractionDigits: 0,
});

const numberFormatter = new Intl.NumberFormat(locale);

/**
 * Parse a server timestamp.
 *
 * Every timestamp the API sends now carries an explicit offset: the
 * `UtcDateTime` column type in `db/base.py` re-attaches UTC on read, so SQLite
 * — which hands SQLAlchemy back naive datetimes — serialises identically to
 * Postgres.
 *
 * Kept as defence in depth rather than deleted. `new Date()` reads an
 * offset-less ISO string as *local* time, so one timestamp escaping without an
 * offset would render a study uploaded an hour ago as "6 hours ago" in Almaty
 * and "in 5 hours" in São Paulo — a wrong answer that looks plausible. One
 * regex is a cheap guard against a silent, hard-to-spot class of bug.
 */
export function parseInstant(iso: string): Date {
  return new Date(/[Z+]|-\d{2}:\d{2}$/.test(iso.slice(10)) ? iso : `${iso}Z`);
}

export function formatDate(iso: string | null): string {
  if (!iso) return '—';
  return dateFormatter.format(parseInstant(iso));
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return '—';
  return dateTimeFormatter.format(parseInstant(iso));
}

const timeFormatter = new Intl.DateTimeFormat(locale, {
  hour: '2-digit',
  minute: '2-digit',
});

export function formatTime(iso: string | null): string {
  if (!iso) return '—';
  return timeFormatter.format(parseInstant(iso));
}

/** Whole days elapsed since `iso`, floored at zero. */
export function daysSince(iso: string): number {
  const elapsed = Date.now() - parseInstant(iso).getTime();
  return Math.max(0, Math.floor(elapsed / 86_400_000));
}

/**
 * Russian numeric agreement, mirroring `_plural` in `services/analytics.py`.
 *
 * Duplicated rather than shared because the two halves pluralise different
 * strings: the server writes insight copy, the client labels counters it
 * computes itself.
 */
export function plural(count: number, one: string, few: string, many: string): string {
  const tens = count % 100;
  if (tens >= 11 && tens <= 14) return many;
  const ones = count % 10;
  if (ones === 1) return one;
  if (ones >= 2 && ones <= 4) return few;
  return many;
}

/** A signed percentage for a ratio delta: `0.12` → `+12%`. */
export function formatChange(ratio: number): string {
  const percent = Math.round(Math.abs(ratio) * 100);
  // U+2212 MINUS SIGN, not a hyphen: it aligns with the digits at the same
  // optical weight as the plus.
  return `${ratio >= 0 ? '+' : '−'}${percent}%`;
}

const RELATIVE_UNITS: readonly (readonly [Intl.RelativeTimeFormatUnit, number])[] = [
  ['year', 31_536_000_000],
  ['month', 2_592_000_000],
  ['week', 604_800_000],
  ['day', 86_400_000],
  ['hour', 3_600_000],
  ['minute', 60_000],
];

export function formatRelative(iso: string | null): string {
  if (!iso) return '—';
  const elapsed = parseInstant(iso).getTime() - Date.now();
  const magnitude = Math.abs(elapsed);

  for (const [unit, ms] of RELATIVE_UNITS) {
    if (magnitude >= ms) {
      return relativeFormatter.format(Math.round(elapsed / ms), unit);
    }
  }
  return relativeFormatter.format(0, 'minute');
}

export function formatPercent(ratio: number): string {
  return percentFormatter.format(ratio);
}

export function formatNumber(value: number): string {
  return numberFormatter.format(value);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  const units = ['КБ', 'МБ', 'ГБ'] as const;
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(1)} ${units[index] ?? 'ГБ'}`;
}

export function formatDuration(ms: number | null): string {
  if (ms === null) return '—';
  return ms < 1000 ? `${ms} мс` : `${(ms / 1000).toFixed(1)} с`;
}

export const SEVERITY_ORDER: readonly Severity[] = [
  'critical',
  'high',
  'medium',
  'low',
  'info',
];

/** Initials for an avatar, safe for any script. */
export function initialsOf(name: string): string {
  const parts = name.split(/\s+/).filter(Boolean).slice(0, 2);
  return parts.map((part) => part.charAt(0).toUpperCase()).join('') || '?';
}
