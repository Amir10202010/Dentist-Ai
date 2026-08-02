/**
 * Client-side half of the icon system.
 *
 * Geometry lives in exactly one place — the `<symbol>` sprite emitted by
 * `templates/partials/icons.html` — and both halves of the app reference it by
 * id. This module never carries path data, so a client-rendered icon and a
 * server-rendered one cannot drift.
 *
 * `IconName` is a literal union rather than `string`: a `<use>` pointing at a
 * missing id renders nothing at all, so a typo would only surface as a blank
 * space in the UI.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const XLINK_NS = 'http://www.w3.org/1999/xlink';

export type IconName =
  | 'dashboard'
  | 'scan'
  | 'users'
  | 'settings'
  | 'sun'
  | 'moon'
  | 'logout'
  | 'menu'
  | 'close'
  | 'upload'
  | 'zoom-in'
  | 'zoom-out'
  | 'maximize'
  | 'eye'
  | 'eye-off'
  | 'check'
  | 'search'
  | 'alert'
  | 'alert-circle'
  | 'chevron-right'
  | 'chevron-down'
  | 'chevron-left'
  | 'arrow-left'
  | 'arrow-right'
  | 'play'
  | 'quote'
  | 'plus'
  | 'download'
  | 'trash'
  | 'more'
  | 'trending-up'
  | 'trending-down'
  | 'activity'
  | 'shield'
  | 'sparkles'
  | 'file-text'
  | 'clock'
  | 'calendar'
  | 'inbox'
  | 'filter'
  | 'sort'
  | 'pencil'
  | 'archive'
  | 'building'
  | 'keyboard'
  | 'lock'
  | 'user'
  | 'check-circle'
  | 'x-circle'
  | 'image-off'
  | 'phone'
  | 'mail'
  | 'stethoscope'
  | 'chart'
  | 'refresh'
  | 'panel-left'
  | 'command'
  | 'layers'
  | 'target'
  | 'external'
  | 'cube'
  | 'clipboard'
  | 'user-plus'
  | 'tooth'
  | 'slice';

export interface IconOptions {
  /** Extra classes, e.g. `icon--lg`. */
  readonly class?: string;
}

/**
 * Builds `<svg class="icon"><use href="#i-name"/></svg>`.
 *
 * Always decorative. An icon inside a control is redundant to the control's
 * own accessible name, and an icon standing alone needs that name on the
 * interactive element — not on the glyph.
 */
export function icon(name: IconName, options: IconOptions = {}): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', options.class ? `icon ${options.class}` : 'icon');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');

  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', `#i-${name}`);
  // Safari below 16 ignores a plain `href` on `<use>`; the xlink form is the
  // only one it honours, and modern engines still accept it.
  use.setAttributeNS(XLINK_NS, 'xlink:href', `#i-${name}`);

  svg.append(use);
  return svg;
}
