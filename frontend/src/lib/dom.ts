/**
 * DOM helpers.
 *
 * `el()` builds nodes by setting properties rather than concatenating HTML,
 * which makes injection structurally impossible.
 */

type Falsy = null | undefined | false;
export type Child = Node | string | number | Falsy;

type Attrs<K extends keyof HTMLElementTagNameMap> = Partial<
  Omit<HTMLElementTagNameMap[K], 'children' | 'style' | 'dataset' | 'classList'>
> & {
  readonly class?: string;
  readonly dataset?: Readonly<Record<string, string | number | boolean>>;
  readonly style?: Readonly<Record<string, string>>;
  readonly aria?: Readonly<Record<string, string | number | boolean>>;
};

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs<K> = {},
  ...children: readonly Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  const { class: className, dataset, style, aria, ...rest } = attrs;

  if (className) node.className = className;

  for (const [key, value] of Object.entries(dataset ?? {})) {
    node.dataset[key] = String(value);
  }
  for (const [key, value] of Object.entries(style ?? {})) {
    node.style.setProperty(key, value);
  }
  for (const [key, value] of Object.entries(aria ?? {})) {
    node.setAttribute(key.startsWith('aria-') ? key : `aria-${key}`, String(value));
  }
  Object.assign(node, rest);

  append(node, children);
  return node;
}

export function append(parent: Element, children: readonly Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : String(child));
  }
}

export function replaceChildren(parent: Element, ...children: readonly Child[]): void {
  parent.replaceChildren();
  append(parent, children);
}

/** Query one element, throwing if absent — a missing hook is a bug, not a case to handle. */
export function must<T extends Element = HTMLElement>(
  selector: string,
  scope: ParentNode = document,
): T {
  const found = scope.querySelector<T>(selector);
  if (!found) throw new Error(`Required element not found: ${selector}`);
  return found;
}

/** Query one element, returning null if absent. */
export function maybe<T extends Element = HTMLElement>(
  selector: string,
  scope: ParentNode = document,
): T | null {
  return scope.querySelector<T>(selector);
}

export function all<T extends Element = HTMLElement>(
  selector: string,
  scope: ParentNode = document,
): readonly T[] {
  return Array.from(scope.querySelectorAll<T>(selector));
}

export function on<K extends keyof HTMLElementEventMap>(
  target: EventTarget,
  type: K,
  handler: (event: HTMLElementEventMap[K]) => void,
  options?: AddEventListenerOptions,
): () => void {
  const listener = handler as EventListener;
  target.addEventListener(type, listener, options);
  return () => target.removeEventListener(type, listener, options);
}

/**
 * Delegated listener. One handler on a container instead of N on rows, so
 * re-rendering a list never leaks listeners or needs re-binding.
 */
export function delegate<K extends keyof HTMLElementEventMap>(
  container: Element,
  type: K,
  selector: string,
  handler: (event: HTMLElementEventMap[K], target: HTMLElement) => void,
): () => void {
  return on(
    container,
    type,
    (event) => {
      const origin = event.target;
      if (!(origin instanceof Element)) return;
      const match = origin.closest<HTMLElement>(selector);
      if (match && container.contains(match)) handler(event, match);
    },
    {},
  );
}

/**
 * Whether the user has asked the system for less animation.
 *
 * Read per call rather than cached: the setting can change mid-session, and a
 * dashboard left open on a wall display would otherwise keep animating.
 */
export function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function debounce<A extends readonly unknown[]>(
  fn: (...args: A) => void,
  waitMs: number,
): (...args: A) => void {
  let timer: number | undefined;
  return (...args: A): void => {
    if (timer !== undefined) window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), waitMs);
  };
}

/** Toggle a button's busy state without shifting layout. */
export function setBusy(button: HTMLButtonElement, busy: boolean): void {
  button.dataset['loading'] = String(busy);
  button.disabled = busy;
  button.setAttribute('aria-busy', String(busy));
}

/** Announce a message to screen readers via a shared live region. */
export function announce(message: string): void {
  let region = document.getElementById('sr-live');
  if (!region) {
    region = el('div', {
      id: 'sr-live',
      class: 'visually-hidden',
      aria: { live: 'polite', atomic: 'true' },
    });
    document.body.append(region);
  }
  // Clearing first guarantees repeat messages are re-announced.
  region.textContent = '';
  window.setTimeout(() => {
    region.textContent = message;
  }, 60);
}
