/**
 * Marketing site entry (every public page except `/`).
 *
 * Every marketing page is server-rendered and legible with JavaScript
 * disabled; this bundle only adds progressive enhancement.
 *
 * Reveal-on-scroll is a scroll-driven CSS animation instead (the
 * `[data-reveal]` rules in `marketing.css`), so a bundle that never loads
 * cannot leave sections below the fold invisible.
 */

import '../styles/tokens.css';
import '../styles/base.css';
import '../styles/components.css';
import '../styles/marketing.css';

import { initSiteChrome, onReady } from '../lib/site-chrome';

onReady(initSiteChrome);
