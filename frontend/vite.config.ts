import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

/**
 * Builds straight into the Python package's static directory.
 *
 * Output is content-hashed and read back through `.vite/manifest.json` by
 * `web/templating.py`, which is what lets the server serve every asset with a
 * one-year immutable cache header — the filename *is* the cache key.
 */
export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  base: '/static/dist/',
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: fileURLToPath(new URL('../src/dentist_ai/static/dist', import.meta.url)),
    emptyOutDir: true,
    manifest: true,
    // Long-lived browser support without shipping polyfills; every target
    // here has native ES modules, CSS nesting fallbacks and dialog support.
    target: 'es2022',
    cssCodeSplit: true,
    reportCompressedSize: false,
    rollupOptions: {
      input: {
        marketing: fileURLToPath(new URL('./src/entries/marketing.ts', import.meta.url)),
        // The landing page is its own entry: it carries the mockups, bento
        // grid and aurora field, none of which the prose pages need.
        landing: fileURLToPath(new URL('./src/entries/landing.ts', import.meta.url)),
        auth: fileURLToPath(new URL('./src/entries/auth.ts', import.meta.url)),
        app: fileURLToPath(new URL('./src/entries/app.ts', import.meta.url)),
      },
      output: {
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash][extname]',
      },
    },
  },
});
