import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// A GitHub Pages project site is served from https://<user>.github.io/<repo>/,
// so every asset URL needs the repo name as its base path. Override with
// VITE_BASE=/ when deploying to a user/org page or a custom domain.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/Bimanual-Manipulation/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
  },
})
