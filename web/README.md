# Project page

React + TypeScript + Vite site for **Embodiment-Aware Planning for Heterogeneous Bimanual
Manipulation**, deployed to GitHub Pages.

Live: `https://sukhvansh2004.github.io/Bimanual-Manipulation/`

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173/Bimanual-Manipulation/
```

| Script | What it does |
|---|---|
| `npm run dev` | Dev server with HMR |
| `npm run build` | Type-check, then build to `dist/` |
| `npm run typecheck` | `tsc` only |
| `npm run preview` | Serve the production build locally |

## Editing content

**All copy and every number live in [`src/content.tsx`](src/content.tsx)** — authors, abstract,
method steps and their LaTeX, task descriptions, both result tables, ablations and the BibTeX
entry. The components under `src/components/` only render it. `src/types.ts` types every content
structure, so a mistyped field fails `npm run typecheck` rather than the page.

`paper.links.arxiv` points to the arXiv preprint ([2610.23551](https://arxiv.org/abs/2610.23551));
the header and footer show an arXiv link whenever it is set.

## Assets

`public/assets/` is committed and served as-is:

| File | Source |
|---|---|
| `paper.pdf` | the arXiv preprint (2610.23551v1) |
| `supplementary_video.mp4` + `video_poster.jpg` | the submission video (720p, 4:02) |
| `fig1_teaser.jpg`, `fig2_framework.jpg`, `fig3_tasks.jpg` | figures extracted from the PDF |
| `tasks/task{A–D}_{1–5}.jpg` | per-task frames cropped out of Figure 3 |

To regenerate the figures after a paper revision, extract the embedded images from the PDF
(pages 1, 4 and 6) and re-crop the Figure 3 filmstrip — Figure 2 has an alpha channel and must be
flattened onto white, since it is displayed inside a light figure card.

## CI and deployment

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) type-checks, lints and builds this site on
every push and pull request to `main`, alongside the Python test suite.

[`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml) publishes on every
push to `main` that touches `web/`. One-time setup: **Settings → Pages → Build and deployment →
Source: GitHub Actions**.

The site is served from `/<repo>/`, which `vite.config.ts` sets as the Vite `base`. For a user page
or a custom domain, build with `VITE_BASE=/`.

## Design notes

- Single dark theme; the palette is fixed in `:root` in `src/styles.css`.
- Chart and task accent colours are the dark-surface categorical steps `#3987e5` (suction),
  `#199e70` (parallel-jaw), `#e66767` (infeasible/handover) and `#9085e9`, validated for
  colour-vision-deficient separation and ≥3:1 contrast against the `#12151c` surface. Every
  colour-coded element is also labelled in text.
- Equations render with KaTeX, bundled locally — the page makes no external network requests.
- Paper figures sit in light "figure frames" because they were authored for a white page.
