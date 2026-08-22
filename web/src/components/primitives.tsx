import { useEffect, useRef, useState } from 'react'
import type { ReactNode, SVGProps } from 'react'
import katex from 'katex'

/* ---------- icons (inline, so nothing is fetched at runtime) ---------- */

const stroke: SVGProps<SVGSVGElement> = {
  width: 16,
  height: 16,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.9,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
}

export const IconPdf = () => (
  <svg {...stroke}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6M9 15h6M9 11h2" />
  </svg>
)

export const IconPlay = () => (
  <svg {...stroke}>
    <path d="M6 4.5v15l13-7.5z" fill="currentColor" stroke="none" />
  </svg>
)

export const IconCode = () => (
  <svg {...stroke}>
    <path d="M16 18l6-6-6-6M8 6l-6 6 6 6" />
  </svg>
)

export const IconQuote = () => (
  <svg {...stroke}>
    <path d="M4 7h16M4 12h16M4 17h10" />
  </svg>
)

export const IconChevron = () => (
  <svg {...stroke} width={18} height={18}>
    <path d="M6 9l6 6 6-6" />
  </svg>
)

export const IconGithub = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 .5a12 12 0 0 0-3.79 23.4c.6.11.82-.26.82-.58v-2.2c-3.34.73-4.04-1.42-4.04-1.42-.55-1.39-1.34-1.76-1.34-1.76-1.1-.75.08-.73.08-.73 1.21.09 1.85 1.24 1.85 1.24 1.07 1.84 2.81 1.31 3.5 1 .1-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.11-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.65 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58A12 12 0 0 0 12 .5z" />
  </svg>
)

/* ---------- KaTeX ---------- */

export function Math({ tex, label }: { tex: string; label?: string }) {
  const html = katex.renderToString(tex, {
    displayMode: true,
    throwOnError: false,
    output: 'html',
  })
  return (
    <div className="math-row">
      {label && <div className="math-label">{label}</div>}
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  )
}

/* ---------- figure ---------- */

export function Figure({
  src,
  alt,
  caption,
}: {
  src: string
  alt: string
  caption?: ReactNode
}) {
  return (
    <figure className="figure">
      <div className="figure-frame">
        <img src={src} alt={alt} loading="lazy" />
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/* ---------- copy-to-clipboard ---------- */

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // The Clipboard API needs a secure context; fall back to a hidden selection.
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    setCopied(true)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setCopied(false), 1800)
  }

  return (
    <button type="button" className="copy-btn" onClick={copy} data-copied={copied}>
      {copied ? 'Copied' : label}
    </button>
  )
}
