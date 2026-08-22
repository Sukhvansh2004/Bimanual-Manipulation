import { paper, highlights, teaser } from '../content'
import { Figure, IconPdf, IconPlay, IconCode, IconQuote } from './primitives'

function Authors() {
  const multiAffil = paper.affiliations.length > 1

  return (
    <>
      <div className="authors">
        {paper.authors.map((a, i) => {
          const label = (
            <>
              {a.name}
              {multiAffil && <sup>{a.affiliation}</sup>}
            </>
          )
          return (
            <span
              className={`author${a.placeholder ? ' author-todo' : ''}`}
              key={`${a.name}-${i}`}
              title={a.placeholder ? 'Placeholder — replace in src/content.tsx' : undefined}
            >
              {a.url && !a.placeholder ? (
                <a href={a.url} target="_blank" rel="noreferrer">
                  {label}
                </a>
              ) : (
                label
              )}
            </span>
          )
        })}
      </div>
      <div className="affils">
        {paper.affiliations.map((f) => (
          <span key={f.id} style={{ marginRight: 16 }}>
            {multiAffil && <sup>{f.id}</sup>}
            {f.name}
          </span>
        ))}
      </div>
    </>
  )
}

export default function Hero() {
  return (
    <header className="hero" id="top">
      <div className="wrap">
        <span className="venue">{paper.venue}</span>
        <h1>{paper.title}</h1>
        <p className="hero-tagline">{paper.tagline}</p>

        <Authors />

        <div className="actions">
          <a className="btn btn-primary" href={paper.links.pdf} target="_blank" rel="noreferrer">
            <IconPdf /> Paper (PDF)
          </a>
          <a className="btn" href="#video">
            <IconPlay /> Video
          </a>
          <a className="btn" href={paper.links.code} target="_blank" rel="noreferrer">
            <IconCode /> Code
          </a>
          <a className="btn" href="#cite">
            <IconQuote /> BibTeX
          </a>
        </div>
      </div>

      <div className="wrap" style={{ marginTop: 'clamp(40px, 5vw, 64px)' }}>
        <Figure src={teaser.src} alt={teaser.alt} caption={teaser.caption} />

        <div className="stats">
          {highlights.map((h) => (
            <div className="stat" key={h.label}>
              <div className="stat-v">
                {h.value}
                {h.unit && <span>{h.unit}</span>}
              </div>
              <div className="stat-l">{h.label}</div>
              <div className="stat-n">{h.note}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  )
}
