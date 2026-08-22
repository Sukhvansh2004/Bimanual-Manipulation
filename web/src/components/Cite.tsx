import { acknowledgements, bibtex, paper } from '../content'
import { CopyButton } from './primitives'

export default function Cite() {
  return (
    <section id="cite" className="band-alt">
      <div className="wrap">
        <div className="cite-grid">
          <div>
            <div className="eyebrow">Citation</div>
            <h2 className="h2">BibTeX</h2>
            <p className="lede" style={{ marginBottom: 22 }}>
              The paper is under review; this entry will be updated with the final venue details on
              acceptance.
            </p>
            <div className="copy-wrap">
              <pre>
                <code>{bibtex}</code>
              </pre>
              <CopyButton text={bibtex} label="Copy BibTeX" />
            </div>
          </div>

          <aside className="aside-card">
            <h3>Built on</h3>
            <div className="ack">
              {acknowledgements.builtOn.map((b) => (
                <div className="ack-item" key={b.name}>
                  <a href={b.url} target="_blank" rel="noreferrer">
                    {b.name}
                  </a>
                  <span>{b.what}</span>
                </div>
              ))}
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--ink-3)', marginTop: 18 }}>
              Questions about the implementation are best raised as issues on the{' '}
              <a href={paper.links.code} target="_blank" rel="noreferrer">
                GitHub repository
              </a>
              .
            </p>
          </aside>
        </div>
      </div>
    </section>
  )
}
