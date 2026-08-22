import { useState } from 'react'
import { codebase } from '../content'
import { CopyButton, IconGithub } from './primitives'

/** Tints the leading binary and any --flags so long commands stay scannable. */
function Shell({ cmd }: { cmd: string }) {
  return (
    <pre>
      <code>
        {cmd.split('\n').map((line, i) => {
          const parts = line.split(/(\s+)/)
          return (
            <span key={i}>
              {parts.map((tok, j) => {
                if (j === 0 && i === 0) return <span className="tok-cmd" key={j}>{tok}</span>
                if (tok.startsWith('--')) return <span className="tok-flag" key={j}>{tok}</span>
                return <span key={j}>{tok}</span>
              })}
              {i < cmd.split('\n').length - 1 ? '\n' : ''}
            </span>
          )
        })}
      </code>
    </pre>
  )
}

export default function Code() {
  const [tab, setTab] = useState(0)
  const step = codebase.quickstart[tab]

  return (
    <section id="code">
      <div className="wrap">
        <div className="section-head">
          <div className="eyebrow">Code</div>
          <h2 className="h2">The full pipeline is open source</h2>
          <p className="lede">
            Perception, grasp generation, SE(3) and joint-space planning, the assignment DP, the
            handover primitives and the evaluation harness behind both tables — all in one MIT-licensed
            repository, installable without sudo.
          </p>

          <div className="code-stats">
            {codebase.stats.map((s) => (
              <div className="code-stat" key={s.l}>
                <b>{s.v}</b>
                <span>{s.l}</span>
              </div>
            ))}
          </div>

          <div className="actions" style={{ marginTop: 24 }}>
            <a className="btn" href={codebase.repo} target="_blank" rel="noreferrer">
              <IconGithub /> Sukhvansh2004/Bimanual-Manipulation
            </a>
          </div>
        </div>

        <div className="mod-grid">
          {codebase.modules.map((m) => (
            <article className="mod" key={m.path}>
              <code className="path">
                <span className="path-prefix">src/bimanual_manipulation/</span>
                {m.path}
              </code>
              <p>{m.what}</p>
              <div className="mod-files">
                {m.files.map((f) => (
                  <code key={f}>{f}</code>
                ))}
              </div>
            </article>
          ))}
        </div>

        <div className="qs">
          <div className="qs-tabs" role="tablist" aria-label="Quickstart steps">
            {codebase.quickstart.map((q, i) => (
              <button
                key={q.label}
                type="button"
                role="tab"
                id={`qs-tab-${i}`}
                aria-selected={i === tab}
                aria-controls={`qs-panel-${i}`}
                className="qs-tab"
                onClick={() => setTab(i)}
              >
                {i + 1}. {q.label}
              </button>
            ))}
          </div>
          <div
            className="qs-panel"
            role="tabpanel"
            id={`qs-panel-${tab}`}
            aria-labelledby={`qs-tab-${tab}`}
          >
            <p className="qs-note">{step.note}</p>
            <div className="copy-wrap">
              <Shell cmd={step.cmd} />
              <CopyButton text={step.cmd} />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
