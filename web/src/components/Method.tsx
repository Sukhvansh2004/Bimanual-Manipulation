import { useState } from 'react'
import { framework, pipeline, setup } from '../content'
import { Figure, IconChevron, Math } from './primitives'
import AssignmentTimeline from './AssignmentTimeline'

export default function Method() {
  // The first step starts open so the section never reads as an empty list.
  const [open, setOpen] = useState<Set<string>>(new Set([pipeline[0].id]))

  const toggle = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <section id="method" className="band-alt">
      <div className="wrap">
        <div className="section-head">
          <div className="eyebrow">Method</div>
          <h2 className="h2">From one demonstration to a globally optimal plan</h2>
          <p className="lede">
            The demonstration says what the object must do. The planner decides which arm does it,
            when the two arms must meet, and what that meeting costs — solved once, over the whole
            horizon, rather than segment by segment.
          </p>
        </div>

        <Figure src={framework.src} alt={framework.alt} caption={framework.caption} />

        <div className="steps">
          {pipeline.map((step, i) => {
            const isOpen = open.has(step.id)
            return (
              <article className="step" key={step.id} data-open={isOpen}>
                <button
                  type="button"
                  className="step-btn"
                  onClick={() => toggle(step.id)}
                  aria-expanded={isOpen}
                  aria-controls={`step-body-${step.id}`}
                >
                  <span className="step-num">{i + 1}</span>
                  <span>
                    <span className="step-title">{step.title}</span>
                    <span className="step-lede">{step.lede}</span>
                  </span>
                  <span className="step-chev">
                    <IconChevron />
                  </span>
                </button>

                {isOpen && (
                  <div className="step-body" id={`step-body-${step.id}`}>
                    {step.body.map((para) => (
                      <p key={para.slice(0, 40)}>{para}</p>
                    ))}

                    {step.math && (
                      <div className="math-block">
                        {step.math.map((m) => (
                          <Math key={m.label} tex={m.tex} label={m.label} />
                        ))}
                      </div>
                    )}

                    <div className="tags">
                      {step.tags.map((t) => (
                        <span className="tag" key={t}>
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </article>
            )
          })}
        </div>

        <AssignmentTimeline />

        <div style={{ marginTop: 'clamp(40px, 5vw, 60px)' }}>
          <div className="eyebrow">Experimental setup</div>
          <dl className="spec">
            {setup.map((s) => (
              <div key={s.k}>
                <dt>{s.k}</dt>
                <dd>{s.v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  )
}
