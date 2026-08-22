import { Fragment } from 'react'
import { ablations, methodRows, tasks } from '../content'
import Charts from './Charts'

const num = (v: number | null, digits = 1, suffix = ''): string =>
  v === null ? '—' : `${v.toFixed(digits)}${suffix}`

/** Keeps the from → to pairs consistent: seconds whole, everything else to 1dp. */
const delta = (v: number, unit: string): string =>
  `${v.toFixed(unit.trim() === 's' ? 0 : 1)}${unit}`

function MethodTable() {
  // Precomputed so the render pass stays free of mutable bookkeeping.
  const startsGroup = methodRows.map((r, i) => i === 0 || methodRows[i - 1].group !== r.group)

  return (
    <div>
      <p className="table-caption">
        <b>Table I — overall performance across methods.</b> Task success is the end-to-end
        completion rate (assignment success × valid execution success). Handover measures the
        success of valid transitions, excluding drops that occurred before the handover. Plan time
        is the assignment computation duration over sampled grasp trajectories; exec time is the
        task execution duration.
      </p>
      <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Assign</th>
            <th scope="col">Task success</th>
            <th scope="col">Handover</th>
            <th scope="col">Avg. switches</th>
            <th scope="col">Plan time</th>
            <th scope="col">Exec time</th>
          </tr>
        </thead>
        <tbody>
          {methodRows.map((r, i) => {
            return (
              <Fragment key={r.method}>
                {startsGroup[i] && (
                  <tr className="row-group">
                    <td colSpan={7}>{r.group}</td>
                  </tr>
                )}
                <tr data-ours={r.ours ? 'true' : undefined}>
                  <th scope="row" style={{ fontWeight: r.ours ? 600 : 400 }}>
                    {r.method}
                  </th>
                  <td>{num(r.assign)}%</td>
                  <td className={r.success === 72.5 ? 'best-cell' : undefined}>
                    {num(r.success)}%
                  </td>
                  <td className={r.handover === 100 ? 'best-cell' : undefined}>
                    {r.handover === null ? '—' : `${num(r.handover)}%`}
                  </td>
                  <td>{num(r.switches, 1)}</td>
                  <td>{num(r.plan, 2, ' s')}</td>
                  <td>{num(r.exec, 1, ' s')}</td>
                </tr>
              </Fragment>
            )
          })}
        </tbody>
      </table>
      </div>
    </div>
  )
}

function TaskTable() {
  return (
    <div>
      <p className="table-caption">
        <b>Table II — per-task evaluation.</b> Breakdown of the same metrics for the practical
        beam 10 / stride 3 configuration.
      </p>
      <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Task</th>
            <th scope="col">Assign</th>
            <th scope="col">Task success</th>
            <th scope="col">Handover</th>
            <th scope="col">Avg. switches</th>
            <th scope="col">Plan time</th>
            <th scope="col">Exec time</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((t) => (
            <tr key={t.id}>
              <th scope="row">
                <span
                  className="swatch"
                  style={{ background: t.color, display: 'inline-block', marginRight: 9 }}
                  aria-hidden="true"
                />
                Task {t.id}: {t.name}
              </th>
              <td>{num(t.metrics.assign)}%</td>
              <td>{num(t.metrics.success)}%</td>
              <td className="best-cell">{num(t.metrics.handover)}%</td>
              <td>{num(t.metrics.switches)}</td>
              <td>{num(t.metrics.plan, 2, ' s')}</td>
              <td>{num(t.metrics.exec, 1, ' s')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  )
}

export default function Results() {
  return (
    <section id="results" className="band-alt">
      <div className="wrap">
        <div className="section-head">
          <div className="eyebrow">Results</div>
          <h2 className="h2">520 simulated runs, four tasks, six configurations</h2>
          <p className="lede">
            The exact dynamic program reaches 72.5% task success against the greedy baseline's
            20.0%, and every valid heterogeneous transition it plans succeeds. Beam search with a
            stride trades 7.5 points of success for a 5.9× speedup.
          </p>
        </div>

        <Charts />

        <div className="results-tables">
          <MethodTable />
          <TaskTable />
        </div>

        <div style={{ marginTop: 'clamp(44px, 5vw, 64px)' }}>
          <div className="eyebrow">Ablations</div>
          <h3 className="h2" style={{ fontSize: '1.4rem' }}>
            What each component is actually buying
          </h3>
          <div className="abl-grid">
            {ablations.map((a) => (
              <article className="abl" key={a.id}>
                <div className="abl-head">
                  <h3>{a.title}</h3>
                  <code className="knob">{a.knob}</code>
                </div>
                <p>{a.finding}</p>
                <p className="contrast">{a.contrast}</p>
                <div className="deltas">
                  {a.deltas.map((d) => (
                    <div className="delta" key={d.label}>
                      <span className="delta-l">{d.label}</span>
                      <span className="delta-v">
                        <span className="delta-from">{delta(d.from, d.unit)}</span>
                        <span className="delta-arrow" aria-hidden="true">
                          →
                        </span>
                        <span className="delta-to">{delta(d.to, d.unit)}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
