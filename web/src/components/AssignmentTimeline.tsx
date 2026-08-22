import { Fragment } from 'react'
import { assignmentRun as run } from '../content'

/**
 * The per-segment arm assignment from a real planner run shipped in the repo
 * (results/iter_1/summary.json). Each cell is one trajectory segment; the
 * marker is the waypoint where the two arms exchange the object.
 */
export default function AssignmentTimeline() {
  const { sequence, handoverAt, robots } = run
  const used = [...new Set(sequence)]

  return (
    <div className="timeline-card">
      <div className="timeline-head">
        <div>
          <h3 className="h3">What the planner actually returns</h3>
          <p>
            {run.experiment} · {run.config} · {run.nSegments} segments
          </p>
        </div>
        <code className="timeline-src">results/iter_1/summary.json</code>
      </div>

      <div className="timeline-track" role="img" aria-label={ariaLabel()}>
        {sequence.map((robot, i) => (
          <Fragment key={i}>
            {i === handoverAt && <span className="tl-hand" title="Handover" />}
            <span
              className="tl-seg"
              style={{ background: robots[robot].color }}
              title={`Segment ${i} · ${robots[robot].name}`}
            />
          </Fragment>
        ))}
      </div>

      <div className="tl-axis">
        <span>segment 0</span>
        <span>handover at {handoverAt}</span>
        <span>segment {sequence.length - 1}</span>
      </div>

      <div className="tl-legend">
        {used.map((r) => (
          <span key={r}>
            <i className="swatch" style={{ background: robots[r].color }} />
            {robots[r].name}
          </span>
        ))}
        <span>
          <i className="swatch" style={{ background: 'var(--warn)', width: 3 }} />
          Handover (suction → parallel-jaw)
        </span>
      </div>

      <dl className="tl-stats">
        <div className="tl-stat">
          <dt>Plan time</dt>
          <dd>{run.planTime.toFixed(1)} s</dd>
        </div>
        <div className="tl-stat">
          <dt>Execution time</dt>
          <dd>{run.execTime.toFixed(1)} s</dd>
        </div>
        <div className="tl-stat">
          <dt>Total cost</dt>
          <dd>{run.totalCost.toFixed(0)}</dd>
        </div>
        <div className="tl-stat">
          <dt>Mean manipulability</dt>
          <dd>{run.meanManipulability.toFixed(3)}</dd>
        </div>
        <div className="tl-stat">
          <dt>Object tracking error</dt>
          <dd>{(run.trackingError * 1000).toFixed(1)} mm</dd>
        </div>
      </dl>
    </div>
  )

  function ariaLabel(): string {
    const first = robots[sequence[0]].name
    const last = robots[sequence[sequence.length - 1]].name
    return `Assignment timeline: ${first} executes segments 0 to ${handoverAt - 1}, then a handover at segment ${handoverAt}, then ${last} executes the remaining segments through ${sequence.length - 1}.`
  }
}
