import { methodRows } from '../content'
import { useInView } from '../hooks/useInView'
import type { MethodRow } from '../types'

interface BarSpec {
  key: string
  label: string
  value: number
  ours: boolean
  display: string
}

function BarChart({
  title,
  sub,
  bars,
  max,
  note,
}: {
  title: string
  sub: string
  bars: BarSpec[]
  max: number
  note: string
}) {
  const [ref, seen] = useInView<HTMLDivElement>()

  return (
    <div className="chart-card" ref={ref}>
      <h3 className="h3">{title}</h3>
      <p className="chart-sub">{sub}</p>

      <div className="bars">
        {bars.map((b) => (
          <div
            className="bar-row"
            key={b.key}
            data-ours={b.ours}
            data-zero={b.value === 0}
          >
            <div className="bar-label">
              <span>{b.label}</span>
              <span className="bar-value">{b.display}</span>
            </div>
            <div
              className="bar-track"
              role="img"
              aria-label={`${b.label}: ${b.display}`}
            >
              <div
                className="bar-fill"
                style={{ width: seen ? `${(b.value / max) * 100}%` : '0%' }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="chart-note">{note}</p>
    </div>
  )
}

const shortName = (r: MethodRow): string =>
  r.method.replace('Proposed — ', 'Ours — ').replace(' (top-k grasps)', '')

export default function Charts() {
  const successBars: BarSpec[] = methodRows.map((r) => ({
    key: r.method,
    label: shortName(r),
    value: r.success,
    ours: Boolean(r.ours),
    display: `${r.success.toFixed(1)}%`,
  }))

  const timeRows = methodRows.filter((r): r is MethodRow & { plan: number } => r.plan !== null)
  const timeBars: BarSpec[] = timeRows.map((r) => ({
    key: r.method,
    label: shortName(r),
    value: r.plan,
    ours: Boolean(r.ours),
    display: `${r.plan.toFixed(0)} s`,
  }))

  return (
    <div className="chart-grid">
      <BarChart
        title="Task success"
        sub="End-to-end completion rate over 520 simulated runs. Higher is better."
        bars={successBars}
        max={100}
        note="Single-robot execution never completes a task, which is what makes these tasks a fair test of heterogeneous coordination. The greedy baseline reaches 20.0% — the gap to 72.5% is the value of global look-ahead."
      />
      <BarChart
        title="Planning time"
        sub="Mean assignment computation over the sampled grasp trajectories. Lower is better."
        bars={timeBars}
        max={Math.max(...timeBars.map((b) => b.value))}
        note="Grasp diversity is a speed feature: without DPP filtering, near-identical grasps inflate the DP state space to 1016 s. Beam 10 / stride 3 brings exact DP's 533 s down to 91 s for a 7.5-point success cost."
      />
    </div>
  )
}
